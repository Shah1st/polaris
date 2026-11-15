"""Run the mapper on a TUM RGB-D sequence and compare the trajectory with ground truth.

    python scripts/eval_tum.py <sequence dir> [frame step] [model]

Reports the absolute trajectory error after similarity alignment (monocular
scale is arbitrary on a handheld sequence), and writes a top-down picture.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from rovermap.mapper import Mapper  # noqa: E402

seq = sys.argv[1]
step = int(sys.argv[2]) if len(sys.argv) > 2 else 4
model = sys.argv[3] if len(sys.argv) > 3 else "depth-anything/DA3-BASE"

rgb = [l.split() for l in open(os.path.join(seq, "rgb.txt")) if not l.startswith("#")]
gt = np.array([[float(x) for x in l.split()] for l in open(os.path.join(seq, "groundtruth.txt"))
               if not l.startswith("#")])

args = dict(a.split("=") for a in sys.argv[4:])
# mode: once | fwdrev (go back the same way through the mapped area) |
#       loop (jump back to the start: relocalisation)
mode = args.pop("mode", "once")
# sensor=<max metres>: feed the Kinect depth, cut beyond that range, as a
# stand-in for a short-range depth camera (RealSense D405: 7-50 cm ideal).
sensor_max = float(args.pop("sensor", "0"))
extra = {k: (float(v) if "." in v else int(v)) for k, v in args.items()}
# Optional calib.json in the sequence dir (simulator recordings): camera
# intrinsics at the recorded resolution, depth PNG scale, true camera height.
# Without it: TUM freiburg1 values, and an arbitrary height (handheld camera).
CALIB = {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3, "depth_factor": 5000.0,
         "cam_height": 1.0}
calib_path = os.path.join(seq, "calib.json")
if os.path.exists(calib_path):
    import json
    CALIB.update(json.load(open(calib_path)))
    print("calib:", CALIB)
extra.setdefault("cam_height", CALIB["cam_height"])
m = Mapper(model_name=model, **extra)
print(model, extra, "mode", mode)
ts, est, dts, qs, ovs, lost = [], [], [], [], [], []
stamps = np.array([float(l[0]) for l in rgb])
dep_all = [l.split() for l in open(os.path.join(seq, "depth.txt")) if not l.startswith("#")]
dstamps_all = np.array([float(l[0]) for l in dep_all])
kf_frame: dict[int, int] = {}


def frames():
    """Frame indices to feed, plus the index where the second pass starts."""
    if step <= 0 and mode == "once":
        return None, None
    fwd = list(range(0, len(rgb), max(step, 1)))
    if mode == "fwdrev":
        return fwd + fwd[::-1][1:], len(fwd)
    if mode == "loop":
        return fwd + fwd, len(fwd)
    if mode == "gap":  # sudden jump forward into a part not seen yet
        a, b = len(fwd) // 3, len(fwd) // 2
        return fwd[:a] + fwd[b:], a
    return fwd, len(fwd)


order, second = frames()
n_kf_first = None
seen_version = -1
k = 0
i = 0
while True:
    if order is not None:
        if k >= len(order):
            break
        i = order[k]
        if k == second:
            n_kf_first = len(m.keyframes)
    elif i >= len(rgb):
        break
    img = cv2.cvtColor(cv2.imread(os.path.join(seq, rgb[i][1])), cv2.COLOR_BGR2RGB)
    sd = None
    if sensor_max > 0:
        jd = int(np.argmin(np.abs(dstamps_all - stamps[i])))
        sd = cv2.imread(os.path.join(seq, dep_all[jd][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / CALIB["depth_factor"]
        sd[sd > sensor_max] = 0.0
    r = m.step(img, sd)
    if r is None:  # frame without detail: skipped by the mapper
        k += order is not None
        i += order is None
        continue
    # The map may have been re-expressed (metric calibration): move the poses
    # recorded so far with it, as the viewer does, so all are in one frame.
    for v_, s_, R_, t_ in m.transforms:
        if v_ > seen_version:
            est = [s_ * R_ @ p + t_ for p in est]
    seen_version = max([seen_version] + [x[0] for x in m.transforms])
    if r.new_keyframe is not None:
        kf_frame[r.new_keyframe.idx] = i
    ts.append(stamps[i])
    est.append(r.c2w[:3, 3].copy())
    dts.append(r.dt)
    qs.append(r.quality)
    ovs.append(r.overlap)
    lost.append(r.lost)
    if order is not None:
        k += 1
    else:  # live-like: next frame is whatever the camera shows when we are done
        i = max(i + 1, int(np.searchsorted(stamps, stamps[i] + r.dt)))
ts, est = np.array(ts), np.array(est)
lost = np.array(lost)
ovs = np.array(ovs)
print(f"overlap p10 {np.percentile(ovs[~lost], 10):.2f} median {np.median(ovs[~lost]):.2f}   "
      f"frames lost {lost.sum()} / {len(lost)}")
ref = np.stack([np.interp(ts, gt[:, 0], gt[:, k]) for k in (1, 2, 3)], 1)


def umeyama(a, b):
    """Similarity (s, R, t) minimising |s R a + t - b|."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


s, R, t = umeyama(est, ref)
aligned = s * est @ R.T + t
err = np.linalg.norm(aligned - ref, axis=1)
if second is not None and mode != "once":
    e1, e2 = err[:second][~lost[:second]], err[second:][~lost[second:]]
    print(f"[{mode}] pass 1: ATE {np.sqrt((e1 ** 2).mean()) * 100:.1f} cm, {n_kf_first} keyframes | "
          f"pass 2: ATE {np.sqrt((e2 ** 2).mean()) * 100:.1f} cm (tracked frames only), "
          f"+{len(m.keyframes) - n_kf_first} new keyframes, lost {lost[second:].sum()} / {len(lost) - second}")
length = np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()
print(f"frames {len(ts)}  keyframes {len(m.keyframes)}  mean step {np.mean(dts[2:]):.3f}s  "
      f"calibrated {m.scale_calibrated}")
print(f"quality mean {np.mean(qs):.2f} min {np.min(qs):.2f}  frames below 0.5: {(np.array(qs) < 0.5).sum()}")
print(f"GT path length {length:.2f} m   ATE rmse {np.sqrt((err ** 2).mean()) * 100:.1f} cm   "
      f"max {err.max() * 100:.1f} cm   scale est->gt {s:.3f}")
# scale drift: alignment scale of first vs second half
h = len(est) // 2
print("scale first half %.3f  second half %.3f" % (umeyama(est[:h], ref[:h])[0], umeyama(est[h:], ref[h:])[0]))

# --- absolute scale check -----------------------------------------------
# For each keyframe used to set the scale, measure the true camera height
# above the same kind of plane with the Kinect depth. If the floor-based
# calibration were perfect, the map scale (est -> metres) would equal the
# true height divided by the height the mapper believes in.
from rovermap.mapper import fit_ground  # noqa: E402

dep = [l.split() for l in open(os.path.join(seq, "depth.txt")) if not l.startswith("#")]
dstamps = np.array([float(l[0]) for l in dep])
K_tum = np.array([[CALIB["fx"], 0, CALIB["cx"]], [0, CALIB["fy"], CALIB["cy"]], [0, 0, 1.0]])
ratios = []
for kf, h_map in zip(m.calib_kfs, m.ground_heights):
    fi = kf_frame[kf.idx]
    j = int(np.argmin(np.abs(dstamps - stamps[fi])))
    d = cv2.imread(os.path.join(seq, dep[j][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / CALIB["depth_factor"]
    g = fit_ground(d, K_tum, d > 0, np.random.default_rng(0))
    if g is not None:
        ratios.append(g[1] / h_map)
if ratios:
    expect = float(np.median(ratios))
    print(f"calib frames {len(m.ground_heights)} (fits with GT {len(ratios)})  "
          f"expected scale {expect:.3f}  measured {s:.3f}  "
          f"=> metric scale error {100 * (s / expect - 1):+.1f}%")

# --- plane-free check -----------------------------------------------------
# Kinect depth / map depth on the same pixels of each keyframe. If the map
# is self-consistent, this ratio equals the trajectory scale `s` above.
pix, fxs = [], []
BANDS = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.5), (1.5, 99.0)]
band_r = {b: [] for b in BANDS}
band_n = {b: 0 for b in BANDS}
for kf in m.keyframes:
    fi = kf_frame.get(kf.idx)
    if fi is None:
        continue
    j = int(np.argmin(np.abs(dstamps - stamps[fi])))
    d = cv2.imread(os.path.join(seq, dep[j][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / CALIB["depth_factor"]
    hh, ww = kf.depth.shape
    dk = cv2.resize(d, (ww, hh), interpolation=cv2.INTER_NEAREST)
    ok = kf.valid & (dk > 0)
    if ok.sum() > 100:
        pix.append(float(np.median(dk[ok] / kf.depth[ok])))
        rr = dk[ok] / kf.depth[ok]
        for lo_, hi_ in BANDS:
            sel = (dk[ok] >= lo_) & (dk[ok] < hi_)
            band_n[(lo_, hi_)] += int(sel.sum())
            if sel.sum() > 50:
                band_r[(lo_, hi_)].append(float(np.median(rr[sel])))
    fxs.append(float(kf.K[0, 0]) * (2 * CALIB["cx"]) / ww)
tot = sum(band_n.values())
for b in BANDS:
    r_ = band_r[b]
    med = f"{np.median(r_):.3f}" if r_ else "  -  "
    print(f"kinect band {b[0]:.1f}-{b[1]:.1f} m: ratio kinect/map median {med} over {len(r_):3d} keyframes, "
          f"{100 * band_n[b] / max(tot, 1):4.1f}% of pixels")
pix = np.array(pix)
print(f"pixel depth ratio kinect/map: median {np.median(pix):.3f}  "
      f"p10 {np.percentile(pix, 10):.3f}  p90 {np.percentile(pix, 90):.3f}  (trajectory scale {s:.3f})")
print(f"focal fx at 640 px: median {np.median(fxs):.1f}  p10 {np.percentile(fxs, 10):.1f}  "
      f"p90 {np.percentile(fxs, 90):.1f}  (true {CALIB['fx']:.1f})")
first, last = pix[: len(pix) // 3], pix[-len(pix) // 3:]
print(f"pixel ratio first third {np.median(first):.3f}  last third {np.median(last):.3f}")

os.makedirs("out", exist_ok=True)
size = 700
allp = np.concatenate([aligned, ref])[:, :2]
lo, hi = allp.min(0) - 0.2, allp.max(0) + 0.2
sc = size / max(hi - lo)
img = np.full((size, size, 3), 255, np.uint8)
for pts, col in ((ref, (0, 160, 0)), (aligned, (0, 0, 255))):
    uv = ((pts[:, :2] - lo) * sc).astype(np.int32)
    cv2.polylines(img, [uv.reshape(-1, 1, 2)], False, col, 2)
cv2.putText(img, "vert: reference   rouge: estimation", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
cv2.imwrite("out/tum_traj.png", img)
m.save_ply("out/tum_map.ply")
