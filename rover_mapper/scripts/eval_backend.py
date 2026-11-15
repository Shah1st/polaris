"""Score the mapper on a TUM-format sequence with a chosen depth network.

    python scripts/eval_backend.py --backend da3|mapanything <seq> <frame step> [model|-] [kwarg=value ...]
        [--tag NAME] [--depth-eval N] [--mv-gap K]

Same scoring as scripts/eval_tum.py (ATE after similarity alignment, scale
est->gt, ground-truth depth / map depth per distance band, focal estimate,
lost frames), plus time per step and peak memory. Run the da3 backend with
.venv and the mapanything backend with .venv-ma (each venv has only its own
network installed).

--depth-eval N: no tracking. Runs the network on N frames spread over the
sequence, alone (1 view) and with 3 earlier frames (4 views, like a tracking
step; frames K apart, --mv-gap), and compares its raw depth with the
ground-truth depth WITHOUT any alignment (ratio gt/pred), then after a
per-frame median scale (AbsRel, delta<1.25).

Outputs go to out/mapanything/<tag>.json (+ _steps.csv, _traj.png, _map.ply).
Backend-specific kwargs: mapanything takes metric=0|1|2, amp=fp16|fp32
(see rovermap/mapanything_backend.py).
"""
import json
import os
import resource
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

# --- arguments ---------------------------------------------------------------
argv = sys.argv[1:]
opts = {"--backend": "da3", "--tag": None, "--depth-eval": "0", "--mv-gap": "8"}
pos = []
while argv:
    a = argv.pop(0)
    if a in opts:
        opts[a] = argv.pop(0)
    else:
        pos.append(a)
backend = opts["--backend"]
assert backend in ("da3", "mapanything"), backend
plain = [a for a in pos if "=" not in a]
seq = plain[0]
step = int(plain[1]) if len(plain) > 1 else 4
model = plain[2] if len(plain) > 2 and plain[2] != "-" else None
n_depth = int(opts["--depth-eval"])
mv_gap = int(opts["--mv-gap"])


def _val(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


args = dict(a.split("=", 1) for a in pos if "=" in a)
mode = args.pop("mode", "once")
sensor_max = float(args.pop("sensor", "0"))
extra = {k: _val(v) for k, v in args.items()}

CALIB = {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3, "depth_factor": 5000.0,
         "cam_height": 1.0}
calib_path = os.path.join(seq, "calib.json")
if os.path.exists(calib_path):
    CALIB.update(json.load(open(calib_path)))
    print("calib:", CALIB)
extra.setdefault("cam_height", CALIB["cam_height"])
# known_K=1 (mapanything only): give the network the true intrinsics instead of
# letting it estimate them (not like-for-like with DA3, which estimates them).
if extra.pop("known_K", 0):
    extra["intrinsics"] = (CALIB["fx"], CALIB["fy"], CALIB["cx"], CALIB["cy"])
if n_depth:
    extra.setdefault("mask_moving", 0)  # not needed to look at raw depth

tag = opts["--tag"] or "{}_{}_s{}{}".format(
    backend, os.path.basename(os.path.normpath(seq)), step,
    "".join(f"_{k}{v}" for k, v in sorted(extra.items()) if k != "cam_height")
    + ("_depth" if n_depth else ""))
OUT = "out/mapanything"
os.makedirs(OUT, exist_ok=True)

import torch  # noqa: E402

if backend == "da3":
    from rovermap.mapper import Mapper as Backend  # noqa: E402
    model = model or "depth-anything/DA3-BASE"
else:
    from rovermap.mapanything_backend import DEFAULT_WEIGHTS, MapAnythingMapper as Backend  # noqa: E402
    model = model or DEFAULT_WEIGHTS

mps = torch.backends.mps.is_available()


def mem():
    return (torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory()) if mps else (0, 0)


t_load = time.time()
m = Backend(model_name=model, **extra)
t_load = time.time() - t_load
mem_model = mem()
print(backend, model, extra, "mode", mode, f"loaded in {t_load:.1f}s, MPS driver {mem_model[0] / 1e9:.2f} GB")

# Record inference time / memory around every network call.
peak = {"driver": mem_model[0], "current": mem_model[1]}
infer_log = []  # (n views, seconds)
_orig_infer = m._infer


def _timed_infer(images):
    t0 = time.time()
    out = _orig_infer(images)
    infer_log.append((len(images), time.time() - t0))
    d, c = mem()
    peak["driver"], peak["current"] = max(peak["driver"], d), max(peak["current"], c)
    return out


m._infer = _timed_infer

rgb = [l.split() for l in open(os.path.join(seq, "rgb.txt")) if not l.startswith("#")]
gt = np.array([[float(x) for x in l.split()] for l in open(os.path.join(seq, "groundtruth.txt"))
               if not l.startswith("#")])
stamps = np.array([float(l[0]) for l in rgb])
dep_all = [l.split() for l in open(os.path.join(seq, "depth.txt")) if not l.startswith("#")]
dstamps_all = np.array([float(l[0]) for l in dep_all])


def load_rgb(i):
    return cv2.cvtColor(cv2.imread(os.path.join(seq, rgb[i][1])), cv2.COLOR_BGR2RGB)


def load_gt_depth(i):
    j = int(np.argmin(np.abs(dstamps_all - stamps[i])))
    return cv2.imread(os.path.join(seq, dep_all[j][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / CALIB["depth_factor"]


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


summary = {"backend": backend, "model": model, "seq": seq, "step": step, "kwargs": extra,
           "mode": mode, "load_s": t_load, "mem_model_gb": mem_model[0] / 1e9}


def finish():
    summary["peak_mps_driver_gb"] = peak["driver"] / 1e9
    summary["peak_mps_current_gb"] = peak["current"] / 1e9
    summary["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    by_n = {}
    for n, t in infer_log[3:]:
        by_n.setdefault(n, []).append(t)
    summary["infer_s_by_views"] = {str(n): float(np.median(v)) for n, v in sorted(by_n.items())}
    print(f"peak MPS driver {summary['peak_mps_driver_gb']:.2f} GB (current {summary['peak_mps_current_gb']:.2f}), "
          f"peak RSS {summary['peak_rss_gb']:.2f} GB, model alone {summary['mem_model_gb']:.2f} GB")
    print("median network call (s) by number of views:", summary["infer_s_by_views"])
    with open(os.path.join(OUT, tag + ".json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("wrote", os.path.join(OUT, tag + ".json"))


# =============================================================================
# Raw depth check: network depth vs ground truth, no alignment
# =============================================================================
if n_depth:
    first = 3 * mv_gap
    idx = np.unique(np.linspace(first, len(rgb) - 1, n_depth).astype(int))
    rows = {1: [], 4: []}
    for i in idx:
        g = load_gt_depth(i)
        for nv in (1, 4):
            frames_ = [i - k * mv_gap for k in range(nv - 1, -1, -1)]
            d, c, L, K, ims = m._infer([load_rgb(j) for j in frames_])
            pred = d[-1]
            hh, ww = pred.shape
            gr = cv2.resize(g, (ww, hh), interpolation=cv2.INTER_NEAREST)
            ok = (gr > 0) & (gr < 20) & np.isfinite(pred) & (pred > 1e-6)
            if ok.sum() < 500:
                continue
            r = gr[ok] / pred[ok]
            med = float(np.median(r))
            pa = pred[ok] * med
            rows[nv].append(dict(
                frame=int(i), ratio=med, ratio_iqr=float(np.percentile(r, 75) - np.percentile(r, 25)),
                absrel_raw=float(np.mean(np.abs(pred[ok] - gr[ok]) / gr[ok])),
                absrel_aligned=float(np.mean(np.abs(pa - gr[ok]) / gr[ok])),
                delta1_aligned=float(np.mean(np.maximum(pa / gr[ok], gr[ok] / pa) < 1.25)),
                fx640=float(K[-1][0, 0] * g.shape[1] / ww),
                coverage=float(ok.mean())))
        print(f"frame {i}: " + "  ".join(
            f"{nv}v ratio {rows[nv][-1]['ratio']:.3f} absrel-aligned {rows[nv][-1]['absrel_aligned']:.3f}"
            for nv in (1, 4) if rows[nv] and rows[nv][-1]["frame"] == i), flush=True)
    summary["depth_eval"] = {}
    print(f"\n{backend} raw depth vs ground truth (ratio = gt / pred, 1.000 = exact metres), "
          f"{len(idx)} frames, {ims.shape[2]}x{ims.shape[1]} px")
    for nv in (1, 4):
        R_ = rows[nv]
        if not R_:
            continue
        ra = np.array([x["ratio"] for x in R_])
        s_ = {
            "frames": len(R_),
            "ratio_median": float(np.median(ra)), "ratio_p10": pct(ra, 10), "ratio_p90": pct(ra, 90),
            "ratio_mean": float(ra.mean()), "ratio_cv": float(ra.std() / ra.mean()),
            "ratio_within_frame_iqr_median": float(np.median([x["ratio_iqr"] for x in R_])),
            "absrel_raw_median": float(np.median([x["absrel_raw"] for x in R_])),
            "absrel_aligned_median": float(np.median([x["absrel_aligned"] for x in R_])),
            "delta1_aligned_median": float(np.median([x["delta1_aligned"] for x in R_])),
            "fx640_median": float(np.median([x["fx640"] for x in R_])), "fx_true": CALIB["fx"],
            "per_frame": R_,
        }
        summary["depth_eval"][str(nv)] = s_
        print(f"  {nv} view(s): ratio median {s_['ratio_median']:.3f}  p10 {s_['ratio_p10']:.3f}  "
              f"p90 {s_['ratio_p90']:.3f}  (cv {s_['ratio_cv']:.3f}, within-frame IQR {s_['ratio_within_frame_iqr_median']:.3f})  "
              f"AbsRel raw {s_['absrel_raw_median']:.3f}  aligned {s_['absrel_aligned_median']:.3f}  "
              f"d1 {s_['delta1_aligned_median']:.3f}  fx@640 {s_['fx640_median']:.1f} (true {CALIB['fx']:.1f})")
    finish()
    sys.exit(0)

# =============================================================================
# Tracking run (same loop as eval_tum.py)
# =============================================================================
ts, est, dts, qs, ovs, lost, align_s, fidx, is_kf = [], [], [], [], [], [], [], [], []
kf_frame: dict[int, int] = {}


def frames():
    if step <= 0 and mode == "once":
        return None, None
    fwd = list(range(0, len(rgb), max(step, 1)))
    if mode == "fwdrev":
        return fwd + fwd[::-1][1:], len(fwd)
    if mode == "loop":
        return fwd + fwd, len(fwd)
    if mode == "gap":
        a, b = len(fwd) // 3, len(fwd) // 2
        return fwd[:a] + fwd[b:], a
    return fwd, len(fwd)


order, second = frames()
n_kf_first = None
seen_version = -1
k = 0
i = 0
t_run = time.time()
while True:
    if order is not None:
        if k >= len(order):
            break
        i = order[k]
        if k == second:
            n_kf_first = len(m.keyframes)
    elif i >= len(rgb):
        break
    img = load_rgb(i)
    sd = None
    if sensor_max > 0:
        sd = load_gt_depth(i)
        sd[sd > sensor_max] = 0.0
    n_before = len(m.keyframes)
    r = m.step(img, sd)
    if r is None:
        k += order is not None
        i += order is None
        continue
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
    align_s.append(m.last_align[2] if n_before else float("nan"))
    fidx.append(i)
    is_kf.append(r.new_keyframe is not None)
    if len(ts) % 50 == 0:
        print(f"  {len(ts)} frames, {len(m.keyframes)} keyframes, lost {int(np.sum(lost))}, "
              f"{time.time() - t_run:.0f}s", flush=True)
    if order is not None:
        k += 1
    else:
        i = max(i + 1, int(np.searchsorted(stamps, stamps[i] + r.dt)))
ts, est = np.array(ts), np.array(est)
lost = np.array(lost)
ovs = np.array(ovs)
align_s = np.array(align_s)
print(f"overlap p10 {pct(ovs[~lost], 10):.2f} median {pct(ovs[~lost], 50):.2f}   "
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
warm = 5  # first steps include MPS kernel compilation for each new number of views
print(f"frames {len(ts)}  keyframes {len(m.keyframes)}  mean step {np.mean(dts[warm:]):.3f}s "
      f"(median {np.median(dts[warm:]):.3f}s)  calibrated {m.scale_calibrated}")
print(f"quality mean {np.mean(qs):.2f} min {np.min(qs):.2f}  frames below 0.5: {(np.array(qs) < 0.5).sum()}")
print(f"GT path length {length:.2f} m   ATE rmse {np.sqrt((err ** 2).mean()) * 100:.1f} cm   "
      f"max {err.max() * 100:.1f} cm   scale est->gt {s:.3f}")
h = len(est) // 2
s1, s2 = umeyama(est[:h], ref[:h])[0], umeyama(est[h:], ref[h:])[0]
print("scale first half %.3f  second half %.3f" % (s1, s2))
# Rigid (no scale) ATE: what a metric map is worth without any rescaling.
ma_, mb_ = est.mean(0), ref.mean(0)
U_, S_, Vt_ = np.linalg.svd((ref - mb_).T @ (est - ma_))
D_ = np.diag([1, 1, np.sign(np.linalg.det(U_ @ Vt_))])
R_r = U_ @ D_ @ Vt_
err_rigid = np.linalg.norm((est - ma_) @ R_r.T + mb_ - ref, axis=1)
print(f"ATE with rigid alignment (scale 1, metres as estimated): {np.sqrt((err_rigid ** 2).mean()) * 100:.1f} cm")
okA = np.isfinite(align_s) & ~lost
if okA.any():
    a_ = align_s[okA]
    q = max(1, len(a_) // 4)
    print(f"align scale map/network per step: median {np.median(a_):.3f}  p10 {pct(a_, 10):.3f}  p90 {pct(a_, 90):.3f}  "
          f"first quarter {np.median(a_[:q]):.3f}  last quarter {np.median(a_[-q:]):.3f}")

summary.update({
    "frames": int(len(ts)), "keyframes": len(m.keyframes), "lost": int(lost.sum()),
    "step_s_mean": float(np.mean(dts[warm:])), "step_s_median": float(np.median(dts[warm:])),
    "run_s": time.time() - t_run, "quality_mean": float(np.mean(qs)),
    "path_m": float(length), "ate_cm": float(np.sqrt((err ** 2).mean()) * 100), "ate_max_cm": float(err.max() * 100),
    "ate_rigid_cm": float(np.sqrt((err_rigid ** 2).mean()) * 100),
    "scale_est_gt": float(s), "scale_first_half": float(s1), "scale_second_half": float(s2),
    "calibrated": bool(m.scale_calibrated),
})
if okA.any():
    summary.update({"align_s_median": float(np.median(a_)), "align_s_p10": pct(a_, 10), "align_s_p90": pct(a_, 90),
                    "align_s_first_q": float(np.median(a_[:q])), "align_s_last_q": float(np.median(a_[-q:]))})

# --- absolute scale check (floor height) --------------------------------------
from rovermap.mapper import fit_ground  # noqa: E402

K_gt = np.array([[CALIB["fx"], 0, CALIB["cx"]], [0, CALIB["fy"], CALIB["cy"]], [0, 0, 1.0]])
ratios = []
for kf, h_map in zip(m.calib_kfs, m.ground_heights):
    fi = kf_frame.get(kf.idx)
    if fi is None:
        continue
    d = load_gt_depth(fi)
    g = fit_ground(d, K_gt, d > 0, np.random.default_rng(0))
    if g is not None:
        ratios.append(g[1] / h_map)
if ratios:
    expect = float(np.median(ratios))
    print(f"calib frames {len(m.ground_heights)} (fits with GT {len(ratios)})  "
          f"expected scale {expect:.3f}  measured {s:.3f}  "
          f"=> metric scale error {100 * (s / expect - 1):+.1f}%")
    print(f"camera height above floor in the map: median {np.median(m.ground_heights):.3f}  "
          f"(cam_height setting {extra['cam_height']:.3f})")
    summary.update({"floor_expected_scale": expect, "map_cam_height_median": float(np.median(m.ground_heights))})

# --- ground-truth depth / map depth on keyframes --------------------------------
pix, fxs = [], []
BANDS = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.5), (1.5, 99.0)]
band_r = {b: [] for b in BANDS}
band_n = {b: 0 for b in BANDS}
for kf in m.keyframes:
    fi = kf_frame.get(kf.idx)
    if fi is None:
        continue
    d = load_gt_depth(fi)
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
    print(f"gt-depth band {b[0]:.1f}-{b[1]:.1f} m: ratio gt/map median {med} over {len(r_):3d} keyframes, "
          f"{100 * band_n[b] / max(tot, 1):4.1f}% of pixels")
pix = np.array(pix)
print(f"pixel depth ratio gt/map: median {pct(pix, 50):.3f}  "
      f"p10 {pct(pix, 10):.3f}  p90 {pct(pix, 90):.3f}  (trajectory scale {s:.3f})")
print(f"focal fx at 640 px: median {pct(fxs, 50):.1f}  p10 {pct(fxs, 10):.1f}  "
      f"p90 {pct(fxs, 90):.1f}  (true {CALIB['fx']:.1f})")
first, last = pix[: len(pix) // 3], pix[-len(pix) // 3:]
print(f"pixel ratio first third {pct(first, 50):.3f}  last third {pct(last, 50):.3f}")
summary.update({"depth_ratio_median": pct(pix, 50), "depth_ratio_p10": pct(pix, 10),
                "depth_ratio_p90": pct(pix, 90), "depth_ratio_first_third": pct(first, 50),
                "depth_ratio_last_third": pct(last, 50), "fx640_median": pct(fxs, 50),
                "bands": {f"{b[0]}-{b[1]}": (float(np.median(band_r[b])) if band_r[b] else None) for b in BANDS}})

# --- files ----------------------------------------------------------------------
with open(os.path.join(OUT, tag + "_steps.csv"), "w") as f:
    f.write("frame,t,dt,quality,overlap,lost,keyframe,align_s,est_x,est_y,est_z,gt_x,gt_y,gt_z,err\n")
    for j in range(len(ts)):
        f.write(f"{fidx[j]},{ts[j]:.4f},{dts[j]:.4f},{qs[j]:.3f},{ovs[j]:.3f},{int(lost[j])},{int(is_kf[j])},"
                f"{align_s[j]:.4f},{est[j][0]:.4f},{est[j][1]:.4f},{est[j][2]:.4f},"
                f"{ref[j][0]:.4f},{ref[j][1]:.4f},{ref[j][2]:.4f},{err[j]:.4f}\n")
size = 700
allp = np.concatenate([aligned, ref])[:, :2]
lo, hi = allp.min(0) - 0.2, allp.max(0) + 0.2
sc = size / max(hi - lo)
img = np.full((size, size, 3), 255, np.uint8)
for pts, col in ((ref, (0, 160, 0)), (aligned, (0, 0, 255))):
    uv = ((pts[:, :2] - lo) * sc).astype(np.int32)
    cv2.polylines(img, [uv.reshape(-1, 1, 2)], False, col, 2)
cv2.putText(img, f"{tag}  green: ground truth  red: estimate", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
cv2.imwrite(os.path.join(OUT, tag + "_traj.png"), img)
m.save_ply(os.path.join(OUT, tag + "_map.ply"))
finish()
