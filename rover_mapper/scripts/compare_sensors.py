"""Replay ONE recorded simulator sequence in each sensor mode and compare with
ground truth (sequence from sim/record_sim_sequence.py).

    python scripts/compare_sensors.py [seq dir] [--step 2] [--frames N]
                                      [--modes rgb,d405,both] [--model ...] [--res 336] [--ground]

Modes (same as the dashboard's "Capteurs" menu):
  rgb   "Caméra du robot (RGB)"  mapper.step(robot rgb)              scale from the floor
  d405  "D405 seule"             mapper.step(d405 rgb, d405 depth)   scale from the sensor
  both  "Les deux"               mapper.step(robot rgb, d405 depth reprojected into the
                                 robot camera with the exact extrinsics)
Every mode sees the same frames (fixed frame step, so results do not depend
on how loaded the GPU is) and the same simulated D405 noise.

Columns (tracked = not lost frames only):
  ATE local  position error (RMS) of the camera after aligning the estimated
             trajectory on the true one with a similarity (rotation, translation
             AND scale), per consecutive 2 m window of true path, median over
             the windows: local accuracy of the path's shape.
  Échelle locale   scale error of each window = 1/s - 1 for the window's
             alignment scale s (+10 % = map 10 % too big): median [min, max]
             over the windows, and the first window alone (calibration bias).
  ATE trajet entier   the same alignment over the whole route: dominated by
             drift (the mapper has no loop closure); its scale (in the .json,
             "scale_err") is distorted once the path shape has drifted, so it
             is not shown as the map's scale.
  Prof.      depth of the final map's keyframes vs true depth of the same
             camera: median |d - d_true| / d_true, as is (métrique) and after
             one scale factor per keyframe (forme).
  perdues    frames whose pose was rejected (robot "lost"); ignorées = frames
             the mapper refused (no detail).
  s/image    median mapper time per frame (noisy: the GPU is shared).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from rovermap.source import (MODE_BOTH, MODE_D405, MODE_RGB, TruthTracker,  # noqa: E402
                             mode_settings, sensor_inputs, umeyama)

KEYS = {"rgb": MODE_RGB, "d405": MODE_D405, "both": MODE_BOTH}
WIN = 2.0   # metres of true path per window for the local metrics

ap = argparse.ArgumentParser()
ap.add_argument("seq", nargs="?", default=os.path.join(ROOT, "out", "sensors", "apt_seq"))
ap.add_argument("--step", type=int, default=2, help="use one frame out of N (recorded at calib hz)")
ap.add_argument("--frames", type=int, default=0, help="only the first N recorded frames (0 = all)")
ap.add_argument("--modes", default="rgb,d405,both")
ap.add_argument("--model", default="depth-anything/DA3-BASE")
ap.add_argument("--res", type=int, default=336)
ap.add_argument("--ground", action="store_true", help="Mapper(ground_robot=True)")
ap.add_argument("--out", default=None, help="results prefix (default out/sensors/compare_<seq>)")
ap.add_argument("--from-json", default=None, help="only rebuild the table from this results .json")
args = ap.parse_args()

calib = json.load(open(os.path.join(args.seq, "calib.json")))
truth = np.load(os.path.join(args.seq, "truth.npz"))
n_rec = len(truth["t"])
idx = list(range(0, args.frames or n_rec, args.step))
K = np.array(calib["K"])
Kd = np.array(calib["d405"]["K"])
F, FD, FDT = calib["depth_factor"], calib["d405"]["depth_factor"], calib["d405"]["true_depth_factor"]
print(f"sequence {args.seq}: {n_rec} frames at {calib['hz']:g} Hz, {calib['path_m']:.1f} m; "
      f"using {len(idx)} (step {args.step})")


def load(i: int) -> dict:
    n = f"{i:05d}.png"
    rd = lambda d: cv2.imread(os.path.join(args.seq, d, n), cv2.IMREAD_UNCHANGED)  # noqa: E731
    return {"rgb": cv2.cvtColor(rd("rgb"), cv2.COLOR_BGR2RGB),
            "depth": rd("depth").astype(np.float32) / F, "K": K, "T": truth["T_rgb"][i],
            "d405_rgb": cv2.cvtColor(rd("d405_rgb"), cv2.COLOR_BGR2RGB),
            "d405_depth": rd("d405_depth").astype(np.float32) / FD,
            "d405_depth_true": rd("d405_true").astype(np.float32) / FDT,
            "d405_K": Kd, "d405_T": truth["T_d405"][i]}


def pct(v, signed=False):
    return "-" if v is None else (f"{100 * v:+.0f} %" if signed else f"{100 * v:.1f} %")


def cm(v):
    return "-" if v is None else f"{100 * v:.1f}"


def make_table(results: dict) -> str:
    lines = [
        f"Séquence `{os.path.basename(os.path.normpath(args.seq))}` : {len(idx)} images (1 sur {args.step}, "
        f"{calib['hz'] / args.step:g} Hz), {calib['path_m']:.1f} m, {args.model.split('/')[-1]} à {args.res}, "
        f"ground_robot={args.ground}. Fenêtres de {WIN:g} m de trajet réel.",
        "",
        "| Capteurs | ATE local (cm) | Échelle locale médiane [min, max] | Échelle 1re fenêtre "
        "| Prof. métrique | Prof. forme | Perdues | Img. clés | Échelle fixée par "
        "| ATE trajet entier (dérive) | s/image |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, m in results.items():
        ws = m.get("win_scale") or []
        loc = (f"{pct(float(np.median(ws)), True)} [{pct(min(ws), True)}, {pct(max(ws), True)}]"
               if ws else "-")
        lines.append(
            f"| {KEYS[key]} | {cm(m.get('ate_local'))} | {loc} | {pct(m.get('scale_first'), True)} "
            f"| {pct(m['depth_abs'])} | {pct(m['depth_shape'])} "
            f"| {m['lost']}/{m['frames']} ({pct(m['lost_rate'])}) | {m['keyframes']} "
            f"| {m['scale_source']} | {cm(m.get('ate'))} | {m['dt']:.2f} |")
    return "\n".join(lines)


prefix = args.out or os.path.join(ROOT, "out", "sensors", "compare_" + os.path.basename(os.path.normpath(args.seq)))
if args.from_json:   # only rebuild the table from a previous run's results
    table = make_table(json.load(open(args.from_json))["results"])
    print(table)
    open(prefix + ".md", "w").write(table + "\n")
    sys.exit(0)

from rovermap.mapper import Mapper  # noqa: E402

mapper = Mapper(model_name=args.model, process_res=args.res, cam_height=calib["cam_height"],
                mask_moving=False, ground_robot=args.ground)
default_range = tuple(mapper.sensor_range)
results, trajs = {}, {}
for key in args.modes.split(","):
    mode = KEYS[key]
    mapper.cam_height, mapper.sensor_range = mode_settings(
        mode, calib["cam_height"], calib["d405"]["cam_height"], default_range)
    mapper.reset()
    mapper.rng = np.random.default_rng(0)
    tr = TruthTracker()
    kf_frame: dict[int, int] = {}
    sensor_px = []
    t0 = time.time()
    for k, i in enumerate(idx):
        snap = load(i)
        img, sd, _, T_true, d_true = sensor_inputs(snap, mode)
        if sd is not None:
            sensor_px.append(float((sd > 0).mean()))
        r = mapper.step(img, sd)
        if r is None:
            tr.n_none += 1
            continue
        tr.add(mapper, r, T_true, d_true)
        if r.new_keyframe is not None:
            kf_frame[r.new_keyframe.idx] = i
        if k % 50 == 0:
            print(f"  [{key}] {k}/{len(idx)}  kf {len(mapper.keyframes)}  lost {sum(tr.lost)}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    m = tr.metrics()
    # Windows of ~WIN m of true path: local accuracy and how the scale moves
    # along the route (first window = calibration bias, last = after drift).
    E_all, G_all = np.array(tr.est), np.array(tr.true)
    okm = ~np.array(tr.lost, bool)
    E_ok, G_ok = E_all[okm], G_all[okm]
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(G_ok, axis=0), axis=1))] if len(G_ok) else np.zeros(0)
    win_scale, win_ate = [], []
    if len(seg):
        for a in np.arange(0.0, seg[-1] - WIN / 2, WIN):
            sel = (seg >= a) & (seg < a + WIN)
            if sel.sum() < 8:
                continue
            s_w, R_w, t_w = umeyama(E_ok[sel], G_ok[sel])
            win_scale.append(1.0 / s_w - 1.0)
            win_ate.append(float(np.sqrt((np.linalg.norm(s_w * E_ok[sel] @ R_w.T + t_w - G_ok[sel], axis=1) ** 2).mean())))
    m.update(scale_first=win_scale[0] if win_scale else None, scale_last=win_scale[-1] if win_scale else None,
             ate_local=float(np.median(win_ate)) if win_ate else None, n_windows=len(win_scale),
             win_scale=win_scale)
    # Depth of the final map (keyframes) vs the true depth of the same camera.
    abs_rel, shape_rel = [], []
    for kf in mapper.keyframes:
        i = kf_frame.get(kf.idx)
        if i is None:
            continue
        d_true = sensor_inputs(load(i), mode)[4]
        h, w = kf.depth.shape
        dt_ = cv2.resize(d_true, (w, h), interpolation=cv2.INTER_NEAREST)
        ok = kf.valid & (dt_ > 0.05) & (dt_ < mapper.max_depth)
        if ok.sum() < 100:
            continue
        d, g = kf.depth[ok], dt_[ok]
        abs_rel.append(float(np.median(np.abs(d - g) / g)))
        ks = float(np.median(g / d))
        shape_rel.append(float(np.median(np.abs(ks * d - g) / g)))
    if mapper.sensor_ratios:
        src = f"capteur ({len(mapper.sensor_ratios)} img. clés)" if mapper.sensor_scaled else \
            f"capteur NON appliqué ({len(mapper.sensor_ratios)}/{mapper.calib_frames}) + sol 1 image"
    elif mapper.scale_final:
        src = f"sol ({len(mapper.ground_heights)} img. clés)"
    elif mapper.scale_calibrated:
        src = f"sol provisoire ({len(mapper.ground_heights)})"
    else:
        src = "aucune"
    m.update(keyframes=len(mapper.keyframes), depth_abs=float(np.median(abs_rel)) if abs_rel else None,
             depth_shape=float(np.median(shape_rel)) if shape_rel else None, scale_source=src,
             sensor_scaled=bool(mapper.sensor_scaled), n_sensor_ratios=len(mapper.sensor_ratios),
             sensor_px=float(np.mean(sensor_px)) if sensor_px else None,
             cam_height=mapper.cam_height, sensor_range=list(mapper.sensor_range),
             wall_s=time.time() - t0)
    results[key] = m
    lost = np.array(tr.lost, bool)
    trajs[key] = (np.array(tr.est), np.array(tr.true), lost)
    print(f"  [{key}] done: {json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in m.items()})}",
          flush=True)


table = make_table(results)
print("\n" + table + "\n")
os.makedirs(os.path.dirname(prefix), exist_ok=True)
open(prefix + ".md", "w").write(table + "\n")
json.dump({"args": vars(args), "results": results}, open(prefix + ".json", "w"), indent=1)
np.savez(prefix + "_traj.npz", **{f"{k}_{n}": a for k, v in trajs.items() for n, a in zip(("est", "true", "lost"), v)})

# Top view: true path (green) and each mode's path after similarity alignment.
COL = {"rgb": (0, 0, 230), "d405": (230, 120, 0), "both": (200, 0, 200)}
size = 800
G = trajs[next(iter(trajs))][1]
lo, hi = G[:, :2].min(0) - 0.5, G[:, :2].max(0) + 0.5
sc = size / max(hi - lo)
img = np.full((size, size, 3), 255, np.uint8)


def uv(p):
    q = (p[:, :2] - lo) * sc
    return np.stack([q[:, 0], size - q[:, 1]], 1).astype(np.int32).reshape(-1, 1, 2)


cv2.polylines(img, [uv(G)], False, (0, 160, 0), 3)
for n_, (key, (E, Gk, lost)) in enumerate(trajs.items()):
    if (~lost).sum() < 3:
        continue
    s, R, t = umeyama(E[~lost], Gk[~lost])
    cv2.polylines(img, [uv(s * E @ R.T + t)], False, COL[key], 2)
    cv2.putText(img, f"{key}", (10, 50 + 22 * n_), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[key], 2)
cv2.putText(img, "vert: verite terrain (camera du mode rgb)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 0), 2)
cv2.imwrite(prefix + "_traj.png", img)
print("written", prefix + ".md", prefix + ".json", prefix + "_traj.png")
