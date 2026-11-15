"""Replay a recorded forest sequence (scripts/vm/record_snapshots.py) through
the Mapper offline, in "both" mode (robot RGB + D405 depth reprojected into the
robot camera), and score it against the simulator's ground truth.

    python scripts/vm/eval_forest.py baseline
    python scripts/vm/eval_forest.py outdoor --set max_depth=15 voxel=0.05 kf_min_dist=0.3 \
        mask_sky=1 --true-k
    python scripts/vm/eval_forest.py --sky-check      # sky mask vs truth depth > 100 m

Reports: lost rate, keyframes, points, median step time, ATE with / without
scale alignment and scale error (TruthTracker), the mapper's depth / true depth
by distance band on tracked frames, and the map against a truth cloud built
from the exact depth and true poses: accuracy (estimated point -> nearest truth
point) and completeness (truth point -> nearest estimated point) by distance
band, after a similarity alignment of the estimated trajectory onto the true one.
One JSON line per run is appended to out/vm/eval_forest.jsonl.
"""
from __future__ import annotations

import argparse
import cProfile
import glob
import json
import os
import pstats
import sys
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rovermap.source import MODE_BOTH, TruthTracker, mode_settings, sensor_inputs, umeyama  # noqa: E402

BANDS = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 50)]
SKY = 100.0          # truth depth beyond this is sky (the renderer gives ~300 m)
TAU = 0.20           # m, a point "matches" when its neighbour is closer than this


def band_name(b) -> str:
    return f"{b[0]}-{b[1]}"


def parse_set(items):
    """key=value pairs -> Mapper kwargs (numbers / bools parsed)."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_seq(path, first=0, last=None, stride=1):
    files = sorted(glob.glob(os.path.join(path, "*.npz")))[first:last:stride]
    for f in files:
        z = np.load(f)
        yield {k: z[k] for k in z.files}


def truth_cloud(path, first, last, stride, voxel=0.05, pix=4, max_d=50.0):
    """Exact depth + true poses of every frame -> world points, one per voxel,
    with the shortest distance at which any frame saw it."""
    keys, dists, pts = [], [], []
    for snap in load_seq(path, first, last, stride):
        d, K, T = snap["depth"], snap["K"], snap["T"]
        h, w = d.shape
        vs, us = np.mgrid[0:h:pix, 0:w:pix]
        z = d[::pix, ::pix]
        ok = (z > 0.02) & (z < max_d)
        z, us, vs = z[ok], us[ok], vs[ok]
        P = np.stack([(us - K[0, 2]) / K[0, 0] * z, (vs - K[1, 2]) / K[1, 1] * z, z], 1)
        dist = np.linalg.norm(P, axis=1)
        W = P @ T[:3, :3].T + T[:3, 3]
        c = np.floor(W / voxel).astype(np.int64) + (1 << 20)
        keys.append((c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2])
        dists.append(dist)
        pts.append(W)
    keys, dists, pts = np.concatenate(keys), np.concatenate(dists), np.concatenate(pts)
    order = np.lexsort((dists, keys))          # per voxel: nearest observation first
    keys, dists, pts = keys[order], dists[order], pts[order]
    first_ = np.r_[True, keys[1:] != keys[:-1]]
    return pts[first_].astype(np.float32), dists[first_].astype(np.float32)


def write_ply(path, P, C) -> None:
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = P, C
        f.write(rec.tobytes())


def top_view(path, G, P, C, E, Gt, title) -> None:
    """Truth cloud (left) and the aligned estimated map (right), seen from above,
    points below 1.5 m above the camera path only (no canopy), with both paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree as KD
    zc = Gt[KD(Gt[:, :2]).query(G[:, :2])[1], 2]
    g = G[G[:, 2] < zc + 1.5]
    zc = Gt[KD(Gt[:, :2]).query(P[:, :2])[1], 2]
    keep = P[:, 2] < zc + 1.5
    rng = np.random.default_rng(0)
    gi = rng.choice(len(g), min(len(g), 400_000), replace=False)
    pi = np.flatnonzero(keep)
    pi = rng.choice(pi, min(len(pi), 400_000), replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    ax[0].scatter(g[gi, 0], g[gi, 1], s=0.2, c=g[gi, 2], cmap="terrain")
    ax[0].set_title("truth (exact depth + true poses)")
    ax[1].scatter(P[pi, 0], P[pi, 1], s=0.2, c=C[pi] / 255.0)
    ax[1].set_title(f"{title}: estimated map, aligned")
    for a in ax:
        a.plot(Gt[:, 0], Gt[:, 1], "r-", lw=1, label="true path")
        a.plot(E[:, 0], E[:, 1], "b-", lw=1, label="estimated path")
        a.set_aspect("equal")
    ax[0].legend(loc="upper right")
    lo, hi = Gt[:, :2].min(0) - 12, Gt[:, :2].max(0) + 12
    ax[0].set_xlim(lo[0], hi[0])
    ax[0].set_ylim(lo[1], hi[1])
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def map_quality(mapper, s, R, t, G, Gd):
    """Estimated map (moved by s, R, t into the true world) vs truth cloud G
    whose points have observation distances Gd. Bands use, for estimated
    points, the (aligned) distance to the keyframe that made them."""
    P, D = [], []
    for k in mapper.keyframes:
        if len(k.points) == 0:
            continue
        P.append(k.points.astype(np.float64))
        D.append(s * np.linalg.norm(k.points - k.c2w[:3, 3], axis=1))
    if not P:
        return {}
    P = np.concatenate(P) @ (s * R).T + t
    D = np.concatenate(D)
    acc = cKDTree(G).query(P, workers=4)[0]
    comp = cKDTree(P).query(G, workers=4, distance_upper_bound=5.0)[0]
    out = {"n_est": int(len(P)), "n_true": int(len(G)),
           "acc_med": float(np.median(acc)), "prec": float(np.mean(acc < TAU)),
           "comp": float(np.mean(comp < TAU))}
    for b in BANDS:
        e = (D >= b[0]) & (D < b[1])
        g = (Gd >= b[0]) & (Gd < b[1])
        out[band_name(b)] = {
            "n_est": int(e.sum()),
            "acc_med": float(np.median(acc[e])) if e.any() else None,
            "prec": float(np.mean(acc[e] < TAU)) if e.any() else None,
            "comp": float(np.mean(comp[g] < TAU)) if g.any() else None,
        }
    return out


def depth_ratio(r, depth_true, acc):
    """Per band, median of mapper depth / true depth on this frame's trusted pixels."""
    if r.lost or r.valid is None:
        return
    h, w = r.depth.shape
    dt = cv2.resize(depth_true.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    for b in BANDS:
        ok = r.valid & (dt >= max(b[0], 0.05)) & (dt < b[1])
        if ok.sum() > 50:
            acc.setdefault(band_name(b), []).append(float(np.median(r.depth[ok] / dt[ok])))
    # Valid pixels that are really sky: they become blobs in the trees.
    acc.setdefault("sky_in_valid", []).append(float(np.mean(dt[r.valid] > SKY)) if r.valid.any() else 0.0)


def sky_check(args):
    from rovermap.mapper import sky_mask
    tp = fp = fn = 0
    for snap in load_seq(args.seq, args.first, args.last, max(args.stride, 5)):
        m = sky_mask(snap["rgb"])
        sky = snap["depth"] > SKY
        tp += int((m & sky).sum())
        fp += int((m & ~sky).sum())
        fn += int((~m & sky).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print(f"sky mask: precision {prec:.3f}  recall {rec:.3f}  (sky share {(tp + fn) / max(tp + fp + fn, 1):.3f} of mask|sky)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", default="run")
    ap.add_argument("--seq", default="out/vm/forest_seq1")
    ap.add_argument("--set", nargs="*", help="Mapper kwargs, e.g. max_depth=15 voxel=0.05")
    ap.add_argument("--true-k", action="store_true", help="give the mapper the simulator's intrinsics")
    ap.add_argument("--res", type=int, default=504)
    ap.add_argument("--cam-height", type=float, default=0.212)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th recorded frame")
    ap.add_argument("--profile", help="write cProfile stats here and print the top entries")
    ap.add_argument("--ply", help="save the aligned estimated map here")
    ap.add_argument("--sweep", help="after the run, rebuild the map with other max_depth:voxel pairs "
                                    "(points only, tracking unchanged), e.g. 10:0.05,15:0.05")
    ap.add_argument("--png", help="save a top view: truth cloud vs aligned estimated map + trajectories")
    ap.add_argument("--sky-check", action="store_true")
    ap.add_argument("--results", default="out/vm/eval_forest.jsonl")
    args = ap.parse_args()
    if args.sky_check:
        sky_check(args)
        return

    from rovermap.mapper import Mapper
    kw = parse_set(args.set)
    # No people in the forest; the person masker would only cost time.
    kw.setdefault("mask_moving", False)
    mapper = Mapper(process_res=args.res, cam_height=args.cam_height, **kw)
    default_range = tuple(mapper.sensor_range)
    tracker = TruthTracker()
    ratios: dict[str, list[float]] = {}
    series = []
    diag = []   # per tracked frame: quality, overlap, pos_err, rot_err, scale, candidate xyz, keyframes
    t_wall = time.time()
    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()
    n = 0
    for snap in load_seq(args.seq, args.first, args.last, args.stride):
        if n == 0:
            mapper.cam_height, mapper.sensor_range = mode_settings(
                MODE_BOTH, args.cam_height, float(snap["d405_mount"][2]), default_range)
        frame, sd, K, T, d = sensor_inputs(snap, MODE_BOTH)
        r = mapper.step(frame, sd, K if args.true_k else None)
        n += 1
        if r is None:
            tracker.n_none += 1
            continue
        tracker.add(mapper, r, T, d)
        diag.append([r.quality, r.overlap, *mapper.last_align, *getattr(mapper, "last_candidate", np.eye(4))[:3, 3],
                     len(mapper.keyframes)])
        nr = len(ratios.get("0-2", []))
        depth_ratio(r, d, ratios)
        # Near-band depth ratio over time: shows scale drift along the route.
        series.append((n, ratios["0-2"][-1] if len(ratios.get("0-2", [])) > nr else np.nan, r.lost))
        if n % 100 == 0:
            m = tracker.metrics()
            print(f"  {n} frames  kf {len(mapper.keyframes)}  lost {m['lost_rate']:.2f}  "
                  f"dt {m['dt']:.3f}  ate {m.get('ate', float('nan')):.3f}", flush=True)
    if prof:
        prof.disable()
        prof.dump_stats(args.profile)
        pstats.Stats(args.profile).sort_stats("cumulative").print_stats(30)
        pstats.Stats(args.profile).sort_stats("tottime").print_stats(25)
    wall = time.time() - t_wall

    m = tracker.metrics()
    res = {"name": args.name, "set": kw, "true_k": args.true_k, "res": args.res, "stride": args.stride,
           "frames": n, "none": tracker.n_none, "lost_rate": m["lost_rate"],
           "keyframes": len(mapper.keyframes), "points": int(sum(len(k.points) for k in mapper.keyframes)),
           "dt_med": m["dt"], "dt_last100": float(np.median(tracker.dts[-100:])) if tracker.dts else None,
           "wall": wall, "ate": m.get("ate"), "ate_noscale": m.get("ate_noscale"),
           "scale_err": m.get("scale_err"), "path": m.get("path"),
           "sensor_scaled": bool(mapper.sensor_scaled), "n_sensor": len(mapper.sensor_ratios),
           "floor_calibrated": bool(mapper.scale_calibrated),
           "sensor_rescales": [float(x) for x in np.percentile(getattr(mapper, "sensor_rescales", []) or [1.0],
                                                                 [10, 50, 90])],
           "depth_ratio": {k: float(np.median(v)) for k, v in ratios.items()}}
    S = np.array(series, float)
    res["series"] = [[int(c * 50), float(np.nanmedian(S[(S[:, 0] >= c * 50) & (S[:, 0] < c * 50 + 50), 1])),
                      float(np.mean(S[(S[:, 0] >= c * 50) & (S[:, 0] < c * 50 + 50), 2]))]
                     for c in range(int(S[:, 0].max()) // 50 + 1)] if len(S) else []
    if m.get("ready"):
        lost = np.array(tracker.lost, bool)
        E, Gt = np.array(tracker.est)[~lost], np.array(tracker.true)[~lost]
        G, Gd = truth_cloud(args.seq, args.first, args.last, max(args.stride, 2))
        s, R, t = umeyama(E, Gt)
        res["map_sim3"] = map_quality(mapper, s, R, t, G, Gd)
        _, R1, t1 = umeyama(E, Gt, with_scale=False)
        res["map_se3"] = map_quality(mapper, 1.0, R1, t1, G, Gd)
        if args.png:
            P, C = mapper.all_points()
            top_view(args.png, G, P.astype(np.float64) @ (s * R).T + t, C, E @ (s * R).T + t, Gt, args.name)
        if args.ply:
            P, C = mapper.all_points()
            write_ply(args.ply, P.astype(np.float64) @ (s * R).T + t, C)
        # Keyframes are final: rebuilding their points with another depth
        # cut / voxel gives exactly the map a run with those values would have.
        for pair in (args.sweep.split(",") if args.sweep else []):
            f = pair.split(":")      # max_depth:voxel[:sky], sky = 0/1 (map-only sky mask)
            md, vx = float(f[0]), float(f[1])
            mapper.max_depth, mapper.voxel = md, vx
            if len(f) > 2:
                mapper.mask_sky = bool(int(f[2]))
            mapper._rebuild_points()
            res[f"map_sim3@{pair}"] = map_quality(mapper, s, R, t, G, Gd)
            res[f"map_se3@{pair}"] = map_quality(mapper, 1.0, R1, t1, G, Gd)
            res[f"points@{pair}"] = int(sum(len(k.points) for k in mapper.keyframes))
            P_, _ = mapper.all_points()
            # Sky points left in the map: aligned points far (> 3 m) from any true surface, above the camera.
            acc_ = cKDTree(G).query(P_.astype(np.float64) @ (s * R).T + t, workers=4)[0]
            res[f"far_off@{pair}"] = float(np.mean(acc_ > 3.0))
    # Trajectories (estimated, re-expressed like the viewer does; true; lost) for later study.
    os.makedirs("out/vm/eval", exist_ok=True)
    np.savez(f"out/vm/eval/traj_{args.name}.npz", est=np.array(tracker.est), true=np.array(tracker.true),
             lost=np.array(tracker.lost), dts=np.array(tracker.dts),
             diag=np.array(diag), transforms_n=len(mapper.transforms))
    with open(args.results, "a") as f:
        f.write(json.dumps(res) + "\n")
    print_result(res)


def fmt(v, p=3):
    return "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def print_result(r) -> None:
    print(f"\n== {r['name']}  {r['set']}  true_k={r['true_k']}")
    print(f"frames {r['frames']}  none {r['none']}  lost {fmt(r['lost_rate'])}  kf {r['keyframes']}  "
          f"points {r['points']}  dt_med {fmt(r['dt_med'])}  dt_last100 {fmt(r['dt_last100'])}  wall {r['wall']:.0f}s")
    print(f"ATE sim3 {fmt(r['ate'])}  ATE se3 {fmt(r['ate_noscale'])}  scale_err {fmt(r['scale_err'])}  "
          f"path {fmt(r['path'], 1)}")
    print("near ratio / lost per 50 frames: " + " ".join(f"{a}:{b:.2f}/{c:.2f}" for a, b, c in r.get("series", [])))
    print("depth ratio " + "  ".join(f"{k}:{fmt(v, 2)}" for k, v in r["depth_ratio"].items()))
    for key in [k for k in r if k.startswith("map_")]:
        q = r.get(key)
        if not q:
            continue
        if "@" in key:
            print(f"   points {r['points@' + key.split('@')[1]]}  far_off(>3 m) {fmt(r.get('far_off@' + key.split('@')[1]))}")
        print(f"{key}: est {q['n_est']}  acc_med {fmt(q['acc_med'])}  prec@{TAU} {fmt(q['prec'])}  "
              f"comp@{TAU} {fmt(q['comp'])}")
        print("   " + "  ".join(f"{band_name(b)}: n{q[band_name(b)]['n_est']} acc {fmt(q[band_name(b)]['acc_med'], 2)} "
                                f"p {fmt(q[band_name(b)]['prec'], 2)} c {fmt(q[band_name(b)]['comp'], 2)}"
                                for b in BANDS))


if __name__ == "__main__":
    main()
