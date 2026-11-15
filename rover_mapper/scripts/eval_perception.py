"""How good is the single-image rover perception (rovermap/perception.py + DA3)?

On simulator frames with exact depth, the same classification is applied twice: to DA3's depth
(scaled by the floor and the known camera height, as perception_service.py does) and to the
true depth (metric). Reported, per distance band of the cells: obstacle precision / recall
(exact cell, and within one cell), the share of cells called free that are truly obstacles
(the dangerous error: a rover 0 erases the drone's obstacles), and the depth scale error.
Both intrinsics are tried on the same inference: DA3's own and the known K.

    CUDA_VISIBLE_DEVICES=0 /home/romaindcrl/rovermap/.venv/bin/python \\
        rover_mapper/scripts/eval_perception.py --out /tmp/eval_perception.json

Frames: --forest 'glob of .npz' (recorded forest route: keys rgb, depth, K, T) and/or
--snapshots URL (the simulator's /snapshot, GET only, --n of them, --every s apart, saved in
--cache to replay). It has NEVER seen the real rover's camera.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rovermap import perception as per  # noqa: E402

BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]


def load_frames(args):
    frames = []
    if args.forest:
        files = sorted(glob.glob(args.forest))
        step = max(1, len(files) // args.n)
        for f in files[::step][: args.n]:
            with np.load(f) as z:
                frames.append(('forest', os.path.basename(f),
                               {k: z[k] for k in ('rgb', 'depth', 'K', 'T')}))
    if args.snapshots:
        os.makedirs(args.cache, exist_ok=True)
        cached = sorted(glob.glob(os.path.join(args.cache, 'snap_*.npz')))
        if len(cached) < args.n:
            import requests

            for k in range(len(cached), args.n):
                r = requests.get(args.snapshots, timeout=10)
                r.raise_for_status()
                with np.load(io.BytesIO(r.content)) as z:
                    keep = {k2: z[k2] for k2 in ('rgb', 'depth', 'K', 'T')}
                np.savez_compressed(os.path.join(args.cache, f'snap_{k:03d}.npz'), **keep)
                time.sleep(args.every)
            cached = sorted(glob.glob(os.path.join(args.cache, 'snap_*.npz')))
        for f in cached[: args.n]:
            with np.load(f) as z:
                frames.append(('apartment', os.path.basename(f), {k: z[k] for k in z.files}))
    return frames


def cell_dist(p: per.Perception) -> np.ndarray:
    r, c = p.cells.shape
    xc = p.x0 + (np.arange(c) + 0.5) * p.res
    yc = p.y0 + (np.arange(r) + 0.5) * p.res
    return np.hypot(*np.meshgrid(xc, yc))


def compare(pred: per.Perception, true: per.Perception) -> dict:
    """Counts per distance band (both grids share their extent: same range and res)."""
    d = cell_dist(true)
    po, to = pred.cells == per.OBSTACLE, true.cells == per.OBSTACLE
    pf, tf = pred.cells == per.FREE, true.cells == per.FREE
    k = np.ones((3, 3), np.uint8)
    to_d = cv2.dilate(to.astype(np.uint8), k) > 0
    po_d = cv2.dilate(po.astype(np.uint8), k) > 0
    out = {}
    for lo, hi in BANDS:
        b = (d >= lo) & (d < hi)
        out[f'{lo}-{hi}'] = {
            'pred_obst': int((po & b).sum()),
            'true_obst': int((to & b).sum()),
            'tp': int((po & to & b).sum()),
            'tp_prec_tol': int((po & to_d & b).sum()),
            'tp_rec_tol': int((to & po_d & b).sum()),
            'pred_free': int((pf & b).sum()),
            'free_on_obst': int((pf & to & b).sum()),
            'free_on_free': int((pf & tf & b).sum()),
            'true_free': int((tf & b).sum()),
        }
    return out


def scale_stats(depth_net, scale, truth, valid, max_range):
    t = cv2.resize(truth, depth_net.shape[::-1], interpolation=cv2.INTER_NEAREST)
    ok = valid & np.isfinite(t) & (t > 0.05)
    out = {}
    for lo, hi in [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)]:
        m = ok & (t >= lo) & (t < hi)
        if m.sum() > 50:
            out[f'{lo}-{hi}'] = float(np.median(depth_net[m] * scale / t[m]))
    return out


def summarise(rows, key):
    tot = {}
    for r in rows:
        for band, c in r[key].items():
            acc = tot.setdefault(band, {})
            for k, v in c.items():
                acc[k] = acc.get(k, 0) + v

    def ratio(a, b):
        return None if b == 0 else round(a / b, 3)

    return {
        band: {
            'precision': ratio(c['tp'], c['pred_obst']),
            'recall': ratio(c['tp'], c['true_obst']),
            'precision_1cell': ratio(c['tp_prec_tol'], c['pred_obst']),
            'recall_1cell': ratio(c['tp_rec_tol'], c['true_obst']),
            'free_on_obstacle': ratio(c['free_on_obst'], c['pred_free']),
            'free_correct': ratio(c['free_on_free'], c['pred_free']),
            'free_recall': ratio(c['free_on_free'], c['true_free']),
            'true_obst_cells': c['true_obst'],
            'pred_obst_cells': c['pred_obst'],
        }
        for band, c in tot.items()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--forest', default='/home/romaindcrl/rovermap/out/vm/forest_seq1/*.npz')
    ap.add_argument('--snapshots', default='http://127.0.0.1:8090/snapshot')
    ap.add_argument('--cache', default='/tmp/eval_perception_snaps')
    ap.add_argument('--n', type=int, default=30, help='frames per set')
    ap.add_argument('--every', type=float, default=1.5, help='s between snapshots')
    ap.add_argument('--cam-height', type=float, default=0.2116,
                    help='nominal camera height (the simulated rover: 0.2116 m)')
    ap.add_argument('--range', type=float, default=2.0)
    ap.add_argument('--res', type=int, default=504)
    ap.add_argument('--model', default='depth-anything/DA3-BASE')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    frames = load_frames(args)
    print(f'{len(frames)} images', flush=True)
    from rovermap.mapper import Mapper, net_intrinsics

    mapper = Mapper(model_name=args.model, process_res=args.res, cam_height=args.cam_height,
                    mask_moving=False, cpu_threads=1)
    rows = []
    for kind, name, f in frames:
        rgb, depth_t, K = f['rgb'], f['depth'].astype(np.float32), f['K']
        true_valid = per.valid_depth(depth_t)
        truth = per.perceive(depth_t, K, cam_height=args.cam_height, valid=true_valid,
                             scale=1.0, max_range=args.range)
        t0 = time.time()
        d, conf, _, k_net, _ = mapper._infer([rgb])
        dt = time.time() - t0
        d, conf = d[0], conf[0]
        valid = per.valid_depth(d, conf)
        row = {'set': kind, 'frame': name, 'infer_s': round(dt, 3), 'truth_floor': truth.floor,
               'true_cam_height': truth.stats.get('floor_height_m')}
        for label, Kn in (('da3K', k_net[0]),
                          ('knownK', net_intrinsics(K, rgb.shape[:2], d.shape))):
            p = per.perceive(d, Kn, cam_height=args.cam_height, valid=valid,
                             max_range=args.range)
            row[label] = {
                'floor': p.floor,
                'tilt_deg': p.stats.get('floor_tilt_deg'),
                'scale_ratio': scale_stats(d, p.scale, depth_t, valid, args.range)
                if p.floor else {},
                'cells': compare(p, truth) if (p.floor and truth.floor) else {},
                'fx': float(Kn[0, 0]),
            }
        rows.append(row)
        print(kind, name, f'{dt:.2f}s', 'truth floor', truth.floor,
              {lab: (row[lab]['floor'], row[lab]['scale_ratio']) for lab in ('da3K', 'knownK')},
              flush=True)

    report = {'args': vars(args), 'frames': len(rows)}
    for kind in sorted({r['set'] for r in rows}):
        rs = [r for r in rows if r['set'] == kind]
        rep = {'images': len(rs), 'truth_floor_found': sum(r['truth_floor'] for r in rs),
               'infer_s_median': float(np.median([r['infer_s'] for r in rs]))}
        for lab in ('da3K', 'knownK'):
            ok = [r for r in rs if r[lab]['cells']]
            sc = {}
            for r in rs:
                for b, v in r[lab]['scale_ratio'].items():
                    sc.setdefault(b, []).append(v)
            rep[lab] = {
                'floor_found': sum(r[lab]['floor'] for r in rs),
                'scale_ratio_median': {b: round(float(np.median(v)), 3) for b, v in sc.items()},
                'scale_ratio_p10_p90': {b: [round(float(np.percentile(v, 10)), 3),
                                            round(float(np.percentile(v, 90)), 3)]
                                        for b, v in sc.items()},
                'cells': summarise([{'c': r[lab]['cells']} for r in ok], 'c'),
            }
        report[kind] = rep
    print(json.dumps({k: v for k, v in report.items() if k != 'args'}, indent=1))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'report': report, 'rows': rows}, fh, indent=1)


if __name__ == '__main__':
    main()
