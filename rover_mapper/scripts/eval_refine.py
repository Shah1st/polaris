"""Does background refinement of keyframes improve the map? Offline test on TUM.

    python scripts/eval_refine.py <tum sequence dir> [n_neighbors] [refine_res] [passes]

1. Build the map exactly as live (fast settings, every 4th frame).
2. Measure keyframes against ground truth: camera positions (mocap) and
   depth maps (Kinect).
3. Re-estimate every keyframe with its nearest neighbours at a higher
   resolution (the slow job that would run in the background), measure again.

Caveat: offline, every keyframe can use neighbours from later in the run,
which a live background job only gets once the robot has been there.
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from rovermap.mapper import Mapper  # noqa: E402

seq = sys.argv[1]
n_nb = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rres = int(sys.argv[3]) if len(sys.argv) > 3 else 504
passes = int(sys.argv[4]) if len(sys.argv) > 4 else 1

rgb = [l.split() for l in open(os.path.join(seq, "rgb.txt")) if not l.startswith("#")]
dep = [l.split() for l in open(os.path.join(seq, "depth.txt")) if not l.startswith("#")]
gt = np.array([[float(x) for x in l.split()] for l in open(os.path.join(seq, "groundtruth.txt"))
               if not l.startswith("#")])
stamps = np.array([float(l[0]) for l in rgb])
dstamps = np.array([float(l[0]) for l in dep])


def umeyama(a, b):
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def kinect(fi):
    j = int(np.argmin(np.abs(dstamps - stamps[fi])))
    return cv2.imread(os.path.join(seq, dep[j][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0


m = Mapper(cam_height=1.0)
kf_frame = {}
t0 = time.time()
for i in range(0, len(rgb), 4):
    r = m.step(cv2.cvtColor(cv2.imread(os.path.join(seq, rgb[i][1])), cv2.COLOR_BGR2RGB))
    if r.new_keyframe is not None:
        kf_frame[r.new_keyframe.idx] = i
print(f"mapping done: {len(m.keyframes)} keyframes in {time.time() - t0:.0f}s")
kins = {k.idx: kinect(kf_frame[k.idx]) for k in m.keyframes}
ref = np.stack([np.interp(stamps[[kf_frame[k.idx] for k in m.keyframes]], gt[:, 0], gt[:, c])
                for c in (1, 2, 3)], 1)


def measure(label):
    est = np.array([k.c2w[:3, 3] for k in m.keyframes])
    s, R, t = umeyama(est, ref)
    ate = np.sqrt((np.linalg.norm(s * est @ R.T + t - ref, axis=1) ** 2).mean())
    absrel, ratio = [], []
    for k in m.keyframes:
        h, w = k.depth.shape
        dk = cv2.resize(kins[k.idx], (w, h), interpolation=cv2.INTER_NEAREST)
        ok = k.valid & (dk > 0)
        if ok.sum() < 100:
            continue
        absrel.append(float(np.mean(np.abs(s * k.depth[ok] - dk[ok]) / dk[ok])))
        ratio.append(float(np.median(dk[ok] / k.depth[ok])))
    ratio = np.array(ratio)
    print(f"{label:28s} keyframe ATE {ate * 100:5.2f} cm | depth error (AbsRel) {np.median(absrel) * 100:5.2f}% | "
          f"depth-scale spread between keyframes p10-p90 {100 * (np.percentile(ratio, 90) / np.percentile(ratio, 10) - 1):5.1f}%")


measure("live map")
for p in range(passes):
    t0 = time.time()
    qs = [m.refine_keyframe(k, n_neighbors=n_nb, res=rres, rebuild=False) for k in m.keyframes]
    m._rebuild_points()
    dt = (time.time() - t0) / len(m.keyframes)
    measure(f"after refine pass {p + 1}")
    print(f"   {dt:.2f}s per keyframe, {sum(q < 0.3 for q in qs)} keyframes left unchanged (neighbours disagree)")
