"""Long-run behaviour in an already mapped place, and reaction to an unknown place.

    python scripts/eval_passes.py <tum sequence dir> loop [passes]
    python scripts/eval_passes.py <tum sequence dir> foreign <other video> [foreign frames]

loop:    the TUM sequence N times in a row (jump back to the start each time).
         Per pass: new keyframes, frames lost, trajectory error.
foreign: TUM, then frames of an unrelated video (a place the map does not
         contain), then TUM again. During the foreign frames the mapper must
         not insert keyframes; afterwards it must find the desk again.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from rovermap.mapper import Mapper  # noqa: E402

seq, mode = sys.argv[1], sys.argv[2]
rgb = [l.split() for l in open(os.path.join(seq, "rgb.txt")) if not l.startswith("#")]
gt = np.array([[float(x) for x in l.split()] for l in open(os.path.join(seq, "groundtruth.txt"))
               if not l.startswith("#")])
stamps = np.array([float(l[0]) for l in rgb])
fwd = list(range(0, len(rgb), 4))


def tum(i):
    return cv2.cvtColor(cv2.imread(os.path.join(seq, rgb[i][1])), cv2.COLOR_BGR2RGB)


def umeyama(a, b):
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean(), R, mb - (np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()) * R @ ma


m = Mapper(cam_height=1.0)
est, ref, seg = [], [], []  # tracked TUM frames only, with their segment number


def run_tum(segment):
    added = lost = 0
    for i in fwd:
        r = m.step(tum(i))
        if r is None:
            continue
        added += r.new_keyframe is not None
        lost += r.lost
        if not r.lost:
            est.append(r.c2w[:3, 3].copy())
            ref.append([np.interp(stamps[i], gt[:, 0], gt[:, c]) for c in (1, 2, 3)])
            seg.append(segment)
    return added, lost


def report_ate():
    E, G, S = np.array(est), np.array(ref), np.array(seg)
    s, R, t = umeyama(E, G)
    err = np.linalg.norm(s * E @ R.T + t - G, axis=1)
    return {k: np.sqrt((err[S == k] ** 2).mean()) * 100 for k in np.unique(S)}


if mode == "loop":
    passes = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    stats = [run_tum(p) for p in range(passes)]
    ate = report_ate()
    for p, (added, lost) in enumerate(stats):
        print(f"pass {p + 1}: +{added:3d} keyframes (total so far {sum(a for a, _ in stats[:p + 1])}), "
              f"lost {lost:3d} / {len(fwd)}, ATE {ate.get(p, float('nan')):.1f} cm")
else:
    other = sys.argv[3]
    n_foreign = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    a1, l1 = run_tum(0)
    cap = cv2.VideoCapture(other)
    added = lost = 0
    k = 0
    while k < n_foreign:
        ok, bgr = cap.read()
        if not ok:
            break
        r = m.step(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        k += 1
        if r is None:
            continue
        added += r.new_keyframe is not None
        lost += r.lost
    a3, l3 = run_tum(2)
    ate = report_ate()
    print(f"desk pass 1:      +{a1} keyframes, lost {l1} / {len(fwd)}, ATE {ate[0]:.1f} cm")
    print(f"unknown place:    +{added} keyframes, lost {lost} / {k}   (want: 0 keyframes, all lost)")
    print(f"desk pass 2:      +{a3} keyframes, lost {l3} / {len(fwd)}, ATE {ate.get(2, float('nan')):.1f} cm")
