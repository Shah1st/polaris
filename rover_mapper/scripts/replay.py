"""Replay a recorded live session (out/rec_*.avi) as if live, and report tracking.

    python scripts/replay.py <video> [start_s] [end_s] [mapper_key=value ...]

Frames are dropped exactly as live: after each step, the video jumps ahead by
the time the step took. Black frames are skipped like run.py does.
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from rovermap.mapper import Mapper  # noqa: E402
from run import is_blank  # noqa: E402

video = sys.argv[1]
start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
end = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
extra = {}
for a in sys.argv[4:]:
    k, v = a.split("=")
    extra[k] = float(v) if "." in v else int(v)

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
m = Mapper(**({"cam_height": 0.15} | extra))
print("mapper", extra)

i = 0
rows = []
blank = 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    t = i / fps
    i += 1
    if t < start:
        continue
    if t > end:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if is_blank(rgb):
        blank += 1
        continue
    t0 = time.time()
    r = m.step(rgb)
    dt = time.time() - t0
    if r is None:
        blank += 1
        continue
    pe, re_, _ = m.last_align
    rows.append((t, r.lost, r.quality, r.overlap, len(m.keyframes), pe, re_, r.c2w[:3, 3].copy()))
    for _ in range(int(dt * fps)):  # what the live camera produced meanwhile
        cap.grab()
        i += 1

lost = np.array([r[1] for r in rows])
pe = np.array([r[5] for r in rows])
pos = np.array([r[7] for r in rows])
print(f"steps {len(rows)}  blank frames skipped {blank}  lost {int(lost.sum())} ({100 * lost.mean():.0f}%)  "
      f"keyframes {rows[-1][4]}")
streaks, c = [], 0
for l_ in lost:
    if l_:
        c += 1
    elif c:
        streaks.append(c)
        c = 0
if c:
    streaks.append(c)
print("lost streaks:", streaks)
jumps = np.linalg.norm(np.diff(pos[~lost], axis=0), axis=1) if (~lost).sum() > 1 else np.zeros(1)
print(f"pos_err tracked median {np.median(pe[~lost]):.3f}, lost median {np.median(pe[lost]) if lost.any() else 0:.3f}; "
      f"largest jump between tracked steps {jumps.max():.2f} m (median {np.median(jumps):.3f} m)")
print(f"path length {jumps.sum():.2f} m, scale: {'métrique' if m.scale_final else 'provisoire'}")
# Events: start/end of losses and big jumps, with time in the video.
prev_lost, prev_pos = False, None
for t, l_, q, ov, nkf, pe_, re2, p in rows:
    if l_ and not prev_lost:
        print(f"  {t:6.1f}s lost   (pos_err {pe_:.2f}, overlap {ov:.2f}, keyframes {nkf})")
    if not l_ and prev_lost:
        print(f"  {t:6.1f}s back   (overlap {ov:.2f})")
    if not l_ and prev_pos is not None and np.linalg.norm(p - prev_pos) > 0.2:
        print(f"  {t:6.1f}s JUMP {np.linalg.norm(p - prev_pos):.2f} m  (overlap {ov:.2f}, quality {q:.2f})")
    if not l_:
        prev_pos = p
    prev_lost = l_
m.save_ply("out/replay_map.ply")
