"""Run the mapper on a video file without the viewer; print stats, save a PLY and a top-down picture."""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from rovermap.mapper import Mapper  # noqa: E402

video = sys.argv[1]
step = int(sys.argv[2]) if len(sys.argv) > 2 else 5
model = sys.argv[3] if len(sys.argv) > 3 else "depth-anything/DA3-BASE"
cam_h = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
out = sys.argv[5] if len(sys.argv) > 5 else "out_test"

m = Mapper(model_name=model, cam_height=cam_h)
cap = cv2.VideoCapture(video)
print("frames", cap.get(cv2.CAP_PROP_FRAME_COUNT), "fps", cap.get(cv2.CAP_PROP_FPS),
      cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
traj, dts, i = [], [], 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    i += 1
    if i % step:
        continue
    r = m.step(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    traj.append(r.c2w[:3, 3].copy())
    dts.append(r.dt)
    if len(traj) % 10 == 0 or r.new_keyframe is not None:
        print(f"f{i:5d} dt={r.dt:.2f}s q={r.quality:.2f} kf={len(m.keyframes)} calib={m.scale_calibrated} "
              f"pos={np.round(r.c2w[:3, 3], 2)} {'KF' if r.new_keyframe else ''}")
traj = np.array(traj)
print("mean dt", np.mean(dts[1:]), "keyframes", len(m.keyframes), "calibrated", m.scale_calibrated)
n = m.save_ply(out + ".ply")
print("points", n)

# top-down render
P, C = m.all_points()
lo, hi = np.percentile(P[:, :2], 1, axis=0) - 0.3, np.percentile(P[:, :2], 99, axis=0) + 0.3
size = 900
sc = size / max(hi - lo)
img = np.full((size, size, 3), 255, np.uint8)
order = np.argsort(P[:, 2])
uv = ((P[order, :2] - lo) * sc).astype(int)
ok = (uv >= 0).all(1) & (uv < size).all(1)
img[size - 1 - uv[ok, 1], uv[ok, 0]] = C[order][ok]
tuv = ((traj[:, :2] - lo) * sc).astype(int)
for a, b in zip(tuv[:-1], tuv[1:]):
    cv2.line(img, (int(a[0]), size - 1 - int(a[1])), (int(b[0]), size - 1 - int(b[1])), (255, 0, 0), 2)
cv2.imwrite(out + "_top.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# side render (x-z)
lo2 = np.array([lo[0], np.percentile(P[:, 2], 1) - 0.2])
img2 = np.full((size // 2, size, 3), 255, np.uint8)
uv = ((P[:, [0, 2]] - lo2) * sc).astype(int)
ok = (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size // 2)
img2[size // 2 - 1 - uv[ok, 1], uv[ok, 0]] = C[ok]
cv2.imwrite(out + "_side.png", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
print("z range", np.percentile(P[:, 2], [1, 50, 99]))
