"""How good is DA3METRIC-LARGE as a source of metric scale? Compare with Kinect depth on TUM.

    python scripts/test_metric.py <tum sequence dir> [process_res] [n_frames]
"""
import os
import sys
import time
import types

import cv2
import numpy as np

sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")
import torch  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402

seq = sys.argv[1]
res = int(sys.argv[2]) if len(sys.argv) > 2 else 504
n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

rgb = [l.split() for l in open(os.path.join(seq, "rgb.txt")) if not l.startswith("#")]
dep = [l.split() for l in open(os.path.join(seq, "depth.txt")) if not l.startswith("#")]
dstamps = np.array([float(l[0]) for l in dep])
FX_TRUE = (517.3 + 516.5) / 2  # freiburg1, at 640 px

metric = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(dev).eval()
anyview = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to(dev).eval()

r_true, r_est, r_near, dts = [], [], [], []
for i in np.linspace(0, len(rgb) - 1, n).astype(int):
    img = cv2.cvtColor(cv2.imread(os.path.join(seq, rgb[i][1])), cv2.COLOR_BGR2RGB)
    j = int(np.argmin(np.abs(dstamps - float(rgb[i][0]))))
    kin = cv2.imread(os.path.join(seq, dep[j][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0

    t = time.time()
    p = metric.inference([img], process_res=res)
    dts.append(time.time() - t)
    out = np.asarray(p.depth[0], np.float32)
    h, w = out.shape
    fx_true = FX_TRUE * w / 640.0
    # Focal the mapper would have on the robot: estimated by the any-view model.
    q = anyview.inference([img], process_res=res)
    Kq = np.asarray(q.intrinsics[0])
    fx_est = (Kq[0, 0] + Kq[1, 1]) / 2 * w / q.depth.shape[2]

    k = cv2.resize(kin, (w, h), interpolation=cv2.INTER_NEAREST)
    ok = (k > 0) & (out > 0)
    if ok.sum() < 500:
        continue
    r_true.append(np.median(k[ok] / (fx_true * out[ok] / 300.0)))
    r_est.append(np.median(k[ok] / (fx_est * out[ok] / 300.0)))
    # Lower half of the image (floor / desk, close range) separately.
    low = ok.copy()
    low[: h // 2] = False
    if low.sum() > 200:
        r_near.append(np.median(k[low] / (fx_true * out[low] / 300.0)))


def summary(name, r):
    r = np.array(r)
    print(f"{name:34s} median {np.median(r):.3f}  p10 {np.percentile(r, 10):.3f}  "
          f"p90 {np.percentile(r, 90):.3f}  (1.000 = exact metric)")


print(f"res {res}, {len(r_true)} frames, metric model {np.mean(dts[1:]):.2f}s/frame on {dev}")
summary("kinect / metric (true focal)", r_true)
summary("kinect / metric (estimated focal)", r_est)
summary("  lower half only (true focal)", r_near)
