"""Quick speed check of Depth Anything 3 on this machine (MPS)."""
import glob
import sys
import time

import types
sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
import torch
from depth_anything_3.api import DepthAnything3

name = sys.argv[1] if len(sys.argv) > 1 else "depth-anything/DA3-SMALL"
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
m = DepthAnything3.from_pretrained(name).to(dev).eval()
imgs = sorted(glob.glob("third_party/da3/assets/examples/SOH/*"))
print(len(imgs), "images", dev)
for n in (1, 2, 4, 6):
    for res in (280, 392, 504):
        ims = (imgs * 6)[:n]
        for _ in range(2):
            t = time.time()
            p = m.inference(ims, process_res=res)
            dt = time.time() - t
        ext = None if p.extrinsics is None else p.extrinsics.shape
        print(name, "n", n, "res", res, "%.3fs" % dt, p.depth.shape, ext)
print("depth range", p.depth.min(), p.depth.max(), "conf", p.conf.min(), p.conf.max())
print(p.extrinsics[:2])
print(p.intrinsics[0])
