"""Rough CPU-only speed of DA3 (upper bound for what a Raspberry Pi 5 could do)."""
import glob
import sys
import time
import types

sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
import torch  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
imgs = sorted(glob.glob("third_party/da3/assets/examples/SOH/*"))
for name in ("depth-anything/DA3-SMALL", "depth-anything/DA3-BASE"):
    m = DepthAnything3.from_pretrained(name).to("cpu").eval()
    for n in (1, 3):
        for res in (224, 336):
            ims = (imgs * 3)[:n]
            m.inference(ims, process_res=res)
            t = time.time()
            m.inference(ims, process_res=res)
            print(f"{name} threads={torch.get_num_threads()} views={n} res={res}: {time.time() - t:.2f}s", flush=True)
