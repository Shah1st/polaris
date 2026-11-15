"""Record what the live simulator serves on /snapshot, to replay a mapping run
offline (scripts/vm/eval_forest.py) without touching the running demo.

    python scripts/vm/record_snapshots.py out/vm/forest_seq1 --seconds 150 --hz 3

One .npz per frame: rgb, exact depth, K, true pose T, D405 depth + K + T, t.
"""
import argparse
import io
import os
import time

import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--sim", default="http://127.0.0.1:5050")
ap.add_argument("--seconds", type=float, default=150.0)
ap.add_argument("--hz", type=float, default=3.0)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
keep = ("t", "rover", "rgb", "depth", "K", "T", "d405_depth", "d405_K", "d405_T", "d405_mount")
sess = requests.Session()
t_end, n = time.time() + args.seconds, 0
while time.time() < t_end:
    t_next = time.time() + 1.0 / args.hz
    z = np.load(io.BytesIO(sess.get(args.sim.rstrip("/") + "/snapshot", timeout=10).content))
    np.savez_compressed(os.path.join(args.out, f"{n:05d}.npz"), **{k: z[k] for k in keep})
    n += 1
    if n % 50 == 0:
        print(n, "frames", flush=True)
    time.sleep(max(0.0, t_next - time.time()))
print("done:", n, "frames in", args.out, flush=True)
