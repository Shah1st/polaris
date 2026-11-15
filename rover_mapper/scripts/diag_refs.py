"""Why are the references disagreeing? Replay a recording until the first long
loss, then for a few failing frames align the frame on each reference alone.

    python scripts/diag_refs.py <video> [start_s] [end_s]
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from rovermap.mapper import Mapper, rot_angle_deg  # noqa: E402
from run import is_blank  # noqa: E402

video = sys.argv[1]
start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
end = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
cap = cv2.VideoCapture(video)
fps = 30.0
m = Mapper(cam_height=0.15)
i, n_lost, shown = 0, 0, 0
while shown < 4:
    ok, bgr = cap.read()
    if not ok:
        break
    t = i / fps
    i += 1
    if t < start or t > end:
        if t > end:
            break
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if is_blank(rgb):
        continue
    refs = m._select_refs() if m.keyframes else []
    t0 = time.time()
    r = m.step(rgb)
    if r.new_keyframe is not None:
        k = r.new_keyframe
        g = cv2.cvtColor(k.image, cv2.COLOR_RGB2GRAY)
        print(f"t={t:5.1f}s  new keyframe #{k.idx} at {np.round(k.c2w[:3, 3], 2)}  "
              f"mean {g.mean():.0f} sharp {cv2.Laplacian(g, cv2.CV_32F).var():.0f} valid {k.valid.mean():.2f}")
    if r.lost:
        n_lost += 1
        if n_lost in (1, 5, 20, 60):
            shown += 1
            print(f"\n--- lost frame t={t:.1f}s, refs {[k.idx for k in refs]}, pos_err {m.last_align[0]:.3f}, "
                  f"rot_err {m.last_align[1]:.1f}, scale {m.last_align[2]:.3f}, overlap {r.overlap:.2f}")
            depth, conf, L, K, imgs = m._infer([k.rgb for k in refs] + [rgb])
            for j, k in enumerate(refs):
                c2w, s, q = m._align([k], depth[[j, len(refs)]], L[[j, len(refs)]])
                d = depth[len(refs)] * s
                v = m._valid_mask(d, conf[len(refs)])
                ov = max(m._overlap(kk, c2w, d, v, K[len(refs)]) for kk in refs)
                # ratio stored depth / fresh depth for this ref
                ok_ = k.valid & (depth[j] > 1e-6)
                ratio = np.median(k.depth[ok_] / depth[j][ok_]) if ok_.sum() > 100 else float("nan")
                print(f"   ref #{k.idx}: pose from it alone {np.round(c2w[:3, 3], 2)}  scale {s:.3f} "
                      f"(ratio {ratio:.3f})  overlap {ov:.2f}  "
                      f"dist to ref {np.linalg.norm(c2w[:3, 3] - k.c2w[:3, 3]):.2f}  "
                      f"angle to ref {rot_angle_deg(c2w[:3, :3], k.c2w[:3, :3]):.0f} deg")
    for _ in range(int((time.time() - t0) * fps)):
        cap.grab()
        i += 1
