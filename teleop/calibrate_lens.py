"""Estimate the rover camera's fisheye model from straight edges, without a calibration target.

    python calibrate_lens.py frame1.jpg frame2.jpg ... [--out calib/plumbline.json]
    python calibrate_lens.py --grab http://127.0.0.1:8082 [--n 6]   # frames from the dashboard

The lens bends straight lines (counter edges, wall boards, pillars). Model: equidistant fisheye
with one distortion term, r = f * theta * (1 + k1 * theta^2), principal point (cx, cy). The four
numbers are the ones that make the edge chains found in the pictures straightest ("plumb line"
calibration). server.py uses the result to serve /video_rect, the pinhole picture the mapper's
network expects.

Limits: an estimate, good enough to straighten the picture; a checkerboard calibration is more
accurate. Use frames with many long straight edges, from several viewpoints if possible, and
LOOK at the rectified picture written next to the result.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from scipy.optimize import minimize


def edge_chains(gray: np.ndarray, min_extent: int = 70) -> list[np.ndarray]:
    """Thin, long connected edge chains of one picture, as (N, 2) pixel coordinates."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 90:
            continue
        ys, xs = np.nonzero(lab == i)
        extent = max(np.ptp(xs), np.ptp(ys))
        if extent < min_extent or len(xs) > 2.2 * extent:  # short, or a blob rather than a line
            continue
        out.append(np.stack([xs, ys], 1).astype(np.float64))
    return out


def undistort(P: np.ndarray, f: float, k1: float, cx: float, cy: float) -> np.ndarray:
    """Pixels -> pinhole coordinates under the model (Newton steps on theta)."""
    x, y = (P[:, 0] - cx) / f, (P[:, 1] - cy) / f
    r = np.hypot(x, y) + 1e-12
    th = r.copy()
    for _ in range(8):
        th = th - (th * (1 + k1 * th**2) - r) / (1 + 3 * k1 * th**2)
    s = np.tan(np.clip(th, 0, 1.5)) / r
    return np.stack([x * s, y * s], 1)


def straightness(p, chains: list[np.ndarray], keep: float = 0.6) -> float:
    """Mean thickness / length of the straightest `keep` share of the chains (0 = all straight).
    The rest are corners and curved objects, which no lens model straightens."""
    res = []
    for P in chains:
        Q = undistort(P, *p)
        sv = np.linalg.svd(Q - Q.mean(0), compute_uv=False)
        res.append(sv[1] / max(sv[0], 1e-9))
    res = np.sort(res)
    return float(np.mean(res[: int(keep * len(res))]))


def fit(chains: list[np.ndarray], size: tuple[int, int]) -> tuple[np.ndarray, float]:
    w, h = size

    def cost(p):
        f, k1, cx, cy = p
        # The principal point stays near the centre, and f in the range where no ray is clipped:
        # without this the search finds a degenerate, very short focal length.
        bounds = 10.0 if not 0.35 * w < f < 0.55 * w or abs(k1) > 0.3 else 0.0
        return straightness(p, chains) + 1e-6 * ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) + bounds

    # Several starts: f and k1 partly trade off, so the search has close local optima.
    starts = [(f0 * w, k0, w / 2, h / 2) for f0 in (0.38, 0.40, 0.42, 0.44) for k0 in (0.0, -0.03)]
    best = min(
        (
            minimize(
                cost,
                s,
                method='Nelder-Mead',
                options={'xatol': 0.3, 'fatol': 1e-7, 'maxiter': 800},
            )
            for s in starts
        ),
        key=lambda r: r.fun,
    )
    return best.x, straightness(best.x, chains)


def rectify_maps(model: dict, fov_deg: float = 100.0):
    """Remap tables to a pinhole picture of `fov_deg` degrees horizontally, and its K."""
    w, h = model['size']
    K = np.array([[model['f'], 0, model['cx']], [0, model['f'], model['cy']], [0, 0, 1.0]])
    fo = (w / 2) / np.tan(np.radians(fov_deg / 2))
    Kn = np.array([[fo, 0, w / 2], [0, fo, h / 2], [0, 0, 1.0]])
    maps = cv2.fisheye.initUndistortRectifyMap(
        K, np.array([model['k1'], 0, 0, 0.0]), np.eye(3), Kn, (w, h), cv2.CV_16SC2
    )
    return maps, Kn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('frames', nargs='*', help='pictures from the rover camera')
    ap.add_argument('--grab', help='dashboard URL to take full-size frames from (/frame.jpg?p=hd)')
    ap.add_argument('--n', type=int, default=6, help='frames to grab, 1.5 s apart')
    ap.add_argument(
        '--out',
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'calib', 'plumbline.json'
        ),
    )
    args = ap.parse_args()

    images = [cv2.imread(f) for f in args.frames]
    if args.grab:
        import requests

        for _ in range(args.n):
            r = requests.get(args.grab.rstrip('/') + '/frame.jpg?c=calib&p=hd', timeout=5)
            if r.status_code == 200:
                images.append(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR))
            time.sleep(1.5)
    images = [im for im in images if im is not None]
    if not images:
        ap.error('no picture: give files or --grab URL')
    h, w = images[0].shape[:2]
    chains = [c for im in images for c in edge_chains(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))]
    raw = straightness((50 * w, 0.0, w / 2, h / 2), chains)  # a very long focal: no correction
    p, cost = fit(chains, (w, h))
    model = {
        'model': 'theta_d = theta*(1+k1*theta^2), r = f*theta_d',
        'f': float(p[0]),
        'k1': float(p[1]),
        'cx': float(p[2]),
        'cy': float(p[3]),
        'size': [w, h],
        'cost': cost,
        'cost_raw': raw,
        'frames': len(images),
        'chains': len(chains),
        'method': 'plumb-line on Canny edge chains, no calibration target',
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(model, f, indent=1)
    maps, _ = rectify_maps(model)
    check = os.path.splitext(args.out)[0] + '_check.jpg'
    cv2.imwrite(check, cv2.remap(images[0], maps[0], maps[1], cv2.INTER_LINEAR))
    print(
        f'f={p[0]:.1f} px  k1={p[1]:.4f}  centre=({p[2]:.1f}, {p[3]:.1f})  '
        f'straightness {raw:.4f} -> {cost:.4f}  ({len(chains)} edge chains, {len(images)} frames)'
    )
    print(f'written: {args.out}\nlook at: {check}')


if __name__ == '__main__':
    main()
