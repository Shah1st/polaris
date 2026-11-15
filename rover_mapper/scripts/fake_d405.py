"""Pretend to be the D405 capture server of the rover (d405_server.py, same HTTP contract)
from the apartment simulator's GET /snapshot, to try run.py --d405 without the robot.

    python scripts/fake_d405.py [--sim http://127.0.0.1:8090] [--port 8246]
    python run.py --d405 http://127.0.0.1:8246 --sensors d405 --d405-height 0.06 ...

GET /info, /frame (?fresh=1), /color.jpg, /depth.png, /health, as on the robot. In the
simulator the D405 colour and depth are one camera: K_depth = K_color, R_dc = identity,
t_dc = 0; depth_scale 0.0001 m (the real D405's default unit). A test instance: at most
2 snapshots per second (each one renders the simulated D405 too). Never point it at the
forest simulator on 5050: a live demo uses it.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import requests

DEPTH_SCALE = 1e-4


class Frames:
    """Newest simulated D405 frame, encoded as the robot would send it."""

    def __init__(self, sim: str, hz: float):
        self.sim, self.period = sim.rstrip("/"), 1.0 / min(hz, 2.0)
        self.cond = threading.Condition()
        self.frame: dict | None = None
        self.t = None               # time.monotonic() of the capture
        self.seq = 0
        self.errors, self.last_error = 0, None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        sess = requests.Session()
        while True:
            t0 = time.monotonic()
            try:
                r = sess.get(self.sim + "/snapshot", timeout=10)
                r.raise_for_status()
                z = np.load(io.BytesIO(r.content))
                t_cap = time.monotonic()
                K = np.asarray(z["d405_K"], np.float64)
                raw = np.clip(np.round(z["d405_depth"].astype(np.float64) / DEPTH_SCALE), 0, 65535).astype(np.uint16)
                # Simulator images are RGB; the contract's JPEG is in OpenCV (BGR) order.
                jpg = cv2.imencode(".jpg", cv2.cvtColor(z["d405_rgb"], cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
                png = cv2.imencode(".png", raw, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1]
                f = {"color_jpg": jpg.ravel(), "depth_png": png.ravel(), "depth_scale": np.float64(DEPTH_SCALE),
                     "K_color": K, "K_depth": K.copy(), "R_dc": np.eye(3), "t_dc": np.zeros(3),
                     "w": raw.shape[1], "h": raw.shape[0]}
                with self.cond:
                    self.frame, self.t, self.seq = f, t_cap, self.seq + 1
                    self.cond.notify_all()
            except Exception as e:  # simulator not up / restarted: retry
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))

    def get(self, fresh: bool, max_age: float):
        """(frame, age_s, seq) or (None, error)."""
        t_req = time.monotonic()
        with self.cond:
            if fresh:
                self.cond.wait_for(lambda: self.t is not None and self.t > t_req, timeout=2.0)
            if self.frame is None:
                return None, "pas d'image du simulateur" + (f" ({self.last_error})" if self.last_error else "")
            age = time.monotonic() - self.t
            if age > max_age or (fresh and self.t <= t_req):
                return None, f"dernière image trop ancienne ({age:.1f} s)"
            return (self.frame, age, self.seq), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", default="http://127.0.0.1:8090", help="simulateur de l'appartement (sim/run_world.py)")
    ap.add_argument("--port", type=int, default=8246)
    ap.add_argument("--hz", type=float, default=2.0, help="instantanés demandés par seconde (2 au plus)")
    ap.add_argument("--max-age", type=float, default=2.0, help="503 si la dernière image est plus vieille (s)")
    args = ap.parse_args()
    if urlparse(args.sim).port == 5050:
        ap.error("pas le simulateur de la forêt sur 5050 : une démo en direct l'utilise")
    frames = Frames(args.sim, args.hz)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str, age: float | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if age is not None:
                self.send_header("X-Age-Ms", f"{1000 * age:.0f}")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            fresh = parse_qs(u.query).get("fresh", ["0"])[0] == "1"
            if u.path == "/health":
                with frames.cond:
                    age = None if frames.t is None else time.monotonic() - frames.t
                    f = frames.frame
                self._json(200, {"ok": age is not None and age <= args.max_age, "device": f is not None,
                                 "age_s": age, "frames": frames.seq, "errors": frames.errors,
                                 "last_error": frames.last_error,
                                 "mode": f"{f['w']}x{f['h']}@{args.hz:g}" if f else None})
                return
            if u.path not in ("/info", "/frame", "/color.jpg", "/depth.png"):
                self._json(404, {"error": "inconnu"})
                return
            got, err = frames.get(fresh, args.max_age)
            if got is None:
                self._json(503, {"error": err})
                return
            f, age, seq = got
            if u.path == "/info":
                Kl = f["K_color"].tolist()
                self._json(200, {"name": "D405 simulée (fake_d405.py)", "serial": "sim", "firmware": "sim",
                                 "usb": "sim", "fps": args.hz,
                                 "depth": {"width": f["w"], "height": f["h"], "K": Kl},
                                 "color": {"width": f["w"], "height": f["h"], "K": Kl},
                                 "depth_scale_m": DEPTH_SCALE,
                                 "depth_to_color": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}})
            elif u.path == "/frame":
                buf = io.BytesIO()
                np.savez(buf, **{k: f[k] for k in ("color_jpg", "depth_png", "depth_scale", "K_color", "K_depth",
                                                   "R_dc", "t_dc")},
                         age_s=np.float64(age), seq=np.int64(seq))
                self._send(200, buf.getvalue(), "application/x-npz", age)
            elif u.path == "/color.jpg":
                self._send(200, f["color_jpg"].tobytes(), "image/jpeg", age)
            else:
                self._send(200, f["depth_png"].tobytes(), "image/png", age)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    srv.daemon_threads = True
    print(f"D405 simulée sur http://127.0.0.1:{args.port} (depuis {args.sim}/snapshot)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
