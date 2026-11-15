"""perception_service.py --d405: obstacles from the D405's measured depth only (no model).
A stand-in capture server gives a synthetic scene: a floor 6 cm under the lens and a box
20 cm wide and 15 cm high, 50 cm ahead."""

import argparse
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import perception_service as ps

W, H, F, HAUTEUR, ECHELLE = 640, 480, 380.0, 0.06, 0.0001


def scene():
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x, y = (u - W / 2) / F, (v - H / 2) / F  # ray (x, y, 1), y down
    z = np.zeros((H, W), np.float32)
    sol = y > 0.02
    z[sol] = HAUTEUR / y[sol]
    z[z > 3.0] = 0
    boite = (np.abs(x * 0.5) <= 0.10) & (y * 0.5 <= HAUTEUR) & (y * 0.5 >= HAUTEUR - 0.15)
    z[boite & ((z == 0) | (z > 0.5))] = 0.5
    return z


def serveur(z):
    buf = io.BytesIO()
    np.savez(
        buf,
        color_jpg=np.frombuffer(cv2.imencode('.jpg', np.zeros((H, W, 3), np.uint8))[1], np.uint8),
        depth_png=np.frombuffer(
            cv2.imencode('.png', np.round(z / ECHELLE).astype(np.uint16))[1], np.uint8
        ),
        depth_scale=np.float64(ECHELLE),
        K_color=np.eye(3),
        K_depth=np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]]),
        R_dc=np.eye(3),
        t_dc=np.zeros(3),
        age_s=np.float64(0.0),
        seq=np.int64(1),
    )
    corps = buf.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)

    srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_d405_seule_voit_la_boite_et_le_sol():
    srv = serveur(scene())
    try:
        url = f'http://127.0.0.1:{srv.server_address[1]}'
        args = argparse.Namespace(
            d405=url, d405_min=0.07, d405_max=3.0, cam_height=HAUTEUR, range=1.5,
            min_range=0.0, band_low=0.04, band_high=0.40,
        )  # fmt: skip
        cam = ps.D405Camera(url)
        out = ps.Perceiver(args).run(cam.grab(0.0, 2.0))
    finally:
        srv.shutdown()
    r = out['result']
    assert r.floor and r.scale == 1.0
    cells = np.asarray(r.cells)
    ii, jj = np.nonzero(cells == 1)
    xs, ys = r.x0 + (jj + 0.5) * r.res, r.y0 + (ii + 0.5) * r.res
    assert len(xs) >= 3  # the front face of the box: about 4 cells of 5 cm
    assert abs(float(np.median(xs)) - 0.5) < 0.06  # 50 cm ahead
    assert float(np.abs(ys).max()) < 0.20  # 20 cm wide, centred
    libres = np.nonzero(cells == 0)
    assert len(libres[0]) > 20  # the floor in front of it is seen free
    assert float((r.x0 + (libres[1] + 0.5) * r.res).min()) < 0.45
