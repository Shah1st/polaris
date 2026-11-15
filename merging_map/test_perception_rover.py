"""Tests de perception_rover.py : geometrie repere rover -> carte, et client HTTP contre un faux
service de perception (127.0.0.1, ports 8130-8139). Ni GPU ni modele."""

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from fusion_carte import CarteFusion, decouper
from perception_rover import RES, percevoir_http, vers_carte

R = 0.05


def grille_rover(obstacles=(), libres=(), portee=1.0):
    """Grille du service : x dans [0, portee], y dans [-portee, portee] ; listes de (x, y)."""
    n = round(portee / R)
    m = np.full((2 * n, n), -1, np.int8)
    for v, pts in ((0, libres), (1, obstacles)):
        for x, y in pts:
            m[math.floor((y + portee) / R), math.floor(x / R)] = v
    return m, 0.0, -portee


def valeur(vue, x, y):
    m, x0, y0 = vue
    i, j = math.floor((y - y0) / R + 1e-9), math.floor((x - x0) / R + 1e-9)
    if not (0 <= i < m.shape[0] and 0 <= j < m.shape[1]):
        return -1
    return int(m[i, j])


def multiple(v):
    return abs(v / RES - round(v / RES)) < 1e-6


def test_cap_nul_a_l_origine_garde_les_cases():
    m, x0, y0 = grille_rover(obstacles=[(0.52, 0.02)], libres=[(0.32, 0.02)])
    vue = vers_carte(m, x0, y0, R, (0.0, 0.0), 0.0)
    assert valeur(vue, 0.52, 0.02) == 1
    assert valeur(vue, 0.32, 0.02) == 0
    assert valeur(vue, 0.72, 0.02) == -1
    assert (vue[0] == 1).sum() == 1 and (vue[0] == 0).sum() == 1


def test_rotation_et_translation():
    m, x0, y0 = grille_rover(obstacles=[(0.52, 0.02)], libres=[(0.32, -0.27)])
    # cap 90 deg : l'avant du rover est +Y de la carte, sa gauche est -X
    vue = vers_carte(m, x0, y0, R, (1.0, 2.0), math.pi / 2)
    assert valeur(vue, 1.0 - 0.02, 2.0 + 0.52) == 1
    assert valeur(vue, 1.0 + 0.27, 2.0 + 0.32) == 0
    assert (vue[0] == 1).sum() == 1
    # cap 180 deg
    vue = vers_carte(m, x0, y0, R, (1.0, 2.0), math.pi)
    assert valeur(vue, 1.0 - 0.52, 2.0 - 0.02) == 1


def test_angle_quelconque_et_position_hors_grille():
    obst = [(0.4 + 0.05 * k, 0.1) for k in range(6)]
    libres = [(0.3, y) for y in (-0.2, -0.1, 0.0)]
    m, x0, y0 = grille_rover(obstacles=obst, libres=libres)
    pos, cap = (3.017, -1.233), math.radians(-37.0)
    mat, gx0, gy0 = vers_carte(m, x0, y0, R, pos, cap)
    assert multiple(gx0) and multiple(gy0)
    assert mat.dtype == np.int8 and set(np.unique(mat)) <= {-1, 0, 1}
    c, s = math.cos(cap), math.sin(cap)
    for x, y in obst:  # chaque obstacle tombe sur une case obstacle de la carte
        xc = (math.floor(x / R) + 0.5) * R  # centre de la case rover
        yc = (math.floor((y + 1.0) / R) + 0.5) * R - 1.0
        xm, ym = pos[0] + c * xc - s * yc, pos[1] + s * xc + c * yc
        assert valeur((mat, gx0, gy0), xm, ym) == 1
    # rien de libre la ou le rover ne voit rien ; libre seulement pres des cases libres
    assert (mat == 0).sum() <= 2 * len(libres) + 2
    assert (mat == 1).sum() >= len(obst)


def test_obstacle_fin_a_45_degres_jamais_perdu():
    for k in range(40):
        cap = math.radians(45 + k * 0.7)
        pos = (0.013 * k, -0.007 * k)
        m, x0, y0 = grille_rover(obstacles=[(0.62, 0.12)])
        mat, gx0, gy0 = vers_carte(m, x0, y0, R, pos, cap)
        assert (mat == 1).sum() >= 1


def test_inconnu_reste_inconnu_et_jamais_zero():
    m = np.full((40, 20), -1, np.int8)
    mat, _, _ = vers_carte(m, 0.0, -1.0, R, (0.5, 0.5), 0.3)
    assert (mat == -1).all()


def test_publie_dans_la_fusion_au_bon_endroit():
    carte = CarteFusion()
    m, x0, y0 = grille_rover(obstacles=[(0.52, 0.02)], libres=[(0.27, 0.02)])
    mat, gx0, gy0 = vers_carte(m, x0, y0, R, (1.0, 1.0), 0.0)
    for msg in decouper(mat, gx0, gy0, carte.zmax, carte, 'rover', 1.0):
        carte.recevoir(msg)
    g = carte.grille(1.0, 1.0, 2.0, 1.1).val  # lignes Y 1.00-1.10, colonnes X 1.00-2.00
    assert g[0, 10] == 1  # X = 1.50-1.55
    assert g[0, 5] == 0  # X = 1.25-1.30


# ---- client HTTP contre un faux service ---------------------------------------------------


class FauxService:
    def __init__(self, port):
        self.reponse = None  # (code, corps)
        self.attente = 0.0
        self.demandes = []
        service = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                service.demandes.append(self.path)
                time.sleep(service.attente)
                code, corps = service.reponse
                b = corps if isinstance(corps, bytes) else json.dumps(corps).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.srv = ThreadingHTTPServer(('127.0.0.1', port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{port}'

    def fermer(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def service():
    s = FauxService(8131)
    yield s
    s.fermer()


def corps(age=0.4, **extra):
    m, x0, y0 = grille_rover(obstacles=[(0.52, 0.02)], libres=[(0.27, 0.02)])
    b = {'t': time.time() - age, 'age_s': age, 'res': R, 'x0': x0, 'y0': y0, 'cells': m.tolist()}
    b.update(extra)
    return b


def test_client_place_la_vue(service):
    service.reponse = (200, corps())
    percevoir = percevoir_http(service.url)
    vue = percevoir((1.0, 1.0), math.pi / 2)
    assert vue is not None
    assert valeur(vue, 1.0 - 0.02, 1.0 + 0.52) == 1
    assert valeur(vue, 1.0 - 0.02, 1.0 + 0.27) == 0
    assert service.demandes == ['/obstacles?fresh=1']


def test_client_avance_camera(service):
    service.reponse = (200, corps())
    vue = percevoir_http(service.url, avance_camera=0.1)((1.0, 1.0), 0.0)
    assert valeur(vue, 1.0 + 0.1 + 0.52, 1.02) == 1


def test_client_none_si_cap_inconnu(service):
    service.reponse = (200, corps())
    assert percevoir_http(service.url)((1.0, 1.0), None) is None
    assert service.demandes == []


@pytest.mark.parametrize(
    'reponse',
    [
        (503, {'error': 'no frame'}),
        (500, {'error': 'boom'}),
        (200, b'pas du json'),
        (200, {'age_s': 0.1}),  # pas de cases
        (200, {'res': R, 'x0': 0, 'y0': 0, 'cells': [[1]]}),  # pas d'age
    ],
)
def test_client_none_si_erreur(service, reponse):
    service.reponse = reponse
    assert percevoir_http(service.url)((0.0, 0.0), 0.0) is None


def test_client_none_si_image_trop_vieille(service):
    service.reponse = (200, corps(age=2.5))
    assert percevoir_http(service.url, age_max=2.0)((0.0, 0.0), 0.0) is None
    service.reponse = (200, corps(age=1.5))
    assert percevoir_http(service.url, age_max=2.0)((0.0, 0.0), 0.0) is not None


def test_client_none_si_trop_lent(service):
    service.reponse = (200, corps())
    service.attente = 0.6
    t0 = time.time()
    assert percevoir_http(service.url, delai=0.2)((0.0, 0.0), 0.0) is None
    assert time.time() - t0 < 0.6


def test_client_none_sans_service():
    assert percevoir_http('http://127.0.0.1:8139', delai=0.5)((0.0, 0.0), 0.0) is None


def test_branche_dans_rasprover_reel(service):
    """observer() de RaspRoverReel publie la vue du service dans la fusion."""
    from rover_reel import RaspRoverReel

    service.reponse = (200, corps())
    rover = RaspRoverReel.__new__(RaspRoverReel)  # seulement observer() / publier_vue()
    rover.percevoir = percevoir_http(service.url)
    rover.est, rover.cap = (1.0, 1.0), 0.0
    publiees = []
    rover.publier_vue = lambda loc, x0, y0: publiees.append((loc, x0, y0))
    rover.observer()
    assert len(publiees) == 1
    assert valeur(publiees[0], 1.52, 1.02) == 1
