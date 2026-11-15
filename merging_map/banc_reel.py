"""
banc_reel.py - La chaine reelle sans materiel, pour test_chaine_reelle.py : le VRAI serveur du
rover (rover/rover_server.py + rover_core.py) avec l'ESP32 simule (rover/fake_esp32.py), et une
camera du dessus simulee EN DIRECT, servie en HTTP comme l'application camera d'un telephone.
Tout tourne dans le processus du test, sur 127.0.0.1, ports 8140-8149.

Le repere de la carte est celui de la camera du dessus (origine au coin en bas a gauche de
l'image, x a droite, y en haut, en metres) ; l'arene fait LARGEUR x HAUTEUR m. La pose vraie
du rover (celle de l'ESP32 simule, dans son propre repere) y est placee par vers_carte : une
rotation de A0 et une translation T0, inconnues de l'adaptateur comme sur le vrai robot.

Ce que le banc garde pour les tests (ce que le vrai monde ne donne pas) :
  - Banc.recus : chaque POST recu par le serveur du rover (instant, chemin, corps, pose
    odometrique au moment de la reception) ;
  - Banc.poses : la pose vraie (x, y, cap) dans le repere de la carte, toutes les 20 ms.
"""

import math
import socket
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
for _dossier in (
    'rover',
    'src/guidance_calib',
    'src/guidance_global_map',
    'src/guidance_rover_pose',
):
    if str(RACINE / _dossier) not in sys.path:
        sys.path.append(str(RACINE / _dossier))

import rover_server  # noqa: E402
from fake_esp32 import FakeEsp32, FakeSerial  # noqa: E402
from rover_core import Config, Rover  # noqa: E402

from guidance_calib.ground_plane import MapBounds  # noqa: E402
from guidance_calib.synthetic import Box, Scene, render_top_down  # noqa: E402
from guidance_calib.synthetic import _paste_marker as coller_marqueur  # noqa: E402

PORTS = range(8140, 8150)
LARGEUR, HAUTEUR = 2.4, 2.0  # m, l'arene vue par la camera
PIXELS_PAR_METRE = 300.0
METRES_PAR_PIXEL = 1.0 / PIXELS_PAR_METRE
MARQUEUR = 10  # id ArUco sur le rover (DICT_4X4_50)
A0 = math.radians(25.0)  # repere de l'ESP32 simule -> repere de la carte
T0 = (0.45, 0.45)
LONGUEUR_ROVER, LARGEUR_ROVER = 0.23, 0.20  # m (rover_size de la camera du dessus)

_pris = set()  # ports deja pris par ce processus (un serveur ferme les rend)
_verrou_ports = threading.Lock()


def vers_carte(bx, by):
    c, s = math.cos(A0), math.sin(A0)
    return T0[0] + c * bx - s * by, T0[1] + s * bx + c * by


def servir(fabrique_handler, port=None):
    """ThreadingHTTPServer sur `port`, ou sur le premier port libre de PORTS."""
    with _verrou_ports:
        for p in [port] if port is not None else PORTS:
            if port is None and p in _pris:
                continue
            try:
                serveur = ThreadingHTTPServer(('127.0.0.1', p), fabrique_handler)
            except OSError:
                continue
            _pris.add(p)
            serveur.daemon_threads = True
            threading.Thread(target=serveur.serve_forever, daemon=True).start()
            return serveur
    return None


def fermer_serveur(serveur):
    serveur.shutdown()
    serveur.server_close()
    with _verrou_ports:
        _pris.discard(serveur.server_address[1])


def boite_touchee(pose, boite, longueur=LONGUEUR_ROVER, largeur=LARGEUR_ROVER):
    """True si le rectangle du rover (centre en pose[:2], long de `longueur` dans la direction
    pose[2]) chevauche la boite (Box, alignee sur les axes). Axes separateurs : ceux de la
    boite et ceux du rover."""
    x, y, cap = pose
    c, s = math.cos(cap), math.sin(cap)
    dl, dw = longueur / 2, largeur / 2
    coins = [(x + c * a - s * b, y + s * a + c * b) for a in (-dl, dl) for b in (-dw, dw)]
    xs, ys = [p[0] for p in coins], [p[1] for p in coins]
    if max(xs) <= boite.x or min(xs) >= boite.x + boite.width:
        return False
    if max(ys) <= boite.y or min(ys) >= boite.y + boite.height:
        return False
    sommets = [
        (boite.x, boite.y),
        (boite.x + boite.width, boite.y),
        (boite.x, boite.y + boite.height),
        (boite.x + boite.width, boite.y + boite.height),
    ]
    for ax, ay, demi in ((c, s, dl), (-s, c, dw)):
        projections = [(px - x) * ax + (py - y) * ay for px, py in sommets]
        if min(projections) >= demi or max(projections) <= -demi:
            return False
    return True


class Banc:
    """rover_server.App + rover_core.Rover + FakeEsp32, servis sur un port de PORTS. Chaque
    POST recu est note dans `recus` ; la pose vraie est notee dans `poses`."""

    def __init__(self, boot_time=0.3):
        self.board = FakeEsp32(seed=1)
        self.cfg = Config(bias_calib_time=0.5)
        self.boot_time = boot_time
        self.rover = self._nouveau_rover()
        camera = rover_server.Camera('none', 640, 480, 15, 70)
        camera.error = 'no camera (bench)'
        camera.request = lambda: None
        self.app = rover_server.App(self.rover, camera, '/dev/null')
        self.recus = []  # (instant, chemin, corps, (x, y) odometrie a la reception)
        self.poses = []  # (instant, x, y, cap) vrais, repere de la carte
        self.camera_ok = True
        self.redemarrages = 0
        self._boot_id = rover_server.BOOT_ID
        self.server = servir(self._handler())
        if self.server is None:
            self.rover.close()
            raise OSError('aucun port libre dans 8140-8149')
        self.port = self.server.server_address[1]
        self.url = f'http://127.0.0.1:{self.port}'
        self._fin = threading.Event()
        threading.Thread(target=self._tracer, daemon=True).start()

    def _nouveau_rover(self):
        rover = Rover(self.cfg, lambda: FakeSerial(self.board, boot_time=self.boot_time))
        rover.start()
        return rover

    def _handler(self):
        banc = self

        class Enregistreur(rover_server.make_handler(self.app)):
            def _body(self):
                corps = super()._body()
                odom = banc.rover.snapshot()['odom']
                banc.recus.append((time.monotonic(), self.path, corps, (odom['x'], odom['y'])))
                return corps

        return Enregistreur

    def _tracer(self):
        while not self._fin.wait(0.02):
            self.poses.append((time.monotonic(), *self.pose_vraie()))

    def pose_vraie(self):
        b = self.board
        x, y = vers_carte(b.x, b.y)
        return x, y, math.atan2(math.sin(b.yaw + A0), math.cos(b.yaw + A0))

    def attendre_pret(self, limite=10.0):
        debut = time.monotonic()
        while time.monotonic() - debut < limite:
            s = self.rover.snapshot()
            if s['link']['connected'] and s['odom']['gyro_ready']:
                return
            time.sleep(0.05)
        raise AssertionError('ESP32 simule pas pret')

    def redemarrer_serveur(self, coupure=0.5):
        """Comme `sudo systemctl restart polaris-rover-api` : le processus s'arrete (le rover
        aussi), le port ne repond plus pendant `coupure` s, puis un nouveau processus (nouveau
        boot_id, odometrie a zero, l'ESP32 redemarre) ecoute sur le meme port. Meme rover
        physique : la pose vraie ne bouge pas."""
        fermer_serveur(self.server)
        self.rover.close()
        time.sleep(coupure)
        self.redemarrages += 1
        rover_server.BOOT_ID = f'banc-{self.redemarrages}'
        self.rover = self._nouveau_rover()
        self.app.rover = self.rover
        self.server = servir(self._handler(), self.port)
        if self.server is None:
            raise OSError(f'port {self.port} pas repris apres le redemarrage')

    def poster(self, chemin, corps=b'{}'):
        """Un autre client (la page du rover, le tableau de bord)."""
        demande = urllib.request.Request(self.url + chemin, data=corps, method='POST')
        urllib.request.urlopen(demande, timeout=2).read()

    def roule(self, depuis):
        """Distance vraie parcourue depuis l'indice `depuis` de board.trail (m)."""
        trace = self.board.trail[depuis:]
        return sum(math.dist(a, b) for a, b in zip(trace, trace[1:], strict=False))

    def fermer(self):
        self._fin.set()
        self.rover.close()
        fermer_serveur(self.server)
        rover_server.BOOT_ID = self._boot_id


class CameraDessus:
    """La camera du dessus en direct : chaque GET rend l'arene vue d'en haut (sol, boites
    `visibles`, marqueur ArUco du rover a sa pose VRAIE) et renvoie une image JPEG, comme
    l'URL "snapshot" d'une application camera IP. camera_reader l'interroge en boucle.

    `pose()` -> (x, y, cap) du rover dans le repere de la carte, ou None (rover hors champ)."""

    ATTENTE = 0.1  # s par image : environ 7 images/s, pour ne pas charger la machine

    def __init__(self, pose, visibles):
        self.pose = pose
        self.scene = Scene(
            bounds=MapBounds(0.0, 0.0, LARGEUR, HAUTEUR),
            marker_positions={},  # des marqueurs aux coins seraient vus comme des obstacles
            margin=0.0,  # repere de la scene = repere de la carte
            boxes=list(visibles),
            rover_marker_id=MARQUEUR,
            pixels_per_metre=PIXELS_PAR_METRE,
        )
        self.fond = render_top_down(self.scene)
        self.images = 0
        camera = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):  # noqa: N802 - http.server API
                time.sleep(camera.ATTENTE)  # l'image est prise apres la demande
                corps = camera.image_jpeg()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(corps)))
                self.end_headers()
                self.wfile.write(corps)

        self.server = servir(Handler)
        if self.server is None:
            raise OSError('aucun port libre dans 8140-8149')
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/shot.jpg'

    def image(self):
        image = self.fond.copy()
        pose = self.pose()
        if pose is not None:
            coller_marqueur(image, self.scene, MARQUEUR, *pose)
        self.images += 1
        return image

    def image_jpeg(self):
        return cv2.imencode('.jpg', self.image(), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    def fermer(self):
        fermer_serveur(self.server)


def percevoir_simule(pose_vraie, boites, portee=0.6, demi_champ=math.pi / 4, res=0.05):
    """Un faux percevoir (interface 4) : ce que verrait la camera du rover, deja dans le
    repere de la carte, (matrice -1/0/1, x0, y0), cases de `res` m, x0 et y0 multiples de res.

    Comme la vraie perception, il voit depuis la pose VRAIE du rover (`pose_vraie()`), mais
    la vue est posee sur la carte a la pose ESTIMEE (`position`, `cap`) : l'erreur de pose de
    l'adaptateur deplace les obstacles vus. Dans le champ (portee, demi_champ) : 1 dans une
    boite, 0 sinon ; ailleurs -1, jamais 0. Sans cap : None."""

    def percevoir(position, cap):
        if cap is None:
            return None
        n = int(math.ceil(portee / res)) + 1
        i0 = math.floor(position[1] / res) - n
        j0 = math.floor(position[0] / res) - n
        jj, ii = np.meshgrid(np.arange(2 * n + 1), np.arange(2 * n + 1))
        cx = (j0 + jj + 0.5) * res - position[0]
        cy = (i0 + ii + 0.5) * res - position[1]
        # le centre de chaque case, dans le repere du rover tel que l'adaptateur le croit
        fx = math.cos(cap) * cx + math.sin(cap) * cy
        fy = -math.sin(cap) * cx + math.cos(cap) * cy
        vu = (np.hypot(fx, fy) <= portee) & (np.abs(np.arctan2(fy, fx)) <= demi_champ)
        # le meme point du repere rover, dans le vrai monde
        vx, vy, vcap = pose_vraie()
        wx = vx + math.cos(vcap) * fx - math.sin(vcap) * fy
        wy = vy + math.sin(vcap) * fx + math.cos(vcap) * fy
        dedans = np.zeros(wx.shape, bool)
        for b in boites:
            dedans |= (wx >= b.x) & (wx <= b.x + b.width) & (wy >= b.y) & (wy <= b.y + b.height)
        matrice = np.full(wx.shape, -1, np.int8)
        matrice[vu] = 0
        matrice[vu & dedans] = 1
        return matrice, j0 * res, i0 * res

    return percevoir


def port_libre():
    """Un port de PORTS ou rien n'ecoute (connexion refusee)."""
    with _verrou_ports:
        for p in reversed(PORTS):
            if p in _pris:
                continue
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1', p)) != 0:
                    return p
    raise OSError('aucun port libre dans 8140-8149')


class TrouNoir:
    """Un rover qui accepte la connexion TCP mais ne repond jamais (Wi-Fi qui avale tout).
    `lignes` : la premiere ligne de chaque requete recue."""

    def __init__(self):
        self.lignes = []
        self._fin = threading.Event()
        self.sock = None
        with _verrou_ports:
            for p in PORTS:
                if p in _pris:
                    continue
                s = socket.socket()
                try:
                    s.bind(('127.0.0.1', p))
                except OSError:
                    s.close()
                    continue
                s.listen(8)
                s.settimeout(0.1)
                _pris.add(p)
                self.sock, self.port = s, p
                break
        if self.sock is None:
            raise OSError('aucun port libre dans 8140-8149')
        self.url = f'http://127.0.0.1:{self.port}'
        self._connexions = []
        threading.Thread(target=self._accepter, daemon=True).start()

    def _accepter(self):
        while not self._fin.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            self._connexions.append(conn)
            threading.Thread(target=self._lire, args=(conn,), daemon=True).start()

    def _lire(self, conn):
        conn.settimeout(0.2)
        donnees = b''
        while not self._fin.is_set() and b'\r\n' not in donnees:
            try:
                bloc = conn.recv(4096)
            except OSError:
                continue
            if not bloc:
                break
            donnees += bloc
        if donnees:
            self.lignes.append(donnees.split(b'\r\n', 1)[0].decode(errors='replace'))

    def fermer(self):
        self._fin.set()
        for conn in self._connexions:
            conn.close()
        self.sock.close()
        with _verrou_ports:
            _pris.discard(self.port)


__all__ = ['Box', 'Banc', 'CameraDessus', 'TrouNoir', 'percevoir_simule', 'boite_touchee']
