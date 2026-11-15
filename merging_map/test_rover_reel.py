"""Tests of the real-rover adapter against a fake of RBDRB's rover server (rover/rover_server.py,
branch rover-teleop), driven through the same requete() function as the HTTP client."""

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cerveau
import numpy as np
import pytest
from fusion_carte import CarteFusion, decouper
from rover_reel import RaspRoverReel, ServeurRoverError, angle_normalise, client_http


class FauxServeur:
    """What rover_server.py does, seen through requete(): a differential rover that follows
    POST /path and POST /move on its OWN odometry, at 20 Hz, with a fake clock.

    Like the real server, a step is first queued, then runs, then the rover stands still for
    0.4 s (settle) before the result goes into ctrl.history; the step is abandoned as
    'stalled' when the wheels do not turn for 1.5 s.

    The odometry starts at (0, 0, 0) where the rover stands, so its frame is shifted and
    rotated from the map. The ground distance is 2 % shorter than the wheels say. Options: a
    wall at x = mur_x where the wheels stall ('bloque') or spin ('patine'), a camera that
    gives a new image only every periode_camera s, a hidden heading, a lost link."""

    TICK, SETTLE, STALL = 0.05, 0.4, 1.5

    def __init__(self, x, y, cap, mur_x=None, mur='bloque', periode_camera=0.0, glissement=0.02):
        self.x, self.y, self.cap = x, y, cap  # truth, map frame
        self.ox = self.oy = self.oh = 0.0  # odometry
        self.mur_x, self.mur, self.glissement = mur_x, mur, glissement
        self.t = 0.0
        self.mode, self.tache, self.queued, self.history = 'idle', None, [], []
        self.joignable, self.connecte, self.gyro_pret_a = True, True, 0.0
        self.coupure = (math.inf, math.inf)  # (t0, t1): no answer in between
        self.drone_ok, self.cap_visible = True, True
        self.periode_camera = periode_camera
        self.passe = [(0.0, x, y, cap)]  # truth over time, for the slow camera
        self.requetes = []
        self.trace = [(x, y)]

    # ---- the HTTP API ------------------------------------------------------------------
    def requete(self, methode, chemin, corps=None):
        self.requetes.append((self.t, methode, chemin, corps))
        if not self.joignable or self.coupure[0] <= self.t < self.coupure[1]:
            return None
        if (methode, chemin) == ('GET', '/state'):
            return json.loads(json.dumps(self.etat()))  # a copy, like an HTTP answer
        if chemin == '/path':
            assert corps['frame'] == 'odom' and 0 < corps['speed'] <= 1
            self._lancer({'kind': 'path', 'but': tuple(corps['points'][-1]), 'v': corps['speed']})
        elif chemin == '/move':
            d = corps['distance']
            but = (self.ox + d * math.cos(self.oh), self.oy + d * math.sin(self.oh))
            self._lancer({'kind': 'move', 'but': but, 'v': 0.15, 'sens': 1 if d >= 0 else -1})
        elif chemin == '/stop':
            if self.mode in ('path', 'move'):
                self._resultat('stopped')
            elif self.mode == 'settle':
                self._resultat(self.statut)
            self.queued, self.mode, self.tache = [], 'idle', None
        elif chemin != '/gimbal':
            raise AssertionError(f'unexpected {methode} {chemin}')
        return {'ok': True}

    def etat(self):
        return {
            'link': {'connected': self.connecte},
            'odom': {
                'x': self.ox,
                'y': self.oy,
                'heading_deg': math.degrees(self.oh),
                'gyro_ready': self.t >= self.gyro_pret_a,
            },
            'ctrl': {
                'mode': self.mode,
                'queued': [dict(q) for q in self.queued],
                'current': None if self.tache is None else {'kind': self.tache['kind']},
                'history': [dict(h) for h in self.history],
            },
        }

    def _lancer(self, tache):
        if self.mode in ('path', 'move'):
            self._resultat('replaced by a new command')
        elif self.mode == 'settle':
            self._resultat(self.statut)
        self.queued, self.mode, self.tache = [tache], 'idle', None

    def _resultat(self, statut):
        entree = {'kind': self.tache['kind'], 'status': statut, 'n': len(self.history)}
        self.history.insert(0, entree)
        self.tache = None

    # ---- the world -----------------------------------------------------------------------
    def dormir(self, dt):
        for _ in range(round(dt / self.TICK)):
            self._tick()
            self.t = round(self.t + self.TICK, 6)
            self.passe.append((self.t, self.x, self.y, self.cap))
            self.trace.append((self.x, self.y))

    def horloge(self):
        return self.t

    def _tick(self):
        if self.mode == 'idle' and self.queued:
            self.tache = self.queued.pop(0)
            self.tache.update(t0=self.t, immobile=0.0)
            self.mode = self.tache['kind']
        if self.mode == 'settle' and self.t >= self.fin_settle:
            self._resultat(self.statut)
            self.mode = 'idle'
        if self.mode in ('path', 'move'):
            self._rouler()

    def _finir(self, statut):
        self.statut, self.mode, self.fin_settle = statut, 'settle', self.t + self.SETTLE

    def _rouler(self):
        k = self.tache
        dx, dy = k['but'][0] - self.ox, k['but'][1] - self.oy
        d = math.hypot(dx, dy)
        if d < 0.02:
            return self._finir('done')
        v, w = 0.0, 0.0
        if k['kind'] == 'move':
            v = k['sens'] * min(k['v'], d / self.TICK)
        else:
            ecart = angle_normalise(math.atan2(dy, dx) - self.oh)
            if abs(ecart) > 0.15:  # pivot on the spot first
                w = math.copysign(1.5, ecart)
            else:
                v, w = min(k['v'], d / self.TICK), 2.0 * ecart
        ds, dh = v * self.TICK, w * self.TICK
        sol = ds * (1 - self.glissement)
        nx = self.x + sol * math.cos(self.cap)
        contre_le_mur = self.mur_x is not None and ds > 0 and nx >= self.mur_x
        if contre_le_mur and self.mur == 'bloque':
            ds = dh = 0.0  # the wheels stop
        elif not contre_le_mur:
            self.x, self.y = nx, self.y + sol * math.sin(self.cap)
        self.cap += dh
        self.oh += dh
        self.ox += ds * math.cos(self.oh)
        self.oy += ds * math.sin(self.oh)
        k['immobile'] = k['immobile'] + self.TICK if (v or w) and not (ds or dh) else 0.0
        if k['immobile'] > self.STALL:
            self._finir('stalled')

    # ---- the overhead camera -------------------------------------------------------------
    def vu_par_le_drone(self):
        """The last image: taken every periode_camera s, so it can show an older position."""
        if not self.drone_ok:
            return None
        t_image = self.t
        if self.periode_camera:
            t_image = math.floor(self.t / self.periode_camera + 1e-9) * self.periode_camera
        _, x, y, cap = max((p for p in self.passe if p[0] <= t_image + 1e-9), key=lambda p: p[0])
        return x, y, (cap if self.cap_visible else None)


class FauxDrone:
    def __init__(self, faux):
        self.faux = faux

    def disponible(self):
        return self.faux.drone_ok

    def zoomer(self, tuile):
        return []


def robot(faux, carte=None, **options):
    return RaspRoverReel(
        carte or CarteFusion(),
        faux.requete,
        faux.vu_par_le_drone,
        lambda position, cap: None,  # no rover camera in these tests
        FauxDrone(faux),
        horloge=faux.horloge,
        dormir=faux.dormir,
        **options,
    )


def chemins(faux, chemin):
    return [c for _, m, ch, c in faux.requetes if ch == chemin and m == 'POST']


def test_angle_normalise() -> None:
    assert math.isclose(angle_normalise(3 * math.pi / 2), -math.pi / 2)


def test_a_step_goes_to_the_server_in_the_odometry_frame() -> None:
    faux = FauxServeur(1.0, 1.0, math.radians(30))
    r = robot(faux)
    r.placer((1.0, 1.0))
    assert r.deplacer((1.3, 1.2)) is None
    assert math.dist((faux.x, faux.y), (1.3, 1.2)) < 0.03
    # the target in the odometry frame: rotated by -30 degrees around the start
    a = math.radians(-30)
    attendu = (0.3 * math.cos(a) - 0.2 * math.sin(a), 0.3 * math.sin(a) + 0.2 * math.cos(a))
    envoye = chemins(faux, '/path')[-1]['points'][0]
    assert math.dist(envoye, attendu) < 0.005
    assert r.position() == (faux.x, faux.y)  # recalibrated on the camera, rover stopped
    assert r.incertitude() == 0


def test_unknown_heading_is_measured_with_a_short_straight_move() -> None:
    """No marker on the rover: the camera gives (x, y, None). The rover first drives 15 cm
    straight; the same motion seen by the camera and by the odometry gives the heading."""
    faux = FauxServeur(1.0, 1.0, math.radians(150))
    faux.cap_visible = False
    r = robot(faux)
    r.placer((1.0, 1.0))
    assert r.drone_disponible() and not r.cap_fiable
    assert r.deplacer((1.4, 1.3)) is None
    assert [ch for _, m, ch, _ in faux.requetes if m == 'POST'] == ['/move', '/path']
    assert r.cap_fiable and abs(angle_normalise(r.cap - faux.cap)) < 0.05
    assert math.dist((faux.x, faux.y), (1.4, 1.3)) < 0.03


def test_odometry_carries_the_position_when_the_camera_is_lost() -> None:
    faux = FauxServeur(1.0, 1.0, math.pi / 2)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.0, 1.3))
    faux.drone_ok = False
    assert not r.drone_disponible()
    r.deplacer((1.3, 1.5))
    r.deplacer((1.3, 1.9))
    assert math.dist(r.position(), (faux.x, faux.y)) < 0.03  # 2 % wheel slip only
    assert r.incertitude() > 0
    faux.drone_ok = True  # the camera sees the rover again
    assert 0 < r.recaler() < 0.03
    assert r.position() == (faux.x, faux.y) and r.incertitude() == 0


def test_heading_is_measured_backwards_when_the_rover_faces_a_wall() -> None:
    """No heading and a wall right in front: the 15 cm straight line stalls, so the rover
    backs up to measure its heading, and reports the wall."""
    faux = FauxServeur(1.0, 1.0, 0.0, mur_x=1.02)
    faux.cap_visible = False
    r = robot(faux)
    r.placer((1.0, 1.0))
    impact = r.deplacer((1.5, 1.0))
    assert [c['distance'] for c in chemins(faux, '/move')] == [0.15, -0.15]
    assert r.cap_fiable and abs(angle_normalise(r.cap - faux.cap)) < 0.05
    assert impact is not None and math.dist(impact, (1.17, 1.0)) < 0.03


def test_stalled_wheels_are_a_bump() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0, mur_x=1.2)
    r = robot(faux)
    r.placer((1.0, 1.0))
    impact = r.deplacer((1.5, 1.0))
    assert faux.history[0]['status'] == 'stalled'
    assert impact is not None and 1.2 <= impact[0] <= 1.45 and abs(impact[1] - 1.0) < 0.05


def test_spinning_wheels_are_a_bump_seen_by_the_camera() -> None:
    """The wheels spin against the obstacle: the server thinks it arrived ('done'), but the
    camera shows the rover did not move."""
    faux = FauxServeur(1.0, 1.0, 0.0, mur_x=1.08, mur='patine')
    r = robot(faux)
    r.placer((1.0, 1.0))
    impact = r.deplacer((1.4, 1.0))
    assert faux.history[0]['status'] == 'done'
    assert impact is not None and 1.08 <= impact[0] <= 1.3
    assert r.position()[0] < 1.09  # and the position comes back to what the camera sees


def test_lost_link_is_not_a_bump() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    faux.joignable = False
    assert r.deplacer((1.4, 1.0)) is None
    assert not r.rover_joignable()


def test_a_few_lost_answers_are_not_a_lost_link() -> None:
    """Wi-Fi drops answers for 1 s in the middle of a step: the server keeps driving, the
    adapter keeps asking, and the brain never hears of it (no no-go zone)."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    faux.coupure = (faux.t + 0.5, faux.t + 1.5)
    assert r.deplacer((1.6, 1.0)) is None
    assert r.rover_joignable()
    assert math.dist((faux.x, faux.y), (1.6, 1.0)) < 0.03


def test_waits_for_the_gyro_before_the_first_move() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.gyro_pret_a = 3.0  # ESP32 boot + gyro bias, the rover must stand still
    r = robot(faux)
    r.placer((1.0, 1.0))
    assert r.deplacer((1.2, 1.0)) is None
    premier = min(t for t, m, _, _ in faux.requetes if m == 'POST')
    assert premier >= 3.0

    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.gyro_pret_a = math.inf
    r = robot(faux, delai_pret=2.0)
    r.placer((1.0, 1.0))
    with pytest.raises(ServeurRoverError, match='gyroscope'):
        r.deplacer((1.2, 1.0))


def test_slow_camera_gives_no_false_bump() -> None:
    """A camera with one image a second, without the rover's heading: the position comes from
    the odometry while the rover moves, and from an image taken after it stopped. (With the
    previous adapter, which trusted every image: 4 false bumps and 35 to 54 cm of error.
    Without waiting for an image taken after the stop: 4 to 7 cm of error at every stop.)"""
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
    for msg in decouper(np.zeros((48, 48)), 0.0, 0.0, 4, carte, 'drone'):
        carte.recevoir(msg)
    faux = FauxServeur(0.4, 0.4, math.radians(-120), periode_camera=1.0)
    faux.cap_visible = False
    lignes, erreurs = [], []
    r = robot(faux, carte, attente_image=1.1)
    deplacer = r.deplacer

    def deplacer_et_mesurer(cible):
        impact = deplacer(cible)
        erreurs.append(math.dist(r.position(), (faux.x, faux.y)))
        return impact

    r.deplacer = deplacer_et_mesurer
    cerveau.mission(
        r,
        carte,
        (0.4, 0.4),
        (2.0, 1.8),
        zone=(0, 0, 2.4, 2.4),
        tolerance=0.15,
        journal=lignes.append,
    )
    texte = '\n'.join(lignes)
    assert 'ARRIVE pres de la victime' in texte
    assert 'CHOC' not in texte
    assert math.dist((faux.x, faux.y), (2.0, 1.8)) <= 0.2
    assert len(erreurs) >= 5 and max(erreurs) < 0.01  # right at every stop


def test_the_brain_runs_with_a_position_without_heading() -> None:
    carte = CarteFusion(n=16, res_fine=0.1, zmax=2, niveau_plan=2, niveau_zoom=1)
    for msg in decouper(np.zeros((16, 16)), 0, 0, 0, carte, 'drone'):
        carte.recevoir(msg)
    faux = FauxServeur(1.0, 1.0, -math.pi / 2)  # faces south, the goal is north
    faux.cap_visible = False
    lignes = []
    cerveau.mission(
        robot(faux, carte),
        carte,
        (1.0, 1.0),
        (3.0, 4.0),
        zone=(0, 0, 6.4, 6.4),
        journal=lignes.append,
    )
    texte = '\n'.join(lignes)
    assert 'ARRIVE pres de la victime' in texte
    assert 'LIAISON DRONE PERDUE' not in texte  # the camera counts as the drone
    assert 'CHOC' not in texte
    assert math.dist((faux.x, faux.y), (3.0, 4.0)) <= 0.6


class _Serveur(BaseHTTPRequestHandler):
    """A tiny stand-in for the old server of main: /state works, /path does not exist."""

    def log_message(self, *args):
        pass

    def _repondre(self, code, corps):
        donnees = json.dumps(corps).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(donnees)))
        self.end_headers()
        self.wfile.write(donnees)

    def do_GET(self):  # noqa: N802 - http.server API
        self._repondre(200, {'odom': {'x': 0.1}})

    def do_POST(self):  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
        if self.path == '/path':
            self._repondre(404, {'error': 'not found'})
        else:
            self._repondre(400, {'error': 'bad request'})


def test_http_client_tells_errors_from_lost_links() -> None:
    serveur = ThreadingHTTPServer(('127.0.0.1', 0), _Serveur)
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    try:
        requete = client_http(f'http://127.0.0.1:{serveur.server_address[1]}/')
        assert requete('GET', '/state') == {'odom': {'x': 0.1}}
        with pytest.raises(ServeurRoverError, match='rover-teleop'):
            requete('POST', '/path', {'points': [[0, 0]]})
        with pytest.raises(ServeurRoverError, match='400'):
            requete('POST', '/move', {'distance': 'loin'})
    finally:
        serveur.shutdown()
        serveur.server_close()
    assert requete('GET', '/state') is None  # nobody answers: the link is lost


def carte_avec_obstacle(x0, y0, x1, y1, **reglages):
    """A map of free floor, 3.2 m wide, with one obstacle rectangle (metres)."""
    carte = CarteFusion(**reglages)
    sol = np.zeros((64, 64), np.int8)
    sol[round(y0 / 0.05) : round(y1 / 0.05), round(x0 / 0.05) : round(x1 / 0.05)] = 1
    for msg in decouper(sol, 0.0, 0.0, 4, carte, 'drone'):
        carte.recevoir(msg)
    return carte


def test_the_heading_step_is_shortened_by_a_known_obstacle() -> None:
    """The heading is not known, so the straight line may go any way: it is limited to what
    the map shows free all around the rover (here an obstacle 25 cm away, radius 12 cm)."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.cap_visible = False
    r = robot(faux, carte_avec_obstacle(1.25, 0.8, 1.6, 1.2, rayon_robot=0.12))
    r.placer((1.0, 1.0))
    r.deplacer((0.6, 1.0))
    assert [c['distance'] for c in chemins(faux, '/move')] == [0.13]
    assert r.cap_fiable


def test_the_heading_step_is_full_when_the_floor_is_free_around() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.cap_visible = False
    r = robot(faux, carte_avec_obstacle(1.6, 0.8, 1.8, 1.2))
    r.placer((1.0, 1.0))
    r.deplacer((0.6, 1.0))
    assert [c['distance'] for c in chemins(faux, '/move')] == [0.15]


def test_the_heading_step_is_the_smallest_measurable_one_next_to_an_obstacle() -> None:
    """Too close to an obstacle to have 10 cm free all around: the rover makes the shortest
    step that still gives a heading (12 cm), and bump detection is the only protection."""
    faux = FauxServeur(1.0, 1.0, math.pi)  # facing away from the obstacle
    faux.cap_visible = False
    r = robot(faux, carte_avec_obstacle(1.2, 0.8, 1.6, 1.2))
    r.placer((1.0, 1.0))
    r.deplacer((0.6, 1.0))
    assert [c['distance'] for c in chemins(faux, '/move')] == [0.12]
    assert r.cap_fiable
