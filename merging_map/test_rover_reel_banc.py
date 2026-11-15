"""The real-rover adapter against the REAL rover server code (rover/rover_server.py and
rover_core.py) with the simulated ESP32 (rover/fake_esp32.py), in process, on 127.0.0.1
ports 8100-8109. Real time: a few seconds per scenario. Adapted from the audit's bench
(work/mm-real/banc.py). The overhead camera is simulated from the fake ESP32's true pose,
in a map frame rotated by 30 degrees and shifted."""

import math
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE / 'rover') not in sys.path:
    sys.path.append(str(RACINE / 'rover'))

rover_server = pytest.importorskip('rover_server')
from fake_esp32 import FakeEsp32, FakeSerial  # noqa: E402
from fusion_carte import CarteFusion  # noqa: E402
from rover_core import Config, Rover  # noqa: E402
from rover_reel import ArretOperateur, RaspRoverReel, ServeurRoverError, client_http  # noqa: E402

A0 = math.radians(30.0)
T0 = (1.0, 0.8)
PORTS = range(8100, 8110)


def vers_carte(bx, by):
    c, s = math.cos(A0), math.sin(A0)
    return T0[0] + c * bx - s * by, T0[1] + s * bx + c * by


class Banc:
    """rover_server.App + rover_core.Rover + FakeEsp32, served on a reserved port."""

    def __init__(self, boot_time=0.3, cap_visible=True, ports=PORTS):
        self.board = FakeEsp32(seed=1)
        self.cfg = Config(bias_calib_time=0.5)
        self.boot_time = boot_time
        self.rover = self._nouveau_rover()
        camera = rover_server.Camera('none', 640, 480, 15, 70)
        camera.error = 'no camera (bench)'
        camera.request = lambda: None
        self.app = rover_server.App(self.cfg, '/dev/null', None, None)
        self.app.rover, self.app.camera, self.app.mode = self.rover, camera, 'sim'
        self.server = None
        for port in ports:
            try:
                self.server = ThreadingHTTPServer(
                    ('127.0.0.1', port), rover_server.make_handler(self.app)
                )
                break
            except OSError:
                continue
        if self.server is None:
            self.rover.close()
            pytest.skip(f'no free port in {ports[0]}-{ports[-1]}')
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.camera_ok, self.cap_visible = True, cap_visible

    def _nouveau_rover(self):
        rover = Rover(self.cfg, lambda: FakeSerial(self.board, boot_time=self.boot_time))
        rover.start()
        return rover

    def vue(self):
        if not self.camera_ok:
            return None
        b = self.board
        x, y = vers_carte(b.x, b.y)
        cap = math.atan2(math.sin(b.yaw + A0), math.cos(b.yaw + A0)) if self.cap_visible else None
        return x, y, cap

    def verite(self):
        return vers_carte(self.board.x, self.board.y)

    def redemarrer_serveur(self):
        """Like a restart of rover_server.py: new Rover (odometry at zero, the ESP32 boots
        again), same physical rover."""
        self.rover.close()
        self.rover = self._nouveau_rover()
        self.app.rover = self.rover

    def attendre_pret(self, limite=10.0):
        debut = time.monotonic()
        while time.monotonic() - debut < limite:
            s = self.rover.snapshot()
            if s['link']['connected'] and s['odom']['gyro_ready']:
                return
            time.sleep(0.05)
        raise AssertionError('fake ESP32 not ready')

    def poster(self, chemin, corps=b'{}'):
        """Another client (the rover's web page, the dashboard)."""
        demande = urllib.request.Request(self.url + chemin, data=corps, method='POST')
        urllib.request.urlopen(demande, timeout=2).read()

    def fermer(self):
        self.rover.close()
        self.server.shutdown()
        self.server.server_close()


class Drone:
    def __init__(self, banc):
        self.banc = banc

    def disponible(self):
        return self.banc.camera_ok

    def zoomer(self, tuile):
        return []


@pytest.fixture
def banc():
    b = Banc()
    b.attendre_pret()
    yield b
    b.fermer()


def adaptateur(banc, **options):
    base = client_http(banc.url, delai=2.0)
    envoyes = []

    def requete(methode, chemin, corps=None):
        if methode == 'POST':
            envoyes.append((time.monotonic(), chemin, corps))
        return base(methode, chemin, corps)

    r = RaspRoverReel(
        CarteFusion(niveau_plan=4, rayon_robot=0.155),
        requete,
        banc.vue,
        lambda position, cap: None,
        Drone(banc),
        rafraichir_drone=lambda: None,
        periode=0.1,
        **options,
    )
    r.envoyes = envoyes
    return r


def roule(board, depuis):
    trace = board.trail[depuis:]
    return sum(math.dist(a, b) for a, b in zip(trace, trace[1:], strict=False))


def test_bench_nominal_step(banc) -> None:
    r = adaptateur(banc)
    x0, y0 = banc.vue()[:2]
    r.placer((x0, y0))
    assert r.deplacer((x0 + 0.3, y0)) is None
    assert math.dist(banc.verite(), (x0 + 0.3, y0)) < 0.05


def test_bench_operator_stop_is_final(banc) -> None:
    r = adaptateur(banc)
    x0, y0 = banc.vue()[:2]
    r.placer((x0, y0))
    r.deplacer((x0 + 0.3, y0))
    t_stop = []

    def operateur():
        time.sleep(0.7)
        banc.poster('/stop')  # STOP button of the rover's page
        t_stop.append(time.monotonic())

    threading.Thread(target=operateur, daemon=True).start()
    with pytest.raises(ArretOperateur, match='stopped'):
        r.deplacer((x0 + 0.6, y0))
    assert t_stop and not [c for t, c, _ in r.envoyes if t > t_stop[0]]
    n = len(banc.board.trail)
    time.sleep(0.5)
    assert roule(banc.board, n) < 0.03  # it stays stopped (coasting only)


def test_bench_manual_takeover_is_final(banc) -> None:
    r = adaptateur(banc)
    x0, y0 = banc.vue()[:2]
    r.placer((x0, y0))
    r.deplacer((x0 + 0.3, y0))

    def operateur():
        time.sleep(0.7)
        banc.poster('/cmd', b'{"v": 0.0, "omega": 0.0}')
        banc.poster('/cmd', b'{"v": -0.05, "omega": 0.0}')

    threading.Thread(target=operateur, daemon=True).start()
    with pytest.raises(ArretOperateur):
        r.deplacer((x0 + 0.6, y0))


@pytest.mark.parametrize('cap_visible', [True, False])
def test_bench_restart_between_steps(banc, cap_visible) -> None:
    """Without the fix: 1.4 m driven for a 30 cm step (audit S2)."""
    banc.cap_visible = cap_visible
    r = adaptateur(banc)
    x0, y0 = banc.vue()[:2]
    r.placer((x0, y0))
    for k in range(1, 3):
        r.deplacer((x0 + 0.3 * k, y0))
    banc.redemarrer_serveur()
    time.sleep(0.2)  # the next step comes while the ESP32 boots again
    n = len(banc.board.trail)
    cible = (x0 + 0.9, y0)
    assert r.deplacer(cible) is None  # no false bump while the ESP32 boots
    assert r.redemarrages == 1
    assert math.dist(banc.verite(), cible) < 0.06
    assert roule(banc.board, n) < (0.4 if cap_visible else 0.75)


def test_bench_stale_frame_step_is_never_sent(banc) -> None:
    r = adaptateur(banc)
    x0, y0 = banc.vue()[:2]
    r.placer((x0, y0))
    r.deplacer((x0 + 0.3, y0))
    r.repere.ty += 1.0  # an odometry jump nothing detected
    n = len(r.envoyes)
    with pytest.raises(ServeurRoverError, match='Pas refuse'):
        r.deplacer((x0 + 0.6, y0))
    assert len(r.envoyes) == n


def test_bench_esp32_silent_during_a_step_is_a_server_error(banc) -> None:
    """The fake ESP32 stops sending its feedback in the middle of a step: the real server
    stops the rover ('no ESP32 feedback'); the adapter must not take it for a finished step."""
    r = adaptateur(banc)
    x0, y0 = banc.vue()[:2]
    r.placer((x0, y0))
    r.deplacer((x0 + 0.3, y0))
    coupe = []

    def panne():
        time.sleep(0.7)
        banc.board.feedback_on = False
        coupe.append(time.monotonic())

    threading.Thread(target=panne, daemon=True).start()
    with pytest.raises(ServeurRoverError, match='no ESP32 feedback'):
        r.deplacer((x0 + 0.6, y0))
    assert coupe and not [c for t, c, _ in r.envoyes if t > coupe[0] and c == '/path']
    assert r._a_recaler  # the next step re-anchors on a fresh image first
    assert banc.rover.snapshot()['ctrl']['fault'] == 'no ESP32 feedback'


def test_bench_stop_between_two_steps_is_final(banc) -> None:
    """STOP pressed on the rover's page while it waits between two steps: no trace in
    ctrl.history, only in stop_count. The next step must not be sent."""
    r = adaptateur(banc)
    x0, y0 = banc.vue()[:2]
    r.placer((x0, y0))
    r.deplacer((x0 + 0.3, y0))
    r.attendre(0.1)  # a stop of our own (the brain does this after an error) is not one
    banc.poster('/stop')  # another client
    t_stop = time.monotonic()
    with pytest.raises(ArretOperateur, match='entre deux pas'):
        r.deplacer((x0 + 0.6, y0))
    assert not [c for t, c, _ in r.envoyes if t > t_stop and c == '/path']


class MurDevant(FakeEsp32):
    """Wheels stuck while driving forward (an unseen wall right in front), free backwards."""

    @property
    def blocked(self):
        tl, tr = self.target
        return tl + tr > 1e-3

    @blocked.setter
    def blocked(self, valeur):
        pass


def test_bench_heading_is_measured_backwards_against_a_wall() -> None:
    """Rover without marker facing an unseen wall: the forward heading step stalls, so it
    backs off to measure its heading. A /path to a point 17 cm behind ended 'done' at once on
    the real server (path_lookahead 0.2 m): no heading, 'Cap impossible a mesurer'."""
    banc = Banc(cap_visible=False, ports=range(8260, 8265))
    banc.board.__class__ = MurDevant
    try:
        banc.attendre_pret()
        r = adaptateur(banc)
        x0, y0 = banc.vue()[:2]
        r.placer((x0, y0))
        impact = r.deplacer((x0 + 0.3, y0))
        assert [c for _, c, _ in r.envoyes][:2] == ['/path', '/move']
        assert r.cap_fiable and abs(r.repere.angle - A0) < 0.1
        assert impact is not None
        assert banc.board.x < -0.12  # it really backed off (about 0.17 m)
    finally:
        banc.fermer()


def test_bench_odometry_reset_during_a_step_stops_the_rover() -> None:
    """POST /reset_odom (button of the rover's page) 0.5 s into a 0.20 m step: the controller
    goes on towards a point of the old odometry frame. Without /stop: 0.66 m driven."""
    banc = Banc(ports=range(8260, 8265))
    try:
        banc.attendre_pret()
        r = adaptateur(banc)
        x0, y0 = banc.vue()[:2]
        r.placer((x0, y0))
        r.deplacer((x0 + 0.3, y0))
        n = len(banc.board.trail)
        threading.Timer(0.5, lambda: banc.poster('/reset_odom')).start()
        assert r.deplacer((x0 + 0.5, y0)) is None
        assert banc.rover.snapshot()['ctrl']['mode'] == 'idle'
        time.sleep(1.0)
        assert roule(banc.board, n) < 0.25
    finally:
        banc.fermer()
