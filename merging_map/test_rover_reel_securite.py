"""Safety behaviour of the real-rover adapter, against the fake server of test_rover_reel.py:
an operator stop is final, a server restart is detected before any motion, a step is bounded,
the heading move survives a lost answer, a cut HTTP answer is a lost answer."""

import math
import socket
import threading

import cerveau
import numpy as np
import pytest
from fusion_carte import CarteFusion, decouper
from mission_settings import ARENA
from rover_reel import (
    ArretOperateur,
    RaspRoverReel,
    ServeurRoverError,
    angle_normalise,
    client_http,
    serveur_redemarre,
)
from test_rover_reel import FauxServeur, chemins, robot

PORTS = range(8100, 8110)  # reserved for these tests, 127.0.0.1 only


def lier(serveur_socket):
    for port in PORTS:
        try:
            serveur_socket.bind(('127.0.0.1', port))
            return port
        except OSError:
            continue
    pytest.skip('no free port in 8100-8109')


def pendant_le_pas(faux, apres, action):
    """Runs `action` once, `apres` s after the first POST /path of the test."""
    dormir = faux.dormir
    fait = []

    def dormir_et_agir(dt):
        dormir(dt)
        envois = [t for t, m, ch, _ in faux.requetes if m == 'POST' and ch == '/path']
        if envois and not fait and faux.t >= envois[0] + apres:
            fait.append(faux.t)
            action()

    faux.dormir = dormir_et_agir
    return fait


def posts_apres(faux, t):
    return [ch for tt, m, ch, _ in faux.requetes if m == 'POST' and tt > t]


# ---- operator stop (interface 5) ------------------------------------------------------------


def test_operator_stop_during_a_step_is_final() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    fait = pendant_le_pas(faux, 0.8, faux.stop_operateur)
    r = robot(faux)
    r.placer((1.0, 1.0))
    with pytest.raises(ArretOperateur, match='stopped'):
        r.deplacer((1.3, 1.0))
    assert fait and posts_apres(faux, fait[0]) == []  # nothing sent after the stop


def test_manual_takeover_and_another_command_are_operator_stops() -> None:
    def manuel(faux):
        faux._resultat('cancelled by manual control')
        faux.queued, faux.mode = [], 'idle'

    def autre_commande(faux):  # another client sends its own step: ours is replaced
        faux._reponse('POST', '/move', {'distance': 0.05})

    for action, motif in ((manuel, 'manual'), (autre_commande, 'replaced')):
        faux = FauxServeur(1.0, 1.0, 0.0)
        pendant_le_pas(faux, 0.8, lambda faux=faux, action=action: action(faux))
        r = robot(faux)
        r.placer((1.0, 1.0))
        with pytest.raises(ArretOperateur, match=motif):
            r.deplacer((1.3, 1.0))


def test_manual_driving_between_steps_stops_the_mission() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    faux.mode = 'manual'  # a teammate drives it with the keyboard
    n = len(chemins(faux, '/path'))
    with pytest.raises(ArretOperateur, match='main'):
        r.deplacer((1.4, 1.0))
    assert len(chemins(faux, '/path')) == n


def test_a_step_dropped_from_the_queue_is_an_operator_stop() -> None:
    """/stop while our step is still queued: it vanishes without a history entry."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    requete = faux.requete
    vu = []

    def requete_et_stop(methode, chemin, corps=None):
        reponse = requete(methode, chemin, corps)
        if chemin == '/path':
            faux.stop_operateur()  # before the server tick takes the step from the queue
            vu.append('a envoyer')
        elif vu == ['a envoyer'] and methode == 'GET':
            reponse['ctrl']['queued'] = [{'kind': 'path'}]  # seen queued once, then gone
            vu.append('vu')
        return reponse

    r.requete = requete_et_stop
    with pytest.raises(ArretOperateur, match='/stop'):
        r.deplacer((1.4, 1.0))


def test_our_own_resend_is_not_an_operator_stop() -> None:
    """The answer to POST /path is lost but the server got it: the adapter sends it again and
    the server replaces the step by the same one. That 'replaced' status is ours."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    faux.perdre_reponse = 1
    assert r.deplacer((1.5, 1.0)) is None
    statuts = [h['status'] for h in faux.history]
    assert 'replaced by a new command' in statuts
    assert math.dist((faux.x, faux.y), (1.5, 1.0)) < 0.03


def test_operator_stop_ends_the_brains_mission() -> None:
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
    for msg in decouper(np.zeros((48, 48)), 0.0, 0.0, 4, carte, 'drone'):
        carte.recevoir(msg)
    faux = FauxServeur(0.4, 0.4, 0.0)
    fait = pendant_le_pas(faux, 0.8, faux.stop_operateur)
    with pytest.raises(ArretOperateur):
        cerveau.mission(
            robot(faux, carte),
            carte,
            (0.4, 0.4),
            (2.0, 1.8),
            zone=(0, 0, 2.4, 2.4),
            tolerance=0.15,
            journal=lambda _: None,
        )
    assert fait and posts_apres(faux, fait[0]) == []


# ---- server restart (interface 3) and bounded steps -----------------------------------------


def test_restart_is_detected_by_boot_id_or_counters() -> None:
    def etat(boot=None, odometre=1.0, rx=100):
        e = {'odom': {'odometer': odometre}, 'link': {'rx_count': rx}}
        if boot is not None:
            e['boot_id'] = boot
        return e

    assert not serveur_redemarre(etat('a'), etat('a', odometre=2.0))
    assert serveur_redemarre(etat('a'), etat('a', odometre=0.1))  # odometry reset, same process
    assert serveur_redemarre(etat('a'), etat('b', odometre=2.0))
    assert serveur_redemarre(etat(), etat('b'))  # newer server started in between
    assert serveur_redemarre(etat(), etat(odometre=0.1))  # older server: odometer went down
    assert serveur_redemarre(etat(), etat(odometre=1.2, rx=3))
    assert not serveur_redemarre(etat(), etat(odometre=1.2, rx=150))
    assert not serveur_redemarre({'odom': {}}, {'odom': {}})


@pytest.mark.parametrize('avec_boot_id', [False, True])
@pytest.mark.parametrize('cap_visible', [True, False])
def test_restart_between_steps_waits_and_reanchors(avec_boot_id, cap_visible) -> None:
    """The server restarts between two steps (seen twice on the robot): the odometry is back
    at zero. Without detection the next 30 cm step drove 1.4 m. Now: wait for the ESP32 and
    the gyro, re-anchor on a fresh image (and measure the heading again if the camera does
    not give it), then drive the step that was asked."""
    faux = FauxServeur(1.0, 1.0, math.radians(30))
    faux.cap_visible = cap_visible
    if avec_boot_id:
        faux.boot_id = 'a'
    lignes = []
    r = robot(faux, signaler=lignes.append)
    r.placer((1.0, 1.0))
    for k in range(1, 4):
        r.deplacer((1.0 + 0.3 * k, 1.0))
    faux.redemarrer(boot=3.0)
    t_redemarrage = faux.t
    assert r.deplacer((2.2, 1.1)) is None
    assert r.redemarrages == 1 and any('redemarre' in ligne for ligne in lignes)
    premier = min(t for t, m, _, _ in faux.requetes if m == 'POST' and t > t_redemarrage)
    assert premier >= faux.gyro_pret_a  # no motion before the gyro is ready
    assert math.dist((faux.x, faux.y), (2.2, 1.1)) < 0.03
    longueur = sum(
        math.dist(a, b) for a, b in zip(faux.trace[-400:], faux.trace[-399:], strict=False)
    )
    assert longueur < 0.6  # 0.31 m asked (+ 2 x 15 cm to measure the heading without it)


def test_restart_during_a_step_is_not_a_bump() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    pendant_le_pas(faux, 0.6, lambda: faux.redemarrer(boot=2.0))
    r = robot(faux)
    r.placer((1.0, 1.0))
    assert r.deplacer((1.3, 1.0)) is None  # history emptied: no 25 s wait, no false bump
    assert faux.t < 10.0
    assert r.deplacer((1.3, 1.0)) is None
    assert math.dist((faux.x, faux.y), (1.3, 1.0)) < 0.03


def test_a_step_farther_than_asked_is_never_sent() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.3, 1.0))
    r.repere.tx -= 1.0  # a stale map <- odometry frame that nothing detected
    n = len(chemins(faux, '/path'))
    with pytest.raises(ServeurRoverError, match='Pas refuse'):
        r.deplacer((1.6, 1.0))
    assert len(chemins(faux, '/path')) == n


# ---- idempotent heading move, HTTP client ---------------------------------------------------


def test_heading_move_is_idempotent_when_its_answer_is_lost() -> None:
    """With /move, a lost answer drove 0.27 m instead of 0.15 m. With /path to an absolute
    odometry point, the resend goes to the same point."""
    faux = FauxServeur(1.0, 1.0, math.radians(150))
    faux.cap_visible = False
    faux.perdre_reponse = 1
    r = robot(faux)
    r.placer((1.0, 1.0))
    assert r.deplacer((0.8, 1.3)) is None
    envoyes = chemins(faux, '/path')
    assert envoyes[0] == envoyes[1]  # the same absolute point, sent twice
    assert 'replaced by a new command' in [h['status'] for h in faux.history]
    assert math.dist((faux.x, faux.y), (0.8, 1.3)) < 0.03


def test_http_client_cut_answer_is_a_lost_answer() -> None:
    """A connection cut in the middle of the body raised IncompleteRead (not an OSError) and
    killed the mission without /stop. It is a lost answer, like a timeout."""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = lier(s)
    s.listen(1)

    def servir():
        conn, _ = s.accept()
        conn.recv(4096)
        conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 2000\r\n\r\n{"odom": {"x": 0')
        conn.close()

    threading.Thread(target=servir, daemon=True).start()
    try:
        assert client_http(f'http://127.0.0.1:{port}', delai=2.0)('GET', '/state') is None
    finally:
        s.close()


# ---- server faults: the server stopped the rover by itself ----------------------------------


@pytest.mark.parametrize(
    'statut', ['no ESP32 feedback', 'ESP32 link lost', 'controller error: ZeroDivisionError']
)
def test_a_server_fault_during_a_step_is_an_error_not_a_finished_step(statut) -> None:
    """rover_core ends a step with these statuses when it stops the rover by itself. The
    adapter took them for a finished step and the mission sent the next one."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    pendant_le_pas(faux, 0.6, lambda: faux.panne(statut))
    r = robot(faux)
    r.placer((1.0, 1.0))
    with pytest.raises(ServeurRoverError, match=statut.split(':')[0]) as e:
        r.deplacer((1.6, 1.0))
    assert faux.history[0]['status'] == statut  # it ended in the server's history
    assert repr(statut) in str(e.value)  # ctrl.fault is named too
    assert r._a_recaler  # the next step re-anchors on a fresh image first
    faux.dormir = FauxServeur.dormir.__get__(faux)  # the fault is over
    assert r.deplacer((1.6, 1.0)) is None
    assert math.dist((faux.x, faux.y), (1.6, 1.0)) < 0.03


def test_a_server_fault_during_the_heading_step_is_an_error() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.cap_visible = False
    pendant_le_pas(faux, 0.3, faux.panne)
    r = robot(faux)
    r.placer((1.0, 1.0))
    with pytest.raises(ServeurRoverError, match='no ESP32 feedback'):
        r.deplacer((1.3, 1.0))
    assert r._a_recaler and len(chemins(faux, '/path')) == 1


def test_a_step_dropped_by_a_fault_before_it_starts_is_an_error() -> None:
    """The ESP32 is already silent: the server drops the queued step at its next tick, with
    no history entry, only ctrl.fault. Not an operator stop, and no 25 s wait."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    requete = faux.requete

    def requete_et_panne(methode, chemin, corps=None):
        reponse = requete(methode, chemin, corps)
        if chemin == '/path':
            faux.panne()
        return reponse

    r.requete = requete_et_panne
    t0 = faux.t
    with pytest.raises(ServeurRoverError, match='no ESP32 feedback'):
        r.deplacer((1.4, 1.0))
    assert faux.t - t0 < 3.0


def _carte_libre():
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
    for msg in decouper(np.zeros((48, 48)), 0.0, 0.0, 4, carte, 'drone'):
        carte.recevoir(msg)
    return carte


def _mission(faux, carte, lignes):
    return cerveau.mission(
        robot(faux, carte),
        carte,
        (0.4, 0.4),
        (2.0, 1.8),
        zone=(0, 0, 2.4, 2.4),
        tolerance=0.15,
        journal=lignes.append,
        settings=ARENA,
    )


def test_the_brain_stops_and_retries_after_one_server_fault() -> None:
    carte = _carte_libre()
    faux = FauxServeur(0.4, 0.4, 0.0)
    fait = pendant_le_pas(faux, 0.6, faux.panne)
    lignes = []
    res = _mission(faux, carte, lignes)
    assert fait and res['issue'] == 'arrivee'
    assert sum('ERREUR du rover' in ligne for ligne in lignes) == 1
    posts = [ch for _, m, ch, _ in faux.requetes if m == 'POST']
    assert posts[:3] == ['/path', '/stop', '/path']  # stopped before trying again


def test_the_brain_ends_with_robot_error_when_the_fault_stays() -> None:
    carte = _carte_libre()
    faux = FauxServeur(0.4, 0.4, 0.0)
    requete = faux.requete

    def requete_et_panne(methode, chemin, corps=None):
        reponse = requete(methode, chemin, corps)
        if chemin == '/path':
            faux.panne()  # the ESP32 never answers again
        return reponse

    faux.requete = requete_et_panne
    lignes = []
    res = _mission(faux, carte, lignes)
    assert res['issue'] == 'erreur_robot' and cerveau.ETAT_SUIVI[res['issue']] == 'erreur'
    assert 'no ESP32 feedback' in res['raison']
    assert len(chemins(faux, '/path')) == ARENA.robot_error_retries + 1


# ---- STOP pressed while the rover waits between two steps -----------------------------------


def test_stop_between_two_steps_is_an_operator_stop() -> None:
    """rover_core.stop() on an idle rover leaves no trace in ctrl.history: only the server's
    stop_count shows it."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    faux.stop_operateur()
    n = len(chemins(faux, '/path'))
    with pytest.raises(ArretOperateur, match='entre deux pas'):
        r.deplacer((1.4, 1.0))
    assert len(chemins(faux, '/path')) == n


def test_our_own_stops_are_not_operator_stops() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    r.attendre(1.0)
    r.arreter()
    assert r.deplacer((1.4, 1.0)) is None
    assert faux.stops == 2
    faux.stop_operateur()  # ours are counted, an operator's one after them is still seen
    with pytest.raises(ArretOperateur):
        r.deplacer((1.6, 1.0))


def test_a_received_stop_whose_answer_is_lost_counts_as_an_operator_stop() -> None:
    """Conservative on purpose: a false operator stop ends the mission, a missed one would
    drive the rover after a STOP."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    faux.perdre_reponse = 1
    r.arreter()
    with pytest.raises(ArretOperateur):
        r.deplacer((1.4, 1.0))


def test_an_older_server_without_stop_count_is_not_checked() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.stops = None
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    faux.stop_operateur()  # not seen, as before: Ctrl-C stays the safe stop
    assert r.deplacer((1.4, 1.0)) is None


def test_reset_odom_is_not_an_operator_stop() -> None:
    """/reset_odom (same process, same boot_id) looks like a restart to the adapter, but the
    server's stop_count goes on: the stops pressed before the mission are not new."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.boot_id, faux.stops = 'a', 3  # STOP pressed during the setup
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    faux.ox = faux.oy = faux.oh = faux.odometre = 0.0  # /reset_odom between two steps
    assert r.deplacer((1.4, 1.0)) is None
    assert r.redemarrages == 1 and math.dist((faux.x, faux.y), (1.4, 1.0)) < 0.03


def test_stop_after_a_server_restart_is_seen() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.boot_id = 'a'
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    r.arreter()
    faux.redemarrer(boot=1.0)  # new process: stop_count starts again from zero
    faux.stop_operateur()
    with pytest.raises(ArretOperateur, match='entre deux pas'):
        r.deplacer((1.4, 1.0))


# ---- review round of 15 Nov: restart during a step, guard, faults, turret, keyboard ---------


def test_odometry_reset_during_a_step_stops_the_rover() -> None:
    """/reset_odom in the middle of a step: same process, so the controller still follows the
    step towards a point of the OLD odometry frame. The adapter returned without /stop and the
    rover drove 0.52 m for a 0.20 m step."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.3, 1.0))  # the odometry is now away from its origin
    faux.requetes.clear()

    def reset_odom():
        faux.ox = faux.oy = faux.oh = faux.odometre = 0.0

    fait = pendant_le_pas(faux, 0.5, reset_odom)
    r.dormir = faux.dormir
    depart = (faux.x, faux.y)
    assert r.deplacer((1.5, 1.0)) is None
    assert fait and posts_apres(faux, fait[0] - 1e-9) == ['/stop']
    assert faux.mode == 'idle' and not faux.queued
    faux.dormir(3.0)
    assert math.dist((faux.x, faux.y), depart) < 0.2  # it did not go on driving


def test_a_rover_still_driving_is_not_ready() -> None:
    """Before re-anchoring on a picture the rover must be idle with nothing queued and no
    tilt-guard back-off (optional fields)."""
    etat = {'link': {'connected': True}, 'odom': {'gyro_ready': True}}
    assert RaspRoverReel._est_pret(etat)  # an older server: nothing more to check
    for ctrl in (
        {'mode': 'path'},
        {'mode': 'idle', 'queued': [{'kind': 'path'}]},
        {'mode': 'idle', 'current': {'kind': 'move'}},
        {'mode': 'idle', 'guard': {'state': 'backoff'}},
        {'mode': 'idle', 'guard': {'state': 'hold'}},
    ):
        assert not RaspRoverReel._est_pret({**etat, 'ctrl': ctrl}), ctrl
    assert RaspRoverReel._est_pret({**etat, 'ctrl': {'mode': 'idle', 'guard': {'state': 'ok'}}})


def test_tilt_guard_back_off_is_still_a_motion() -> None:
    """The tilt guard ends the step ('tilt: ...') then backs the rover off in mode 'idle'
    (ctrl.guard.state 'backoff', then 'hold' 1 s): the step is over only after that. It is a
    bump, past the back-off: otherwise the next step drives onto the same slope."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.garde = 'ok'
    debut = []

    def bascule():
        faux._resultat('tilt: roll 40 deg')
        faux.queued, faux.mode, faux.fault, faux.garde = [], 'idle', 'tilt: roll', 'backoff'
        debut.append(faux.t)

    pendant_le_pas(faux, 0.6, bascule)
    dormir = faux.dormir

    def dormir_et_garde(dt):
        dormir(dt)
        if debut and faux.garde == 'backoff' and faux.t >= debut[0] + 2.0:
            faux.garde = 'hold'
        elif debut and faux.garde == 'hold' and faux.t >= debut[0] + 3.0:
            faux.garde = 'ok'

    faux.dormir = dormir_et_garde
    r = robot(faux)
    r.placer((1.0, 1.0))
    impact = r.deplacer((1.3, 1.0))
    assert faux.t >= debut[0] + 3.0  # the picture was taken after the back-off and the hold
    assert impact is not None and math.isclose(impact[0], r.position()[0] + 0.30, abs_tol=0.02)


def test_a_step_that_times_out_is_a_bump() -> None:
    """'timeout': the server gave up a step that never ended (wheels spinning against a soft
    obstacle): a bump, not a step to send again."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    pendant_le_pas(faux, 0.6, lambda: faux._finir('timeout'))
    r = robot(faux)
    r.placer((1.0, 1.0))
    impact = r.deplacer((1.3, 1.0))
    assert impact is not None and math.isclose(impact[0], r.position()[0] + 0.15, abs_tol=0.02)


def test_a_step_done_while_a_fault_is_set_is_a_fault() -> None:
    """A fault during the settle (0.4 s): history says 'done' but ctrl.fault is set."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    finir = faux._finir

    def finir_et_panne(statut):
        finir(statut)
        faux.fault = 'ESP32 link lost'

    faux._finir = finir_et_panne
    r = robot(faux)
    r.placer((1.0, 1.0))
    with pytest.raises(ServeurRoverError, match='ESP32 link lost'):
        r.deplacer((1.3, 1.0))
    assert r._a_recaler


def _vues(faux, **options):
    vues = []
    r = robot(faux, **options)
    r.percevoir = lambda position, cap: vues.append((faux.t, cap))
    return r, vues


def test_the_camera_heading_uses_the_turret_pan_of_the_state() -> None:
    """The turret was turned 30 deg to the right with the keyboard: the camera looks 30 deg
    right of the rover (cap - pan). Without gimbal in /state: pan 0, as before."""
    faux = FauxServeur(1.0, 1.0, math.radians(20))
    faux.gimbal = {'pan': 30.0, 'tilt': -20.0, 'moving': False}
    r, vues = _vues(faux)
    r.placer((1.0, 1.0))
    r.observer()  # before the first step: /state is read for the pan
    r.deplacer((1.3, 1.1))
    r.observer()
    assert vues[1][1] == pytest.approx(angle_normalise(r.cap - math.radians(30)))
    faux = FauxServeur(1.0, 1.0, math.radians(20))
    r, vues = _vues(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.3, 1.1))
    r.observer()
    assert vues[-1][1] == pytest.approx(r.cap)


def test_full_turn_waits_for_the_turret_and_the_video_and_skips_a_lost_gimbal_order() -> None:
    faux = FauxServeur(1.0, 1.0, 0.0)
    faux.gimbal = {'pan': 0.0, 'tilt': -20.0, 'moving': False}
    r, vues = _vues(faux, latence_video=0.7)
    r.placer((1.0, 1.0))
    r.deplacer((1.3, 1.0))
    requete = faux.requete
    perdus = []

    def requete_gimbal(methode, chemin, corps=None):
        if chemin == '/gimbal' and corps['pan'] == -90 and not perdus:
            perdus.append(faux.t)
            return None  # this order never reached the rover
        return requete(methode, chemin, corps)

    r.requete = requete_gimbal
    faux.requetes.clear()
    r.observer(tour_complet=True)
    ordres = [(t, c['pan']) for t, m, ch, c in faux.requetes if ch == '/gimbal']
    assert perdus and len(vues) == 7  # no view for the pan whose order was lost
    assert len(ordres) == 8 and ordres[-1][1] == 0  # back to pan 0 at the end
    for (t_ordre, pan), (t_vue, _) in zip(ordres[:-1], vues, strict=True):
        assert t_vue - t_ordre >= 0.4 + 0.7 - 1e-9, pan  # turret, then a picture after it
    assert r._pan(r._dernier_etat) == 0.0


def test_keyboard_driving_between_two_steps_is_an_operator_stop() -> None:
    """Driven 15 cm with the keyboard then released: after manual_timeout (0.5 s) ctrl.mode is
    'idle' again with no trace in ctrl.history; only the odometer shows it."""
    faux = FauxServeur(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    faux.x += 0.15
    faux.ox += 0.15
    faux.odometre += 0.15
    n = len(chemins(faux, '/path'))
    with pytest.raises(ArretOperateur, match='entre deux pas'):
        r.deplacer((1.4, 1.0))
    assert len(chemins(faux, '/path')) == n


@pytest.mark.parametrize('recu', [True, False])
def test_the_backward_heading_move_is_sent_once(recu) -> None:
    """A /move is relative: sent again after a lost answer it drives twice. It is sent again
    only when GET /state shows that the server never got it."""
    faux = FauxServeur(1.0, 1.0, 0.0, mur_x=1.02)
    faux.cap_visible = False
    requete = faux.requete
    perdus = []

    def requete_perdue(methode, chemin, corps=None):
        if chemin == '/move' and not perdus:
            perdus.append(faux.t)
            if recu:
                requete(methode, chemin, corps)  # the server got it, its answer is lost
            return None
        return requete(methode, chemin, corps)

    r = robot(faux)
    r.requete = requete_perdue
    r.placer((1.0, 1.0))
    impact = r.deplacer((1.5, 1.0))
    assert perdus and len(chemins(faux, '/move')) == 1  # what reached the server
    assert r.cap_fiable and r.rover_joignable() and impact is not None
    assert 0.8 < faux.x < 0.9  # backed off ~0.17 m once from the wall at 1.02
