"""The real-mission launcher (lancer_overhead): fails fast without a rover, shows what it sees
before moving, stops the rover on every way out, dry-run sends no POST, writes the live
follow-up file, and its command line wires camera, scale, perception and HTTP delay."""

import json
import math
import os
import signal
import sys
import types
from pathlib import Path

import lancer_overhead
import pytest
from lancer_overhead import arreter_le_rover, lancer, main, sans_post
from rover_reel import ArretOperateur
from test_lancer_overhead import FausseCamera
from test_rover_reel import FauxServeur

RACINE = Path(__file__).resolve().parent.parent


def options(faux, **autres):
    return {'horloge': faux.horloge, 'dormir': faux.dormir, **autres}


def posts(faux):
    return [(t, ch) for t, m, ch, _ in faux.requetes if m == 'POST']


def test_unreachable_rover_fails_fast_and_sends_nothing() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    faux.joignable = False
    with pytest.raises(RuntimeError, match='GET /state'):
        lancer(FausseCamera(faux), faux.requete, (2.0, 2.0), journal=lambda _: None)
    assert posts(faux) == []
    assert len(faux.requetes) == 1  # one GET, not hours of retries


def test_start_shows_zone_rover_and_scale() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    lignes = []
    lancer(
        FausseCamera(faux),
        faux.requete,
        (2.0, 2.0),
        options_rover=options(faux),
        journal=lignes.append,
    )
    texte = '\n'.join(lignes[:4])
    assert 'Zone vue par la camera : 2.40 x 2.40 m' in texte
    assert 'Rover vu par la camera : (0.40, 0.40) m, cap inconnu' in texte
    assert 'echelle' in texte


def test_operator_stop_ends_the_mission_cleanly_with_a_stop() -> None:
    faux = FauxServeur(0.4, 0.4, math.pi)
    dormir, fait = faux.dormir, []

    def dormir_et_stop(dt):
        dormir(dt)
        envois = [t for t, ch in posts(faux) if ch == '/path']
        if len(envois) >= 3 and not fait:
            fait.append(faux.t)
            faux.stop_operateur()

    faux.dormir = dormir_et_stop
    lignes = []
    resultat = lancer(
        FausseCamera(faux),
        faux.requete,
        (2.0, 2.0),
        options_rover=options(faux),
        journal=lignes.append,
    )
    assert resultat['etat'] == 'arret_operateur' and 'stopped' in resultat['raison']
    apres = [ch for t, ch in posts(faux) if t >= fait[0]]
    assert apres == ['/stop']  # no step after the operator's stop, then our own /stop
    assert any('ARRET OPERATEUR' in ligne for ligne in lignes)
    assert len(resultat['fin']['traj']) >= 2


@pytest.mark.parametrize('exception', [KeyboardInterrupt, ZeroDivisionError])
def test_any_exit_sends_stop(exception) -> None:
    """Ctrl-C or a crash in the middle of a step: the rover is told to stop."""
    faux = FauxServeur(0.4, 0.4, 0.0)
    dormir = faux.dormir

    def dormir_puis_casser(dt):
        dormir(dt)
        if any(ch == '/path' for _, ch in posts(faux)) and faux.mode == 'path':
            raise exception

    faux.dormir = dormir_puis_casser
    with pytest.raises(exception):
        lancer(
            FausseCamera(faux),
            faux.requete,
            (2.0, 2.0),
            options_rover=options(faux),
            journal=lambda _: None,
        )
    assert posts(faux)[-1][1] == '/stop'
    assert faux.history[0]['status'] == 'stopped'


def test_stop_is_retried_when_its_answer_is_lost() -> None:
    reponses = [None, None, {'ok': True}]
    envoyes = []

    def requete(methode, chemin, corps=None):
        envoyes.append((methode, chemin))
        return reponses.pop(0)

    assert arreter_le_rover(requete, journal=lambda _: None, dormir=lambda _: None)
    assert envoyes == [('POST', '/stop')] * 3


def test_dry_run_sends_no_post_and_stops_after_the_first_decision() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    lignes = []
    resultat = lancer(
        FausseCamera(faux),
        sans_post(faux.requete, lignes.append),
        (2.0, 2.0),
        options_rover=options(faux),
        journal=lignes.append,
        dry_run=True,
    )
    assert resultat['etat'] == 'dry_run' and resultat['decision']['action']
    assert posts(faux) == [] and faux.t == 0.0  # nothing sent, nothing waited for
    texte = '\n'.join(lignes)
    assert '[dry-run] decision' in texte and '[dry-run] pas 1' in texte
    assert 'cap du rover inconnu' in texte
    with pytest.raises(RuntimeError, match='dry-run'):
        sans_post(faux.requete, lignes.append)('POST', '/path', {'points': [[0, 0]]})


def test_follow_up_file_is_written_during_and_after_the_mission(tmp_path) -> None:
    faux = FauxServeur(0.4, 0.4, math.pi)
    fichier = tmp_path / 'suivi.json'
    vus = []
    dormir = faux.dormir

    def dormir_et_lire(dt):
        dormir(dt)
        if fichier.exists() and not vus:
            vus.append(json.loads(fichier.read_text()))

    faux.dormir = dormir_et_lire
    lancer(
        FausseCamera(faux),
        faux.requete,
        (2.0, 2.0),
        options_rover=options(faux),
        journal=lambda _: None,
        suivi=str(fichier),
    )
    assert vus and vus[0]['etat'] == 'en_cours'
    doc = json.loads(fichier.read_text())
    assert doc['etat'] == 'arrive'
    assert set(doc) >= {
        't',
        'etat',
        'rover',
        'victime',
        'zone',
        'decision',
        'chemin',
        'traj',
        'journal',
        'grille',
    }
    assert doc['victime'] == [2.0, 2.0] and doc['zone'] == [0.0, 0.0, 2.4, 2.4]
    assert len(doc['traj']) >= 5 and len(doc['journal']) <= 30
    assert doc['decision']['action'] and doc['decision']['raison']
    grille = doc['grille']
    assert grille['res'] == 0.05 and len(grille['lignes']) == 48
    assert all(len(ligne) == 48 and set(ligne) <= set('012?') for ligne in grille['lignes'])
    # the box of the fake camera (x 1.0..1.4, y 0.9..1.5), row 0 = south
    assert grille['lignes'][24][24] == '1' and grille['lignes'][5][5] == '0'
    assert list(tmp_path.iterdir()) == [fichier]  # no temp file left behind


def test_follow_up_file_says_operator_stop(tmp_path) -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    fichier = tmp_path / 'suivi.json'
    camera = FausseCamera(faux)
    requete = faux.requete

    def requete_et_arret(methode, chemin, corps=None):
        if chemin == '/path':
            raise ArretOperateur('test')
        return requete(methode, chemin, corps)

    resultat = lancer(
        camera,
        requete_et_arret,
        (2.0, 2.0),
        options_rover=options(faux),
        journal=lambda _: None,
        suivi=str(fichier),
    )
    assert resultat['etat'] == 'arret_operateur'
    assert json.loads(fichier.read_text())['etat'] == 'arret_operateur'


# ---- command line ---------------------------------------------------------------------------


@pytest.fixture
def faux_modules(monkeypatch):
    """guidance_global_map replaced by recorders: main() is tested without OpenCV."""
    vu = {}

    class Reglages:
        def __init__(self, **kw):
            vu['reglages'] = kw

    class Drone:
        def __init__(self, lecteur, reglages):
            vu['lecteur'] = lecteur

    class Lecteur:
        def __init__(self, source, **options):
            vu['source'], vu['options_lecteur'] = source, options

        def __call__(self):
            return 'image'  # the stream has its first frame: main() does not wait

        def close(self):
            vu['ferme'] = True

    drone = types.ModuleType('guidance_global_map.drone')
    drone.OverheadDrone = Drone
    drone.image_file_reader = lambda chemin: ('image', chemin)
    drone.camera_reader = Lecteur
    pipeline = types.ModuleType('guidance_global_map.pipeline')
    pipeline.OverheadSettings = Reglages
    paquet = types.ModuleType('guidance_global_map')
    monkeypatch.setitem(sys.modules, 'guidance_global_map', paquet)
    monkeypatch.setitem(sys.modules, 'guidance_global_map.drone', drone)
    monkeypatch.setitem(sys.modules, 'guidance_global_map.pipeline', pipeline)

    def faux_lancer(camera, requete, victime, depart, **kw):
        vu['lancer'] = {'requete': requete, 'victime': victime, 'depart': depart, **kw}
        return {'etat': 'dry_run' if kw['dry_run'] else 'arrive', 'fin': {'traj': [(0, 0)]}}

    def faux_client(url, delai=1.0):
        vu['client'] = (url, delai)
        return lambda methode, chemin, corps=None: {'ok': True}

    monkeypatch.setattr(lancer_overhead, 'lancer', faux_lancer)
    monkeypatch.setattr(lancer_overhead, 'client_http', faux_client)
    return vu


def test_command_line_refuses_a_still_photo_to_drive_the_rover(faux_modules) -> None:
    base = ['--victime', '1', '1']
    with pytest.raises(SystemExit):
        main(['--image', 'p.jpg', '--rover', 'http://127.0.0.1:8100', *base])
    with pytest.raises(SystemExit):
        main(['--camera', '0', *base])  # no --rover and no --dry-run
    with pytest.raises(SystemExit):
        main(['--camera', '0', '--image', 'p.jpg', '--dry-run', *base])
    assert 'lancer' not in faux_modules
    main(['--image', 'p.jpg', '--dry-run', *base])
    assert faux_modules['lancer']['dry_run'] is True


def test_command_line_wires_camera_scale_and_options(faux_modules, monkeypatch, tmp_path) -> None:
    perception = types.ModuleType('perception_rover')
    perception.percevoir_http = lambda url: ('percevoir', url)
    monkeypatch.setitem(sys.modules, 'perception_rover', perception)
    main(
        [
            '--camera',
            'http://127.0.0.1:8101/snapshot.jpg',
            '--rover',
            'http://127.0.0.1:8102',
            '--victime',
            '1.2',
            '0.9',
            '--metres-par-pixel',
            '0.002',
            '--focale-35mm',
            '24',
            '--capteur',
            '1280',
            '720',
            '--marqueur',
            '10',
            '--hauteur-marqueur',
            '0.15',
            '--perception',
            'http://127.0.0.1:8103',
            '--suivi',
            str(tmp_path / 'suivi.json'),
            '--tolerance',
            '0.2',
            '--vitesse',
            '0.15',
            '--delai-http',
            '2.5',
        ]
    )
    assert faux_modules['source'] == 'http://127.0.0.1:8101/snapshot.jpg'
    assert faux_modules['options_lecteur'] == {'latency': 0.5}  # --latence by default
    assert faux_modules['ferme'] is True  # the live reader is closed at the end
    assert faux_modules['reglages'] == {
        'camera_height': 1.8,
        'metres_per_pixel': 0.002,
        'focal_length_35mm': 24.0,
        'sensor_size': (1280, 720),
        'rover_marker_id': 10,
        'marker_height': 0.15,
    }
    assert faux_modules['client'] == ('http://127.0.0.1:8102', 2.5)
    appel = faux_modules['lancer']
    assert appel['percevoir'] == ('percevoir', 'http://127.0.0.1:8103')
    assert appel['suivi'] == str(tmp_path / 'suivi.json') and appel['tolerance'] == 0.2
    assert appel['options_rover'] == {'vitesse': 0.15} and appel['dry_run'] is False
    assert appel['victime'] == (1.2, 0.9) and appel['depart'] is None


def test_command_line_dry_run_blocks_posts(faux_modules) -> None:
    main(['--camera', '0', '--rover', 'http://127.0.0.1:8104', '--victime', '1', '1', '--dry-run'])
    requete = faux_modules['lancer']['requete']
    assert requete('GET', '/state') == {'ok': True}
    with pytest.raises(RuntimeError):
        requete('POST', '/stop', {})


def test_command_line_explains_a_missing_live_camera_reader(faux_modules) -> None:
    del sys.modules['guidance_global_map.drone'].camera_reader
    with pytest.raises(SystemExit, match='camera_reader'):
        main(['--camera', '0', '--dry-run', '--victime', '1', '1'])


def test_real_pipeline_dry_run_on_the_synthetic_photo(capsys) -> None:
    """The real OverheadDrone on the repository's photo, dry-run without a rover."""
    for paquet in ('guidance_calib', 'guidance_global_map', 'guidance_rover_pose'):
        chemin = str(RACINE / 'src' / paquet)
        if chemin not in sys.path:
            sys.path.append(chemin)
    pytest.importorskip('cv2')
    pytest.importorskip('guidance_global_map.drone')
    image = RACINE / 'data' / 'overhead' / 'synthetic_scene.png'
    resultat = main(
        [
            '--image',
            str(image),
            '--dry-run',
            '--metres-par-pixel',
            '0.003',
            '--depart',
            '0.3',
            '0.3',
            '--victime',
            '1.5',
            '1.2',
        ]
    )
    assert resultat['etat'] == 'dry_run'
    sortie = capsys.readouterr().out
    assert 'Zone vue par la camera : 2.34 x 1.89 m' in sortie  # 780 x 630 px x 3 mm
    assert '3.000 mm/pixel' in sortie and '[dry-run] decision' in sortie


# ---- the final state follows the mission's issue (cerveau.ETAT_SUIVI) -----------------------


def test_repeated_rover_errors_end_in_the_error_state(tmp_path) -> None:
    """Every mission that did not arrive ended 'abandon', even a broken rover."""
    faux = FauxServeur(0.4, 0.4, 0.0)
    requete = faux.requete

    def requete_et_panne(methode, chemin, corps=None):
        reponse = requete(methode, chemin, corps)
        if chemin == '/path':
            faux.panne()  # the ESP32 never answers again
        return reponse

    fichier = tmp_path / 'suivi.json'
    resultat = lancer(
        FausseCamera(faux),
        requete_et_panne,
        (2.0, 2.0),
        options_rover=options(faux),
        journal=lambda _: None,
        suivi=str(fichier),
    )
    assert resultat['issue'] == 'erreur_robot' and resultat['etat'] == 'erreur'
    assert 'no ESP32 feedback' in resultat['raison']
    assert json.loads(fichier.read_text())['etat'] == 'erreur'
    assert posts(faux)[-1][1] == '/stop'


def test_a_degraded_goal_ends_in_abandon_and_an_arrival_in_arrive(tmp_path) -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    fichier = tmp_path / 'suivi.json'
    resultat = lancer(
        FausseCamera(faux),
        faux.requete,
        (1.2, 1.2),  # inside the fake camera's box: no way to reach it
        options_rover=options(faux),
        journal=lambda _: None,
        suivi=str(fichier),
    )
    assert resultat['issue'] == 'objectif_degrade' and resultat['etat'] == 'abandon'
    assert json.loads(fichier.read_text())['etat'] == 'abandon'
    faux = FauxServeur(0.4, 0.4, 0.0)
    resultat = lancer(
        FausseCamera(faux),
        faux.requete,
        (2.0, 2.0),
        options_rover=options(faux),
        journal=lambda _: None,
    )
    assert resultat['issue'] == 'arrivee' and resultat['etat'] == 'arrive'


def test_command_line_prints_the_issue_and_its_reason(faux_modules, monkeypatch, capsys) -> None:
    def faux_lancer(*args, **kw):
        return {
            'etat': 'erreur',
            'issue': 'erreur_robot',
            'raison': 'statut no ESP32 feedback',
            'fin': {'traj': [(0, 0), (0.5, 0)]},
        }

    monkeypatch.setattr(lancer_overhead, 'lancer', faux_lancer)
    main(['--camera', '0', '--rover', 'http://127.0.0.1:8185', '--victime', '1', '1',
          '--metres-par-pixel', '0.002'])  # fmt: skip
    sortie = capsys.readouterr().out
    assert 'Fin de mission (erreur) : 0.50 m parcourus.' in sortie
    assert 'Issue : erreur_robot ; raison : statut no ESP32 feedback' in sortie


# ---- review round of 15 Nov: final /stop, signals, follow-up file, slow link ----------------


def test_a_second_ctrl_c_does_not_cancel_the_final_stop() -> None:
    """The operator presses Ctrl-C again while the final /stop goes out: it was interrupted
    and main() still said '/stop envoye'."""
    faux = FauxServeur(0.4, 0.4, 0.0)
    dormir = faux.dormir

    def dormir_puis_ctrl_c(dt):
        dormir(dt)
        if any(ch == '/path' for _, ch in posts(faux)) and faux.mode == 'path':
            raise KeyboardInterrupt

    faux.dormir = dormir_puis_ctrl_c
    requete = faux.requete

    def requete_et_second_ctrl_c(methode, chemin, corps=None):
        if chemin == '/stop':
            os.kill(os.getpid(), signal.SIGINT)  # the second Ctrl-C, as the /stop leaves
        return requete(methode, chemin, corps)

    with pytest.raises(KeyboardInterrupt) as e:
        lancer(
            FausseCamera(faux),
            requete_et_second_ctrl_c,
            (2.0, 2.0),
            options_rover=options(faux),
            journal=lambda _: None,
        )
    assert posts(faux)[-1][1] == '/stop' and faux.history[0]['status'] == 'stopped'
    assert e.value.stop_recu is True
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored


@pytest.mark.parametrize(
    ('stop_recu', 'attendu'),
    [(True, '/stop recu'), (False, 'SANS reponse'), (None, 'aucun pas envoye')],
)
def test_command_line_says_whether_the_stop_was_received(faux_modules, monkeypatch,
                                                         stop_recu, attendu) -> None:  # fmt: skip
    def faux_lancer(*args, **kw):
        e = KeyboardInterrupt()
        if stop_recu is not None:
            e.stop_recu = stop_recu
        raise e

    monkeypatch.setattr(lancer_overhead, 'lancer', faux_lancer)
    with pytest.raises(SystemExit, match=attendu):
        main(['--camera', '0', '--rover', 'http://r', '--victime', '1', '1',
              '--metres-par-pixel', '0.002'])  # fmt: skip


def test_sigterm_and_sighup_stop_the_mission_like_ctrl_c(faux_modules, monkeypatch) -> None:
    """The SSH session to the VM drops (SIGHUP) or systemd stops it (SIGTERM): the process died
    without its finally, so without /stop. Now they raise KeyboardInterrupt, then restored."""
    vus = {}

    def faux_lancer(*args, **kw):
        for nom in ('SIGTERM', 'SIGHUP'):
            vus[nom] = signal.getsignal(getattr(signal, nom))
        vus['SIGTERM'](signal.SIGTERM, None)

    monkeypatch.setattr(lancer_overhead, 'lancer', faux_lancer)
    avant = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)
    with pytest.raises(SystemExit, match='Interrompu'):
        main(['--camera', '0', '--rover', 'http://r', '--victime', '1', '1',
              '--metres-par-pixel', '0.002'])  # fmt: skip
    assert callable(vus['SIGHUP'])
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)) == avant


def test_follow_up_file_errors_do_not_end_the_mission(tmp_path) -> None:
    """The follow-up file is only a display: a write error is said once, the mission goes on."""
    faux = FauxServeur(0.4, 0.4, 0.0)
    lignes = []
    resultat = lancer(
        FausseCamera(faux),
        faux.requete,
        (2.0, 2.0),
        options_rover=options(faux),
        journal=lignes.append,
        suivi=str(tmp_path / 'absent' / 'mission.json'),
    )
    assert resultat['etat'] == 'arrive'
    assert sum('suivi non ecrit' in ligne for ligne in lignes) == 1


def test_command_line_refuses_a_follow_up_file_in_a_missing_folder(faux_modules, tmp_path):
    absent = str(tmp_path / 'absent' / 'm.json')
    with pytest.raises(SystemExit):
        main(['--camera', '0', '--rover', 'http://r', '--victime', '1', '1',
              '--metres-par-pixel', '0.002', '--suivi', absent])  # fmt: skip
    assert 'lancer' not in faux_modules


def test_command_line_slow_link_options(faux_modules) -> None:
    base = ['--camera', '0', '--rover', 'http://r', '--victime', '1', '1',
            '--metres-par-pixel', '0.002']  # fmt: skip
    main([*base, '--liaison-lente'])
    assert faux_modules['client'] == ('http://r', 8.0)
    assert faux_modules['lancer']['options_rover'] == {
        'delai_liaison': 15.0,
        'attente_pas': True,
        'periode': 1.0,
    }
    main([*base, '--liaison-lente', '--delai-http', '5', '--delai-liaison', '20'])
    assert faux_modules['client'] == ('http://r', 5.0)
    assert faux_modules['lancer']['options_rover']['delai_liaison'] == 20.0
    main([*base, '--delai-liaison', '6'])
    assert faux_modules['client'] == ('http://r', 1.0)  # defaults unchanged
    assert faux_modules['lancer']['options_rover'] == {'delai_liaison': 6.0}
    main(base)
    assert faux_modules['lancer']['options_rover'] == {}


def test_command_line_checks_the_rover_before_the_camera(faux_modules, monkeypatch) -> None:
    """A rover that does not answer is said at once, before the 15 s wait for a first frame."""
    monkeypatch.setattr(
        lancer_overhead, 'client_http', lambda url, delai=1.0: lambda m, c, corps=None: None
    )
    with pytest.raises(SystemExit, match='ne repond pas a GET /state') as fin:
        main(['--camera', '0', '--rover', 'http://r', '--victime', '1', '1',
              '--metres-par-pixel', '0.002'])  # fmt: skip
    assert str(fin.value.code).startswith('ERREUR') and "Rien n'a ete envoye" in fin.value.code
    assert 'source' not in faux_modules  # the camera was not even opened
    assert 'lancer' not in faux_modules
