"""Repetition de la chaine reelle de bout en bout, sans materiel (banc_reel.py) :

camera du dessus EN DIRECT (images JPEG servies en HTTP, rendues a la pose vraie du rover)
  -> guidance_global_map.drone.camera_reader -> OverheadDrone -> lancer_overhead.lancer
  (CarteFusion, cerveau.mission, --suivi) -> rover_reel.RaspRoverReel -> HTTP
  -> le VRAI rover/rover_server.py + rover_core.py -> ESP32 simule (fake_esp32.py).

Temps reel : une mission de ~1.4 m dure 15 a 25 s. Ports 127.0.0.1:8140-8149."""

import json
import math
import threading
import time

import pytest

pytest.importorskip('cv2')

import banc_reel as br  # noqa: E402
import lancer_overhead  # noqa: E402
from banc_reel import (  # noqa: E402
    Banc,
    Box,
    CameraDessus,
    TrouNoir,
    boite_touchee,
    percevoir_simule,
)
from mission_settings import ARENA  # noqa: E402
from rover_reel import client_http  # noqa: E402

from guidance_global_map.drone import OverheadDrone, camera_reader  # noqa: E402
from guidance_global_map.pipeline import OverheadSettings  # noqa: E402

VICTIME = (1.6, 1.2)
# Sur la ligne droite du depart (T0 = (0.45, 0.45)) a la victime : il faut la contourner.
BOITE = Box(0.85, 0.65, 0.25, 0.25, color=(40, 60, 200))
DECOR = Box(1.75, 0.20, 0.30, 0.20, color=(60, 140, 40))
# Meme endroit, mais invisible d'en haut (sous une table, sous un auvent) : seul le rover la voit.
CACHEE = Box(0.95, 0.72, 0.20, 0.20)
ETATS = {'en_cours', 'arrive', 'arret_operateur', 'abandon', 'erreur'}
CLES = {'t', 'etat', 'rover', 'victime', 'zone', 'decision', 'chemin', 'traj', 'journal', 'grille'}
MARGE_PAS = 0.2  # RaspRoverReel(marge_pas=0.2), la borne d'un pas au-dela du pas demande
POSTS_QUI_BOUGENT = ('/path', '/move', '/turn', '/sequence', '/cmd')


# ---- interface 6 (document --suivi) ---------------------------------------------------------


def _nombre(v):
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _points(liste):
    return isinstance(liste, list) and all(
        isinstance(p, list) and len(p) == 2 and all(map(_nombre, p)) for p in liste
    )


def verifier_document(doc):
    """Le document de lancer_overhead --suivi suit l'interface 6 de REAL_CONTEXT.md."""
    assert set(doc) == CLES, set(doc) ^ CLES
    assert _nombre(doc['t'])
    assert doc['etat'] in ETATS
    rover = doc['rover']
    assert isinstance(rover, list) and len(rover) == 3
    assert _nombre(rover[0]) and _nombre(rover[1]) and (rover[2] is None or _nombre(rover[2]))
    assert doc['victime'] == list(VICTIME)
    assert isinstance(doc['zone'], list) and len(doc['zone']) == 4
    assert all(map(_nombre, doc['zone']))
    # null avant la premiere decision (premiere ecriture apres le premier observer()) :
    # teleop/mission.html le tolere (d.decision || {}).
    decision = doc['decision']
    assert decision is None or (
        set(decision) == {'action', 'raison'}
        and all(isinstance(v, str) for v in decision.values())
    )
    assert _points(doc['chemin']) and _points(doc['traj'])
    journal = doc['journal']
    assert isinstance(journal, list) and len(journal) <= 30
    assert all(isinstance(ligne, str) for ligne in journal)
    grille = doc['grille']
    assert set(grille) == {'x0', 'y0', 'res', 'lignes'}
    assert _nombre(grille['x0']) and _nombre(grille['y0']) and grille['res'] > 0
    lignes = grille['lignes']
    assert lignes and all(isinstance(ligne, str) for ligne in lignes)
    assert len({len(ligne) for ligne in lignes}) == 1
    assert set(''.join(lignes)) <= set('012?')


def cases_boite(doc, boite):
    """Caracteres de la grille du document aux centres des cases de la boite."""
    g, r = doc['grille'], doc['grille']['res']
    valeurs = []
    for i, ligne in enumerate(g['lignes']):
        for j, c in enumerate(ligne):
            x, y = g['x0'] + (j + 0.5) * r, g['y0'] + (i + 0.5) * r
            if boite.x < x < boite.x + boite.width and boite.y < y < boite.y + boite.height:
                valeurs.append(c)
    return valeurs


def _refuser(nom):
    raise ValueError(f'{nom} dans le JSON de suivi')


@pytest.fixture
def suivi(monkeypatch, tmp_path):
    """(fichier, documents) : chaque ecriture du suivi est relue sur le disque, telle qu'une
    page la lirait."""
    fichier, documents = tmp_path / 'suivi.json', []
    ecrire = lancer_overhead.Suivi.ecrire

    def ecrire_et_relire(self, etat=None):
        ecrire(self, etat)
        if self.fichier is not None:
            with open(self.fichier) as f:
                documents.append(json.load(f, parse_constant=_refuser))

    monkeypatch.setattr(lancer_overhead.Suivi, 'ecrire', ecrire_et_relire)
    return fichier, documents


def verifier_suivi(documents, fin):
    assert documents
    for doc in documents:
        verifier_document(doc)
    assert [d['t'] for d in documents] == sorted(d['t'] for d in documents)
    # en_cours, puis l'etat final seulement (l'evenement 'arrivee' l'ecrit deja avant la fin)
    etats = [d['etat'] for d in documents]
    k = next((i for i, e in enumerate(etats) if e != 'en_cours'), len(etats))
    assert etats[k:] and set(etats[k:]) == {fin}, etats


# ---- le banc ---------------------------------------------------------------------------------


@pytest.fixture
def banc():
    b = Banc()
    b.attendre_pret()
    yield b
    b.fermer()


def lancer_mission(banc, visibles, fichier, percevoir=None):
    """La mission comme lancer_overhead.main la monte avec --camera URL --metres-par-pixel
    --marqueur 10 --rover URL --suivi FICHIER : (resultat, lignes du journal)."""
    camera = CameraDessus(lambda: banc.pose_vraie() if banc.camera_ok else None, visibles)
    lecteur = camera_reader(camera.url, latency=0.5)
    lignes = []
    try:
        assert lancer_overhead._attendre_image(lecteur, delai=5.0)
        reglages = OverheadSettings(
            camera_height=1.9, metres_per_pixel=br.METRES_PAR_PIXEL, rover_marker_id=br.MARQUEUR
        )
        drone = OverheadDrone(lecteur, reglages)
        resultat = lancer_overhead.lancer(
            drone,
            client_http(banc.url, delai=1.0),
            VICTIME,
            percevoir=percevoir,
            journal=lignes.append,
            suivi=str(fichier),
        )
    except BaseException:
        print('\n'.join(lignes))
        raise
    finally:
        lecteur.close()
        camera.fermer()
    print('\n'.join(lignes))
    return resultat, lignes


def attendre(condition, limite=60.0):
    fin = time.monotonic() + limite
    while time.monotonic() < fin:
        if condition():
            return True
        time.sleep(0.02)
    return False


def chemins_recus(banc, chemin='/path'):
    return [(t, corps, odom) for t, c, corps, odom in banc.recus if c == chemin]


def distance_vraie(banc, t0, t1):
    poses = [(x, y) for t, x, y, _ in banc.poses if t0 <= t <= t1]
    return sum(math.dist(a, b) for a, b in zip(poses, poses[1:], strict=False))


def jamais_touchee(banc, boite):
    touches = [p for p in banc.poses if boite_touchee(p[1:], boite)]
    assert not touches, f'le vrai rover touche {boite} en {touches[0][1:]}'


# ---- les scenarios ---------------------------------------------------------------------------


def ligne_de_commande(camera_url, rover_url, fichier):
    """Ce qu'un humain tape cet apres-midi (lancer_overhead.py, sans --dry-run)."""
    return [
        '--camera',
        camera_url,
        '--metres-par-pixel',
        str(br.METRES_PAR_PIXEL),
        '--marqueur',
        str(br.MARQUEUR),
        '--rover',
        rover_url,
        '--victime',
        *map(str, VICTIME),
        '--suivi',
        str(fichier),
    ]


def test_nominal_mission_reaches_the_victim(banc, suivi) -> None:
    """Par main() : la ligne de commande, camera_reader(--latence), OverheadSettings, client_http
    (--delai-http) sont ceux du vrai essai."""
    fichier, documents = suivi
    camera = CameraDessus(banc.pose_vraie, [BOITE, DECOR])
    try:
        resultat = lancer_overhead.main(ligne_de_commande(camera.url, banc.url, fichier))
    finally:
        camera.fermer()

    assert resultat['etat'] == 'arrive'
    assert math.dist(banc.pose_vraie()[:2], VICTIME) < lancer_overhead.TOLERANCE_PETITE_ZONE + 0.05
    verifier_suivi(documents, 'arrive')
    with open(fichier) as f:
        assert json.load(f) == documents[-1]
    jamais_touchee(banc, BOITE)
    assert all(corps['frame'] == 'odom' for _, corps, _ in chemins_recus(banc))
    assert banc.recus[-1][1] == '/stop'  # dernier ordre de la mission


def test_operator_stop_during_a_step_ends_the_mission(banc, suivi) -> None:
    fichier, documents = suivi
    t_stop = []

    def operateur():
        def deuxieme_pas_en_cours():
            ctrl = banc.rover.snapshot()['ctrl']
            return ctrl['mode'] == 'path' and len(ctrl['history']) >= 1 and ctrl['progress']

        if attendre(deuxieme_pas_en_cours):
            banc.poster('/stop')  # bouton STOP de la page du rover : un autre client
            t_stop.append(time.monotonic())

    threading.Thread(target=operateur, daemon=True).start()
    resultat, lignes = lancer_mission(banc, [BOITE, DECOR], fichier)

    assert t_stop, 'le STOP est-il arrive avant la fin de la mission ?'
    assert resultat['etat'] == 'arret_operateur'
    verifier_suivi(documents, 'arret_operateur')
    assert any('ARRET OPERATEUR' in ligne for ligne in lignes)
    apres = [c for t, c, _, _ in banc.recus if t > t_stop[0]]
    assert not [c for c in apres if c in POSTS_QUI_BOUGENT], apres
    n = len(banc.board.trail)
    time.sleep(1.0)
    assert banc.roule(n) < 0.02  # il reste arrete
    ctrl = banc.rover.snapshot()['ctrl']
    assert ctrl['mode'] == 'idle' and not ctrl['queued'] and ctrl['current'] is None


def test_rover_server_restart_between_steps(banc, suivi) -> None:
    fichier, documents = suivi
    redemarre = []

    def systemctl_restart():
        def entre_deux_pas():
            ctrl = banc.rover.snapshot()['ctrl']
            # 2 lignes droites de la verification de l'echelle, puis 2 pas de la mission
            return ctrl['mode'] == 'idle' and len(ctrl['history']) >= 4

        if attendre(entre_deux_pas):
            redemarre.append(time.monotonic())
            banc.redemarrer_serveur()

    threading.Thread(target=systemctl_restart, daemon=True).start()
    resultat, lignes = lancer_mission(banc, [BOITE, DECOR], fichier)

    assert redemarre and banc.redemarrages == 1
    assert any('serveur du rover redemarre' in ligne for ligne in lignes)
    pas = chemins_recus(banc)
    assert any(t > redemarre[0] for t, _, _ in pas)  # la mission a continue apres
    # Aucun pas plus long que la borne de l'adaptateur, mesure par le serveur a la reception
    # (cible / odometrie du nouveau processus), puis sur la trajectoire vraie.
    for t, corps, odom in pas:
        assert math.dist(corps['points'][-1], odom) <= ARENA.step + MARGE_PAS + 0.01, (t, corps)
    instants = [t for t, _, _ in pas] + [time.monotonic()]
    for t0, t1 in zip(instants, instants[1:], strict=False):
        assert distance_vraie(banc, t0, t1) <= ARENA.step + MARGE_PAS + 0.05
    assert resultat['etat'] == 'arrive'
    verifier_suivi(documents, 'arrive')
    jamais_touchee(banc, BOITE)


@pytest.mark.parametrize('panne', ['trou_noir', 'refuse'])
def test_unreachable_rover_fails_fast_and_sends_nothing(panne, tmp_path) -> None:
    camera = CameraDessus(lambda: (0.6, 0.6, 0.3), [BOITE, DECOR])
    trou = TrouNoir() if panne == 'trou_noir' else None
    url = trou.url if trou else f'http://127.0.0.1:{br.port_libre()}'
    fichier = tmp_path / 'suivi.json'
    debut = time.monotonic()
    try:
        with pytest.raises(SystemExit) as fin:
            lancer_overhead.main(ligne_de_commande(camera.url, url, fichier))
        duree = time.monotonic() - debut
    finally:
        camera.fermer()
        if trou:
            trou.fermer()

    message = str(fin.value.code)
    assert message.startswith('ERREUR') and 'ne repond pas a GET /state' in message
    assert "Rien n'a ete envoye" in message
    assert duree < 5.0  # premiere image + un delai HTTP (1 s)
    assert not fichier.exists()
    if trou:
        assert trou.lignes and all(ligne.startswith('GET /state') for ligne in trou.lignes)


def test_obstacle_seen_only_by_the_rover_is_avoided(banc, suivi) -> None:
    fichier, documents = suivi
    percevoir = percevoir_simule(banc.pose_vraie, [CACHEE, DECOR])
    resultat, _ = lancer_mission(banc, [DECOR], fichier, percevoir=percevoir)

    assert any(set(cases_boite(d, CACHEE)) == {'0'} for d in documents)  # libre vue d'en haut
    assert '2' in cases_boite(documents[-1], CACHEE)  # obstacle vu par le rover
    jamais_touchee(banc, CACHEE)
    assert resultat['etat'] == 'arrive'
    verifier_suivi(documents, 'arrive')
