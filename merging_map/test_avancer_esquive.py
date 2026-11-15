"""Tests de avancer_esquive.py : avancer de D m droit devant en contournant ce que voit la camera
du rover, d'une traite, sur une liaison lente.

- Contre le faux serveur de test_rover_reel.py (horloge simulee) rendu LENT : chaque requete
  coute 1,5 s, une reponse sur 10 est perdue (le client attend alors son delai de 8 s). On
  compte les requetes et le temps, et on compare a la boucle par pas qui a roule sur le vrai
  rover le 15 novembre.
- Contre le VRAI serveur du rover (rover_server.App + rover_core + ESP32 simule, banc_reel.Banc)
  dans le processus, ports 127.0.0.1:8275-8279 : un essai de bout en bout par main(), en temps
  reel (une vingtaine de secondes).

La perception est simulee (banc_reel.percevoir_simule) : elle voit depuis la pose VRAIE du rover
et le cap VRAI de la camera (cap du rover - pan de la tourelle)."""

import json
import math

import pytest

pytest.importorskip('cv2')

import avancer_esquive as ae  # noqa: E402
import banc_reel  # noqa: E402
from banc_reel import Box, percevoir_simule  # noqa: E402
from fusion_carte import CarteFusion, decouper  # noqa: E402
from lancer_overhead import RAYON_ROVER, sans_post  # noqa: E402
from rover_reel import RaspRoverReel, angle_normalise  # noqa: E402
from test_rover_reel import FauxServeur  # noqa: E402

D = ae.DECALAGE
PORTS_BANC = range(8275, 8280)
PORTEE, DEMI_CHAMP = 2.0, math.radians(45)


class FauxSegments(FauxServeur):
    """Le faux serveur, plus ce que le vrai fait et dont ce pilote se sert : un /path a
    plusieurs points suivis dans l'ordre, et la tourelle (gimbal dans GET /state, POST
    /gimbal {pan}, 0,3 s de mouvement)."""

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.pan, self.fin_tourelle = 0.0, 0.0

    def _reponse(self, methode, chemin, corps):
        if chemin == '/gimbal':
            self.pan = float(corps.get('pan', self.pan))
            self.fin_tourelle = self.t + 0.3
            return {'ok': True}
        reponse = super()._reponse(methode, chemin, corps)
        if chemin == '/path':
            tache = self.queued[-1]
            tache['reste'] = [tuple(p) for p in corps['points']]
            tache['but'] = tache['reste'].pop(0)
        return reponse

    def _rouler(self):
        k = self.tache
        if k.get('reste') and math.dist(k['but'], (self.ox, self.oy)) < 0.05:
            k['but'] = k['reste'].pop(0)
        super()._rouler()

    def etat(self):
        etat = super().etat()
        etat['gimbal'] = {'pan': self.pan, 'tilt': -4.5, 'moving': self.t < self.fin_tourelle}
        return etat


def lent(faux, delai=1.5, perte=10, delai_http=8.0, avant=None):
    """requete() vers `faux` a travers une liaison lente : chaque requete coute `delai` s
    (moitie a l'aller, moitie au retour), la reponse d'une requete sur `perte` se perd (le
    serveur l'a recue ; le client attend son delai_http). `avant(methode, chemin)` : appele a
    l'arrivee de chaque requete au serveur."""
    n = [0]

    def requete(methode, chemin, corps=None):
        n[0] += 1
        faux.dormir(delai / 2)
        if avant is not None:
            avant(methode, chemin)
        reponse = faux.requete(methode, chemin, corps)
        if perte and n[0] % perte == 0:
            faux.dormir(delai_http - delai / 2)
            return None
        faux.dormir(delai / 2)
        return reponse

    return requete


def percevoir_faux(faux, boites, delai=0.0):
    """La camera du rover sur le faux : vue depuis la pose vraie, cap vrai de la camera."""
    voir = percevoir_simule(
        lambda: (faux.x, faux.y, faux.cap - math.radians(faux.pan)),
        boites,
        portee=PORTEE,
        demi_champ=DEMI_CHAMP,
    )

    def percevoir(position, cap):
        faux.dormir(delai)  # le service de perception prend l'image du rover par la liaison
        return voir(position, cap)

    return percevoir


def distance_boite(p, b):
    dx = max(b.x - p[0], 0.0, p[0] - (b.x + b.width))
    dy = max(b.y - p[1], 0.0, p[1] - (b.y + b.height))
    return math.hypot(dx, dy)


def posts(faux, chemin):
    """Corps des POST `chemin` recus par le faux serveur."""
    return [corps for _, m, c, corps in faux.requetes if (m, c) == ('POST', chemin)]


def conduire(faux, requete, percevoir, journal=None, **options):
    lignes = [] if journal is None else journal
    bilan = {}
    issue = ae.conduire(
        requete,
        percevoir,
        journal=lignes.append,
        dormir=faux.dormir,
        horloge=faux.horloge,
        bilan=bilan,
        **options,
    )
    return issue, bilan, lignes


# ---- morceaux ------------------------------------------------------------------------------
def test_un_troncon_ne_va_jamais_dans_des_cases_pas_vues() -> None:
    """Le chemin traverse une zone vue libre puis des cases jamais vues : le troncon s'arrete
    un rayon de rover avant la premiere case pas vue."""
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.2)
    vue = [[0] * 20 for _ in range(10)]  # 1 m devant, 0,5 m de large, vu libre
    for msg in decouper(vue, D, D - 0.25, 4, carte, 'rover'):
        carte.recevoir(msg)
    zone = (D - 0.8, D - 1.5, D + 3.0, D + 1.5)
    dec = carte.decider((D, D), (D + 2.5, D), zone=zone, tolerance=0.2, drone_disponible=False)
    points, raison = ae.troncon_vu(dec['chemin'], dec['grille'], carte, 2.0, (D, D))
    assert raison == 'inconnu'
    assert 0.7 <= points[-1][0] - D <= 0.8 + 1e-6  # 1 m vu, moins 0,2 m de rover
    assert len(points) == 2  # tout droit : un seul point envoye
    points, raison = ae.troncon_vu(dec['chemin'], dec['grille'], carte, 0.5, (D, D))
    assert raison == 'vu' and abs(points[-1][0] - D - 0.5) < 0.03


# ---- faux serveur, liaison rapide -------------------------------------------------------
def test_une_boite_qui_apparait_apres_le_premier_segment_est_evitee() -> None:
    """La boite n'est pas la au depart (rien a voir) : elle apparait sur la ligne une fois le
    premier segment fini. Le rover la voit apres l'arret et la contourne."""
    faux = FauxSegments(D, D, 0.0)
    boites = []
    boite = Box(D + 2.0, D - 0.25, 0.4, 0.5)

    def avant(methode, chemin):  # le premier segment est fini : la boite arrive
        if chemin == '/state' and posts(faux, '/path') and faux.history and not boites:
            boites.append(boite)

    issue, bilan, lignes = conduire(
        faux, lent(faux, 0.0, 0, avant=avant), percevoir_faux(faux, boites), distance=4.0
    )
    assert issue == 'arrive', lignes
    assert boites
    assert min(distance_boite(p, boite) for p in faux.trace) >= RAYON_ROVER
    assert math.dist((faux.x, faux.y), (D + 4.0, D)) < 0.35


def test_un_mur_sur_tout_le_couloir_bloque_apres_un_tour_de_tourelle() -> None:
    faux = FauxSegments(D, D, 0.0)
    mur = Box(D + 1.0, D - 2.0, 0.2, 4.0)
    issue, _, lignes = conduire(faux, faux.requete, percevoir_faux(faux, [mur]), distance=4.0)
    assert issue == 'bloque', lignes
    pans = [c['pan'] for c in posts(faux, '/gimbal')]
    assert -40 in pans and 40 in pans and pans[-1] == 0
    assert posts(faux, '/path') == []  # le rover n'a pas bouge
    assert (faux.x, faux.y) == (D, D)


def test_sans_perception_le_rover_ne_bouge_pas() -> None:
    faux = FauxSegments(D, D, 0.0)
    issue, _, lignes = conduire(faux, faux.requete, lambda position, cap: None, distance=3.0)
    assert issue == 'perception_indisponible', lignes
    assert posts(faux, '/path') == []


def test_un_stop_operateur_pendant_un_segment_est_definitif() -> None:
    faux = FauxSegments(D, D, 0.0)
    fait = []

    def avant(methode, chemin):
        if (methode, chemin) == ('GET', '/state') and faux.x > D + 0.3 and not fait:
            fait.append(faux.t)
            faux.stop_operateur()

    issue, _, lignes = conduire(
        faux, lent(faux, 0.0, 0, avant=avant), percevoir_faux(faux, []), distance=4.0
    )
    assert issue == 'arret_operateur', lignes
    assert fait and len(posts(faux, '/path')) == 1  # plus rien apres le STOP
    assert faux.mode == 'idle'


def test_essai_n_envoie_aucun_post() -> None:
    faux = FauxSegments(D, D, 0.0)
    issue, _, lignes = conduire(
        faux,
        sans_post(faux.requete),
        percevoir_faux(faux, [Box(D + 1.0, D - 0.2, 0.3, 0.4)]),
        distance=3.0,
        essai=True,
    )
    assert issue == 'essai', lignes
    assert [r for r in faux.requetes if r[1] == 'POST'] == []  # (t, methode, chemin, corps)
    assert any('[essai]' in ligne for ligne in lignes)


# ---- faux serveur, liaison LENTE --------------------------------------------------------
def boucle_par_pas(requete, percevoir, horloge, dormir, distance, pas=0.25, max_pas=80):
    """La boucle qui a roule sur le vrai rover le 15 novembre (avance_esquive.py) : un pas
    de 25 cm par cycle percevoir / decider / deplacer, avec les reglages de ce jour
    (delai_liaison 10 s, periode 0,5 s) ; sa pseudo-camera du dessus fait un GET /state a
    chaque lecture. Renvoie (issue, nombre de pas)."""
    from cerveau import avancer

    class Odometrie:
        def __init__(self):
            etat = requete('GET', '/state')
            self.x0, self.y0, self.h0 = ae.pose_odom(etat)

        def pose(self):
            etat = requete('GET', '/state')
            if etat is None:
                return None
            x, y, h = ae.pose_odom(etat)
            dx, dy = x - self.x0, y - self.y0
            c, s = math.cos(-self.h0), math.sin(-self.h0)
            return D + c * dx - s * dy, D + s * dx + c * dy, angle_normalise(h - self.h0)

        def disponible(self):
            return True

        def zoomer(self, tuile):
            return []

    odo = Odometrie()
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.2)
    rover = RaspRoverReel(
        carte, requete, odo.pose, percevoir, odo, vitesse=0.12, delai_liaison=10.0,
        periode=0.5, delai_pret=30.0, horloge=horloge, dormir=dormir,
    )  # fmt: skip
    but = (D + distance, D)
    zone = (D - 0.8, D - 1.5, D + distance + 0.8, D + 1.5)
    rover.placer((D, D))
    rover.recaler()
    for k in range(max_pas):
        pos = rover.position()
        if math.dist(pos, but) <= 0.25:
            return 'arrive', k
        vue = percevoir(rover.est, rover.cap)
        if vue is not None:
            rover.publier_vue(*vue)
        dec = carte.decider(pos, but, zone=zone, tolerance=0.25, drone_disponible=False)
        chemin = [tuple(p) for p in dec.get('chemin') or []]
        if len(chemin) < 2:
            return ('arrive' if math.dist(pos, but) <= 0.35 else 'bloque'), k
        cible, _ = avancer(pos, list(chemin), pas)
        rover.deplacer(cible)
        if not rover.rover_joignable():
            return 'liaison_perdue', k
    return 'max_pas', max_pas


def test_liaison_lente_5_m_d_une_traite_et_peu_de_requetes(capsys) -> None:
    """Chaque requete coute 1,5 s et une reponse sur 10 se perd (8 s d'attente) ; la
    perception coute 1,5 s aussi. 5 m avec une boite sur la ligne : arrive, sans fausse
    « liaison perdue », au plus 6 requetes par segment. Comparaison avec l'ancienne boucle
    par pas sur le meme faux."""
    boite = Box(D + 1.6, D - 0.25, 0.4, 0.5)
    faux = FauxSegments(D, D, 0.0)
    issue, bilan, lignes = conduire(
        faux, lent(faux), percevoir_faux(faux, [boite], delai=1.5), distance=5.0
    )
    assert issue == 'arrive', lignes
    assert not any('liaison perdue' in ligne for ligne in lignes), lignes
    assert min(distance_boite(p, boite) for p in faux.trace) >= RAYON_ROVER
    assert bilan['par_segment'] and max(bilan['par_segment']) <= 6, bilan['par_segment']
    metres = bilan['metres']
    assert metres >= 4.5
    t_nouveau, n_nouveau = bilan['fin'] - bilan['debut'], len(bilan['requetes'])

    ancien = FauxSegments(D, D, 0.0)
    debut = ancien.t
    issue_ancien, pas = boucle_par_pas(
        lent(ancien),
        percevoir_faux(ancien, [boite], delai=1.5),
        ancien.horloge,
        ancien.dormir,
        5.0,
    )
    t_ancien = ancien.t - debut
    metres_ancien = sum(
        math.dist(a, b) for a, b in zip(ancien.trace, ancien.trace[1:], strict=False)
    )
    with capsys.disabled():
        print(
            f'\n[liaison lente] segments : {issue}, {metres:.2f} m, {bilan["segments"]} '
            f'segments, {bilan["tours"]} tours de tourelle, {n_nouveau} requetes '
            f'({n_nouveau / metres:.1f}/m, par segment {bilan["par_segment"]}), '
            f'{t_nouveau:.0f} s ({t_nouveau / metres:.1f} s/m)\n'
            f'[liaison lente] ancienne boucle par pas : {issue_ancien}, {pas} pas, '
            f'{metres_ancien:.2f} m, {len(ancien.requetes)} requetes '
            f'({len(ancien.requetes) / max(metres_ancien, 0.01):.1f}/m), {t_ancien:.0f} s '
            f'({t_ancien / max(metres_ancien, 0.01):.1f} s/m)'
        )
    assert t_nouveau / metres < 0.5 * t_ancien / max(metres_ancien, 0.01)


# ---- le vrai serveur du rover, de bout en bout --------------------------------------------
@pytest.fixture
def banc(monkeypatch):
    monkeypatch.setattr(banc_reel, 'PORTS', PORTS_BANC)
    b = banc_reel.Banc()
    b.attendre_pret()
    yield b
    b.fermer()


def percevoir_banc(banc, boites):
    """La camera du rover sur le banc, dans le repere de la mission (origine au depart
    decalee de D, x droit devant) ; tourelle lue dans le rover simule."""
    x0, y0, c0 = banc.pose_vraie()

    def vers_mission(x, y, cap):
        dx, dy = x - x0, y - y0
        c, s = math.cos(-c0), math.sin(-c0)
        return D + c * dx - s * dy, D + s * dx + c * dy, angle_normalise(cap - c0)

    def pose_camera():
        x, y, cap = vers_mission(*banc.pose_vraie())
        return x, y, cap - math.radians(banc.rover.gimbal.pan)

    voir = percevoir_simule(pose_camera, boites, portee=PORTEE, demi_champ=DEMI_CHAMP)
    return voir, vers_mission


def test_banc_reel_contourne_une_boite_de_bout_en_bout(banc, tmp_path, capsys) -> None:
    """main() contre le vrai serveur : une boite sur la ligne a 1 m, but a 2,2 m. Arrive, la
    pose vraie ne s'approche jamais de la boite a moins du rayon du rover, /stop a la fin, et
    le document --suivi est ecrit."""
    boite = Box(D + 1.0, D - 0.2, 0.3, 0.4)
    percevoir, vers_mission = percevoir_banc(banc, [boite])
    suivi = tmp_path / 'suivi.json'
    issue = ae.main(
        ['--rover', banc.url, '--distance', '2.2', '--vitesse', '0.25', '--suivi', str(suivi)],
        percevoir=percevoir,
    )
    assert issue == 'arrive'
    poses = [vers_mission(x, y, c) for _, x, y, c in banc.poses]
    assert min(distance_boite(p, boite) for p in poses) >= RAYON_ROVER
    assert math.dist(poses[-1][:2], (D + 2.2, D)) < 0.35
    chemins = [c for _, c, _, _ in banc.recus]
    assert chemins[-1] == '/stop'
    assert 1 <= chemins.count('/path') <= 5
    doc = json.loads(suivi.read_text())
    assert doc['etat'] == 'arrive' and len(doc['traj']) >= 2
    sortie = capsys.readouterr().out
    assert 'requetes' in sortie and 's/m' in sortie


def test_banc_reel_essai_n_envoie_aucun_post(banc) -> None:
    percevoir, _ = percevoir_banc(banc, [])
    assert ae.main(['--rover', banc.url, '--essai'], percevoir=percevoir) == 'essai'
    assert banc.recus == []  # Banc.recus note chaque POST recu
