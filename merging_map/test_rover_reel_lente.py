"""The real-rover adapter over a SLOW link (15 Nov: reverse SSH tunnel, 1.3 to 4 s per HTTP
request whatever its size, some answers lost): a slow answer is not a lost link, the mission
still arrives, and the number of requests per step is what costs. Fake clock only."""

import math

import cerveau
import numpy as np
from fusion_carte import CarteFusion, decouper
from lancer_overhead import LIAISON_LENTE
from test_rover_reel import FauxServeur, robot

# What --liaison-lente gives RaspRoverReel (the HTTP timeout goes to client_http)
LENTE = {k: v for k, v in LIAISON_LENTE.items() if k != 'delai_http'}


class LiaisonLente:
    """requete() of a slow link in front of the fake server: every request costs `aller_retour`
    s of the fake clock (the rover drives meanwhile); every `perdre_tous`-th answer is lost
    after the server got the request, and the client waits its whole HTTP timeout for it."""

    def __init__(self, faux, aller_retour=1.5, perdre_tous=7, delai_http=8.0):
        self.faux, self.aller_retour = faux, aller_retour
        self.perdre_tous, self.delai_http = perdre_tous, delai_http
        self.n = self.perdues = 0

    def requete(self, methode, chemin, corps=None):
        self.n += 1
        self.faux.dormir(self.aller_retour / 2)
        reponse = self.faux.requete(methode, chemin, corps)
        if self.perdre_tous and self.n % self.perdre_tous == 0:
            self.perdues += 1
            self.faux.dormir(self.delai_http - self.aller_retour / 2)
            return None
        self.faux.dormir(self.aller_retour / 2)
        return reponse


def serveur_du_robot(x, y, cap):
    """The robot runs RBDRB's server: no boot_id, no stop_count."""
    faux = FauxServeur(x, y, cap)
    faux.stops = None
    return faux


def carte_libre():
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
    for msg in decouper(np.zeros((48, 48)), 0.0, 0.0, 4, carte, 'drone'):
        carte.recevoir(msg)
    return carte


def requetes_par_pas(**options):
    """Requests to the rover for one 0.2 m step, heading known, steady state, fast link."""
    faux = serveur_du_robot(1.0, 1.0, 0.0)
    r = robot(faux, **options)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    n = len(faux.requetes)
    assert r.deplacer((1.4, 1.0)) is None
    assert math.dist((faux.x, faux.y), (1.4, 1.0)) < 0.03
    return [(m, ch) for _, m, ch, _ in faux.requetes[n:]]


def test_one_step_reads_the_state_once_before_its_post() -> None:
    """deplacer() read GET /state, then _faire_un_pas read it again before the POST."""
    requetes = requetes_par_pas()
    assert requetes[:2] == [('GET', '/state'), ('POST', '/path')]
    assert len(requetes) <= 10  # 1 GET + 1 POST + the follow-up at 0.2 s (was 11)


def test_waiting_for_the_step_leaves_only_a_few_reads() -> None:
    """attente_pas: no GET /state before the expected end of the step (0.2 m at 0.2 m/s)."""
    requetes = requetes_par_pas(attente_pas=True, periode=0.2)
    assert requetes[:2] == [('GET', '/state'), ('POST', '/path')]
    assert len(requetes) <= 5
    faux = serveur_du_robot(1.0, 1.0, 0.0)
    r = robot(faux, attente_pas=True)
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    r.deplacer((1.4, 1.0))
    t_post = max(t for t, m, ch, _ in faux.requetes if ch == '/path')
    lectures = [t for t, m, ch, _ in faux.requetes if t > t_post and ch == '/state']
    assert lectures and lectures[0] - t_post >= 0.2 / 0.2 - 1e-9


def mission_lente(options, journal):
    faux = serveur_du_robot(0.4, 0.4, math.radians(-120))
    faux.cap_visible = False  # no marker: the heading is measured on the way
    lien = LiaisonLente(faux)
    carte = carte_libre()
    r = robot(faux, carte, **options)
    r.requete = lien.requete
    res = cerveau.mission(
        r, carte, (0.4, 0.4), (2.0, 1.8), zone=(0, 0, 2.4, 2.4), tolerance=0.15, journal=journal
    )
    pas = sum(1 for _, m, ch, _ in faux.requetes if m == 'POST' and ch in ('/path', '/move'))
    return res, faux, lien, pas


def test_a_mission_over_a_slow_link_arrives_without_a_false_lost_link() -> None:
    """1.5 s per request and one answer in 7 lost (8 s HTTP timeout): with --liaison-lente the
    mission arrives, the link is never called lost and no answer loss becomes a no-go zone."""
    lignes = []
    res, faux, lien, pas = mission_lente(LENTE, lignes.append)
    texte = '\n'.join(lignes)
    assert res['issue'] == 'arrivee', texte
    assert 'LIAISON ROVER PERDUE' not in texte and 'CHOC' not in texte
    assert lien.perdues >= 3  # the losses did happen
    assert math.dist((faux.x, faux.y), (2.0, 1.8)) <= 0.2
    # Requests that reached the rover, per step sent (resends included), and fake-clock time.
    # Before this round (delai_liaison 15 s only): 4.8 per step, 23 per metre, 136 s.
    requetes = len(faux.requetes)
    assert requetes / pas <= 3.5, (requetes, pas)
    assert faux.t <= 115.0


def test_the_fast_link_defaults_take_a_slow_answer_for_a_lost_link() -> None:
    """Why --liaison-lente exists: with the 2 s of a LAN, one answer lost in the middle of a
    step (8 s HTTP timeout) is a lost link for the brain; with it, the step just goes on."""
    for options, joignable in (({}, False), (LENTE, True)):
        faux = serveur_du_robot(1.0, 1.0, 0.0)
        lien = LiaisonLente(faux, perdre_tous=0)
        r = robot(faux, **options)
        r.requete = lien.requete
        r.placer((1.0, 1.0))
        r.deplacer((1.2, 1.0))
        lien.perdre_tous, lien.n = 3, 0  # the 3rd request of the next step: a GET /state
        r.deplacer((1.4, 1.0))
        assert lien.perdues == 1
        assert r.rover_joignable() is joignable, options


def test_the_wait_of_attente_pas_is_not_counted_as_a_silent_link() -> None:
    """attente_pas: no GET /state for the expected 4 s of the step (0.2 m at 0.05 m/s). That
    silence is chosen, not a link that went quiet: a lost first answer (1 s) with a 2 s
    delai_liaison is still only one lost answer."""
    faux = serveur_du_robot(1.0, 1.0, 0.0)
    lien = LiaisonLente(faux, aller_retour=0.0, perdre_tous=0, delai_http=1.0)
    r = robot(faux, attente_pas=True, periode=0.2, vitesse=0.05, delai_liaison=2.0)
    r.requete = lien.requete
    r.placer((1.0, 1.0))
    r.deplacer((1.2, 1.0))
    lien.perdre_tous, lien.n = 3, 0  # GET /state, POST /path, then the first read: lost
    assert r.deplacer((1.4, 1.0)) is None
    assert lien.perdues == 1
    assert r.rover_joignable()
    assert math.dist((faux.x, faux.y), (1.4, 1.0)) < 0.03
