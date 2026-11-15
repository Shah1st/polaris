"""
avancer_esquive.py - Le rover avance de DISTANCE m droit devant lui en contournant ce que sa
propre camera voit, sans camera du dessus, D'UNE TRAITE sur une liaison lente.

    python avancer_esquive.py --rover http://127.0.0.1:18080 \\
        --perception http://127.0.0.1:8095 --distance 5 --essai   # plan seulement, AUCUN POST
    python avancer_esquive.py ... --distance 5 --suivi suivi.json  # pour de vrai, STOP pret

La liaison VM -> robot (tunnel SSH inverse) coute 1,3 a 4 s par requete HTTP, quelle que soit
sa taille, perd des reponses et se coupe parfois quelques minutes. Le 15 novembre, la boucle
par pas (un pas de 22 cm par cycle percevoir / decider / deplacer) a mis 37 s par pas. Ce qui
compte est donc le NOMBRE DE REQUETES PAR METRE, et une reponse lente n'est jamais une
coupure :
- SEGMENTS : un seul POST /path porte tout un segment (jusqu'a --segment m, quelques points) ;
  le suiveur du rover le suit seul a 20 Hz. Apres le POST, on dort la duree prevue du segment
  avant le premier GET /state, puis on interroge toutes les --periode s.
- Un segment suit le chemin du A* seulement sur des cases VUES libres par la camera du rover
  (jamais dans une case pas vue : on s'arrete un rayon de rover avant), simplifie en quelques
  points ; plus court quand la partie vue est plus courte.
- UNE image apres chaque arret. Un tour de tourelle (POST /gimbal pan -40, +40 puis 0, une
  image a chaque angle) seulement au depart, quand le chemin est ferme, ou quand il entre dans
  des cases pas vues avant la fin du prochain segment ; s'il reste moins de 0,15 m a rouler
  apres un tour fait au meme endroit : bloque.
- Rester pres de la ligne : le but est DISTANCE m devant, sur la ligne de depart ; le A*
  cherche dans un couloir de --demi-largeur m, et prend le chemin du couloir deux fois plus
  etroit s'il n'est pas beaucoup plus long.

Le reste :
- Position : l'odometrie du rover (GET /state), tenue pour la camera du dessus par un
  pseudo-drone (Odometrie) qui relit le dernier etat recu au lieu de redemander ; repere de la
  mission : origine au depart decalee de DECALAGE m, x droit devant au depart, y a gauche.
- Obstacles : percevoir(position, cap), par defaut le service de perception
  (perception_rover.percevoir_http), avec le cap de la camera = cap du rover - pan de la
  tourelle (lu dans GET /state, + = a droite).
- Carte et chemin : CarteFusion.decider(drone_disponible=False), A* sur des cases de 5 cm.
- Conduite : RoverSegments, une sous-classe de rover_reel.RaspRoverReel (STOP operateur
  definitif, redemarrage du serveur detecte, panne du serveur, borne de longueur).
- Fin : arrive, bloque, perception_indisponible, arret_operateur, erreur_rover, max_segments
  (ou essai), puis le trajet et un bilan (m roules, segments, requetes, temps, s/m).
  --suivi FICHIER ecrit le document de suivi (interface 6, lancer_overhead.Suivi) que la page
  /mission affiche. Ctrl-C (meme deux fois), SIGTERM et SIGHUP envoient /stop au rover, comme
  toute fin du programme.

Limites :
- La « camera du dessus » est l'odometrie : le patinage contre un obstacle ne se voit pas,
  seul un pas 'stalled' du serveur compte comme choc. La derive n'est corrigee par rien.
- Apres un redemarrage du serveur, l'odometrie repart de zero : Odometrie reprend la derniere
  pose connue comme nouvelle origine (un deplacement pendant le redemarrage est perdu).
- Seule la ligne du chemin est controlee « vue libre » : les flancs du rover peuvent passer
  sur des cases pas vues en bord de champ de la camera.
- Un chemin qui part derriere le rover (hors du tour de tourelle) finit en 'bloque'.

!! Seulement essaye sur bancs (faux serveur lent, rover_server.py en simulation). Le vrai
   rover a roule avec la boucle par pas d'origine. !!
"""

import argparse
import math
import signal
import time

from fusion_carte import INCONNU, LIBRE, CarteFusion
from lancer_overhead import Suivi, arreter_le_rover, sans_post
from perception_rover import percevoir_http
from rover_reel import (
    ARRETS_OPERATEUR,
    ArretOperateur,
    RaspRoverReel,
    ServeurRoverError,
    angle_normalise,
    client_http,
    nouvelles_entrees,
    pose_odom,
    serveur_redemarre,
)

DECALAGE = 10.0  # m : le depart est en (10, 10) dans le repere de la mission (coordonnees > 0)
PANS_TOUR = (-40.0, 40.0, 0.0)  # deg (+ = a droite) : gauche, droite, et retour au centre
SEGMENT_MIN = 0.15  # m : moins a rouler sur du vu libre -> regarder autour d'abord
PREFERENCE_LIGNE = 1.15  # le couloir etroit gagne si son chemin n'est pas plus long que ca
VITESSE_PIVOT = 1.0  # rad/s, pour prevoir la duree d'un segment
ISSUES = (
    'arrive',
    'bloque',
    'perception_indisponible',
    'arret_operateur',
    'erreur_rover',
    'max_segments',
    'essai',
)
# issue -> etat du document de suivi (lancer_overhead.Suivi, page /mission)
ETATS_SUIVI = {
    'arrive': 'arrive',
    'bloque': 'abandon',
    'perception_indisponible': 'abandon',
    'arret_operateur': 'arret_operateur',
    'erreur_rover': 'erreur',
    'max_segments': 'abandon',
    'essai': 'en_cours',
}


class Odometrie:
    """Tient lieu de camera du dessus : la pose du rover dans le repere de la mission (origine
    au depart decalee de DECALAGE, x droit devant), tiree du DERNIER GET /state recu (le rover
    le lui passe par nouvel_etat : aucune requete de plus). Si le serveur redemarre (odometrie
    remise a zero), la derniere pose connue devient la nouvelle origine."""

    def __init__(self, requete, essais=5, dormir=time.sleep):
        for _ in range(essais):  # liaison lente : quelques essais avant d'abandonner
            etat = requete('GET', '/state')
            if etat is not None:
                break
            dormir(1.0)
        else:
            raise ServeurRoverError('le rover ne repond pas a GET /state')
        self.etat = etat
        self.redemarrages = 0
        self._ancrer(pose_odom(etat), (DECALAGE, DECALAGE, 0.0))

    def _ancrer(self, odo, mission):
        """L'odometrie `odo` (x, y, cap) correspond a la pose `mission` (x, y, cap)."""
        self.h0 = angle_normalise(odo[2] - mission[2])
        c, s = math.cos(self.h0), math.sin(self.h0)
        dx, dy = mission[0] - DECALAGE, mission[1] - DECALAGE
        self.x0 = odo[0] - (c * dx - s * dy)
        self.y0 = odo[1] - (s * dx + c * dy)

    def nouvel_etat(self, etat):
        if serveur_redemarre(self.etat, etat):
            self.redemarrages += 1
            self._ancrer(pose_odom(etat), self.pose())
        self.etat = etat

    def pose(self):
        x, y, h = pose_odom(self.etat)
        dx, dy = x - self.x0, y - self.y0
        c, s = math.cos(-self.h0), math.sin(-self.h0)
        return (
            DECALAGE + c * dx - s * dy,
            DECALAGE + s * dx + c * dy,
            angle_normalise(h - self.h0),
        )

    def pan_deg(self):
        """Pan de la tourelle (deg, + = a droite pour le serveur), 0 s'il n'est pas donne."""
        return float(((self.etat or {}).get('gimbal') or {}).get('pan') or 0.0)

    def disponible(self):
        return True

    def zoomer(self, tuile):
        return []


def pret(etat):
    """ESP32 connecte, gyroscope calibre, rien en cours (champs ctrl facultatifs)."""
    ctrl = etat.get('ctrl') or {}
    return (
        bool((etat.get('link') or {}).get('connected'))
        and bool((etat.get('odom') or {}).get('gyro_ready'))
        and ctrl.get('mode', 'idle') == 'idle'
        and not ctrl.get('queued')
        and ctrl.get('current') is None
        and (ctrl.get('guard') or {}).get('state') not in ('backoff', 'hold')
    )


class RoverSegments(RaspRoverReel):
    """RaspRoverReel qui roule par SEGMENTS : suivre(points) envoie plusieurs points dans UN
    POST /path. De rover_reel, n'utilise que le comportement public de RaspRoverReel et ses
    methodes _faire_un_pas (qui appelle _borner) et _arret.

    Toutes les requetes passent par _transmettre : chaque GET /state recu nourrit
    l'Odometrie ; une reponse perdue a un POST /path est verifiee par un GET /state avant
    d'etre renvoyee (renvoye, le chemin repartirait de son premier point, en arriere). Apres
    un POST /path recu, le premier dormir() dure le temps prevu du segment."""

    def __init__(self, carte, requete, odo, percevoir, dormir=time.sleep, **options):
        self.odo = odo
        self._brute, self._dormir_brut = requete, dormir
        self._prevu = 0.0  # s : duree prevue du segment en cours d'envoi
        self._a_dormir = 0.0  # s : a dormir avant le premier GET /state apres le POST /path
        self._vies = 0  # redemarrages de l'Odometrie au dernier recalage du repere
        super().__init__(
            carte,
            self._transmettre,
            odo.pose,
            percevoir,
            odo,
            rafraichir_drone=lambda: None,  # la « camera » est l'odometrie : rien a attendre
            dormir=self._dormir,
            **options,
        )

    def _transmettre(self, methode, chemin, corps=None):
        avant = self.odo.etat
        reponse = self._brute(methode, chemin, corps)
        if reponse is not None and (methode, chemin) == ('GET', '/state'):
            self.odo.nouvel_etat(reponse)
        if (methode, chemin) == ('POST', '/path'):
            if reponse is None:
                reponse = self._recu(avant)
            if reponse is not None:
                self._a_dormir = self._prevu
        return reponse

    def _recu(self, avant):
        """Reponse au POST /path perdue : un GET /state dit si le serveur l'a recu (pas en
        cours, ou nouvelle entree dans ctrl.history). None s'il ne l'a pas recu."""
        etat = self._brute('GET', '/state')
        if etat is None:
            return None
        self.odo.nouvel_etat(etat)
        ctrl, ctrl0 = etat.get('ctrl') or {}, (avant or {}).get('ctrl') or {}
        en_cours = ctrl.get('mode') in ('path', 'settle') or ctrl.get('queued')
        nouveau = (ctrl.get('history') or [None])[0] != (ctrl0.get('history') or [None])[0]
        if en_cours or ctrl.get('current') is not None or nouveau:
            return {'ok': True}
        return None

    def _dormir(self, duree):
        attente, self._a_dormir = self._a_dormir, 0.0
        self._dormir_brut(max(duree, attente))

    def drone_disponible(self):
        return True  # l'odometrie est toujours la (et sans requete)

    def suivre(self, points, duree):
        """Un segment : du rover (arrete) par `points` (repere de la mission), en UN POST /path
        dans le repere de l'odometrie, suivi jusqu'a l'arret. `duree` : temps prevu (s), dormi
        avant le premier GET /state. Renvoie (statut, impact) : statut du serveur ('done',
        'stalled', 'timeout'...), 'delai', 'redemarrage' (rien envoye) ou 'liaison' (liaison
        perdue) ; impact = point d'un choc, ou None. Leve ArretOperateur (STOP, conduite a la
        main) et ServeurRoverError (panne du serveur, segment trop long pour le repere)."""
        etat = self.odo.etat
        if (etat.get('ctrl') or {}).get('mode') == 'manual':
            raise ArretOperateur('rover conduit a la main (commande manuelle /cmd)')
        longueur = longueur_chemin([self.est, *points])
        odom = [self.repere.vers_odom(*p) for p in points]
        # Borne, comme un pas : la longueur du chemin a partir de la pose odometrique lue en
        # dernier (rover arrete) ne depasse pas le segment demande + marge_pas
        if self.odo.redemarrages == self._vies:
            reelle = longueur_chemin([pose_odom(etat)[:2], *odom])
            if reelle > longueur + self.marge_pas:
                raise ServeurRoverError(
                    f'Segment refuse : {reelle:.2f} m depuis la pose odometrique du rover pour '
                    f'{longueur:.2f} m demandes (marge {self.marge_pas:.2f} m) : repere faux.'
                )
        corps = {
            'points': [[round(x, 4), round(y, 4)] for x, y in odom],
            'frame': 'odom',
            'speed': self.vitesse,
        }
        delai = self.delai_liaison
        # le premier GET /state ne part qu'apres `duree` : ce silence n'est pas une coupure
        self.delai_liaison, self._prevu = delai + duree, duree
        try:
            statut, etat = self._faire_un_pas('/path', corps, longueur)
        finally:
            self.delai_liaison, self._prevu, self._a_dormir = delai, 0.0, 0.0
        if statut is None:
            return 'liaison', None
        if statut == 'redemarrage':
            return statut, None
        patine = self._arret(statut, etat)
        if not (patine or statut == 'stalled'):
            return statut, None
        cap = self.cap if self.cap is not None else 0.0
        return statut, (self.est[0] + 0.15 * math.cos(cap), self.est[1] + 0.15 * math.sin(cap))

    def recaler_apres(self):
        """Apres un redemarrage du serveur ou une coupure : nouveau repere sur l'odometrie."""
        self.recaler()
        self._vies = self.odo.redemarrages


def longueur_chemin(chemin):
    return sum(math.dist(chemin[i], chemin[i + 1]) for i in range(len(chemin) - 1))


def valeur(grille, p):
    """Valeur de la case fine de `grille` sous le point p ; INCONNU hors de la grille."""
    nh, nw = grille.val.shape
    i = math.floor(p[1] / grille.res) - grille.i0
    j = math.floor(p[0] / grille.res) - grille.j0
    return int(grille.val[i, j]) if 0 <= i < nh and 0 <= j < nw else INCONNU


def echantillons(chemin, pas):
    """(abscisse, point) tous les `pas` m le long de `chemin`, extremites comprises."""
    out, s = [(0.0, tuple(chemin[0]))], 0.0
    for a, b in zip(chemin, chemin[1:], strict=False):
        d = math.dist(a, b)
        n = max(1, math.ceil(d / pas))
        for k in range(1, n + 1):
            t = k / n
            out.append((s + t * d, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))))
        s += d
    return out


def couper(chemin, longueur):
    """Le debut de `chemin`, long de `longueur` m (tout le chemin s'il est plus court)."""
    out, s = [tuple(chemin[0])], 0.0
    for a, b in zip(chemin, chemin[1:], strict=False):
        d = math.dist(a, b)
        if s + d >= longueur:
            t = 0.0 if d == 0 else (longueur - s) / d
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            return out
        out.append(tuple(b))
        s += d
    return out


def troncon_vu(chemin, grille, carte, longueur_max, depart):
    """Le debut de `chemin` (du A*, depuis `depart`) que le rover peut suivre sans que son
    centre approche une case jamais vue a moins d'un rayon (rayon_effectif de la carte), au
    plus `longueur_max` m, simplifie en quelques points : chaque corde ne passe que par des
    cases vues libres et hors des obstacles gonfles. Le disque du rover au depart ne compte
    pas (la camera ne voit pas sous lui). Renvoie (points, raison) : raison 'inconnu' si une
    case pas vue a raccourci le troncon, sinon 'vu'."""
    rayon = carte.rayon_effectif
    limite, raison = longueur_max, 'vu'
    for s, p in echantillons(chemin, grille.res / 2):
        if s > longueur_max + rayon:
            break
        if math.dist(p, depart) > rayon and valeur(grille, p) != LIBRE:
            limite, raison = max(0.0, s - rayon), 'inconnu'
            break
    points = couper(chemin, limite)
    plan = carte.grille_plan(grille)

    def libre(a, b):
        for _, p in echantillons([a, b], grille.res / 2):
            if math.dist(p, depart) <= rayon:
                continue
            i, j = plan.ij(*p)
            if not plan.dedans(i, j) or plan.bloque[i, j] or valeur(grille, p) != LIBRE:
                return False
        return True

    simples, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not libre(points[i], points[j]):
            j -= 1
        simples.append(points[j])
        i = j
    return simples, raison


def planifier(carte, pos, but, depart, demi_largeur, tolerance):
    """CarteFusion.decider dans le couloir de `demi_largeur` m autour de la ligne depart ->
    but, et dans le couloir deux fois plus etroit : le chemin etroit (plus pres de la ligne)
    gagne s'il n'est pas plus long que PREFERENCE_LIGNE fois l'autre. Renvoie (decision,
    demi-largeur)."""
    x0 = min(depart[0], pos[0]) - 0.8
    x1 = max(but[0], pos[0]) + 0.8

    def decider(w):
        w = max(w, abs(pos[1] - depart[1]) + 0.3)  # le couloir contient toujours le rover
        zone = (x0, depart[1] - w, x1, depart[1] + w)
        return carte.decider(pos, but, zone=zone, tolerance=tolerance, drone_disponible=False), w

    large, w_large = decider(demi_largeur)
    if large['action'] != 'SUIVRE':
        return large, w_large
    etroit, w = decider(demi_largeur / 2)
    if w < w_large and etroit['action'] == 'SUIVRE':
        if longueur_chemin(etroit['chemin']) <= PREFERENCE_LIGNE * longueur_chemin(
            large['chemin']
        ):
            return etroit, w
    return large, w_large


def duree_prevue(chaine, cap, vitesse):
    """Temps (s) que le rover met a suivre `chaine` (depuis sa position, cap `cap`) : les
    pivots aux changements de direction, la longueur a `vitesse`, l'arret (settle)."""
    angles, direction = 0.0, cap
    for a, b in zip(chaine, chaine[1:], strict=False):
        d = math.atan2(b[1] - a[1], b[0] - a[0])
        angles += abs(angle_normalise(d - direction))
        direction = d
    return longueur_chemin(chaine) / vitesse + angles / VITESSE_PIVOT + 0.5


def conduire(
    requete,
    percevoir,
    distance=5.0,
    segment=1.0,
    vitesse=0.12,
    demi_largeur=1.5,
    rayon=0.20,
    tolerance=0.25,
    max_segments=40,
    essai=False,
    delai_liaison=15.0,
    periode=1.0,
    latence_video=0.6,
    attente_liaison=180.0,
    suivi=None,
    journal=print,
    dormir=time.sleep,
    horloge=time.monotonic,
    bilan=None,
):
    """Percevoir, planifier, rouler un segment, regarder, jusqu'au but. Renvoie l'issue (dans
    ISSUES). `bilan` : dict rempli au fur et a mesure, meme si une exception (Ctrl-C)
    interrompt la conduite : trajet, segments, tours, requetes (methode, chemin),
    par_segment (requetes de chaque segment), metres, debut, fin, suivi. `suivi` : fichier du
    document de suivi (None : pas de document)."""
    bilan = {} if bilan is None else bilan
    requetes = []
    bilan.update(debut=horloge(), fin=horloge(), trajet=[], segments=0, tours=0, metres=0.0)
    bilan.update(requetes=requetes, par_segment=[], suivi=None)

    def compter(methode, chemin, corps=None):
        requetes.append((methode, chemin))
        return requete(methode, chemin, corps)

    odo = Odometrie(compter, dormir=dormir)
    carte = CarteFusion(niveau_plan=4, rayon_robot=rayon)
    rover = RoverSegments(
        carte,
        compter,
        odo,
        percevoir,
        vitesse=vitesse,
        # liaison lente (1 a 4 s par requete) : une reponse lente n'est pas une coupure
        delai_liaison=delai_liaison,
        periode=periode,
        delai_pret=30.0,
        signaler=lambda t: journal(f'  {t}'),
        horloge=horloge,
        dormir=dormir,
    )
    depart = (DECALAGE, DECALAGE)
    but = (DECALAGE + distance, DECALAGE)
    zone = (DECALAGE - 0.8, DECALAGE - demi_largeur, but[0] + 0.8, DECALAGE + demi_largeur)
    if suivi is not None:
        suivi = Suivi(suivi, carte, rover, but, zone, journal=journal)
        journal = suivi.journal
        bilan['suivi'] = suivi
    traj = bilan['trajet']

    def regarder(pan=None):
        """Une image, la tourelle a `pan` deg (par defaut : le pan du dernier GET /state).
        True si la perception a repondu."""
        pan = odo.pan_deg() if pan is None else pan
        vue = percevoir(rover.est, angle_normalise(rover.cap - math.radians(pan)))
        if vue is None:
            return False
        rover.publier_vue(*vue)
        return True

    def regarder_encore():
        for k in range(3):
            if regarder():
                return True
            journal(f'  perception indisponible ({k + 1}/3) : le rover ne bouge pas')
            dormir(1.0)
        return False

    def tour_de_tourelle():
        """Pan -40, +40 puis 0 (+ = a droite) : une image a chaque angle, prise une fois la
        tourelle arretee (gimbal.moving, s'il existe) et --latence-video s apres."""
        bilan['tours'] += 1
        vues = 0
        for pan in PANS_TOUR:
            if rover.requete('POST', '/gimbal', {'pan': pan}) is None:
                continue  # tourelle peut-etre pas tournee : la vue serait posee de travers
            reel = pan
            if 'moving' in (odo.etat.get('gimbal') or {}):
                debut = horloge()
                while True:
                    etat = rover.requete('GET', '/state')
                    if etat is None:
                        break
                    g = etat.get('gimbal') or {}
                    if not g.get('moving') or horloge() - debut > 5.0:
                        reel = float(g.get('pan', pan))
                        break
                    dormir(0.2)
            dormir(latence_video)
            vues += regarder(reel)
        return vues

    def attendre_pret():
        debut = horloge()
        while not pret(odo.etat):
            if (odo.etat.get('ctrl') or {}).get('mode') == 'manual':
                raise ArretOperateur('rover conduit a la main (commande manuelle /cmd)')
            if horloge() - debut > 30.0:
                raise ServeurRoverError(
                    'rover pas pret apres 30 s : ESP32, gyroscope, ou pas en cours (GET /state)'
                )
            dormir(1.0)
            rover.requete('GET', '/state')

    def attendre_liaison(avant):
        """Liaison perdue pendant un segment : le serveur le finit puis s'arrete. On attend
        que la liaison revienne et que le rover soit arrete, puis on se recale."""
        journal(f'  liaison perdue : attente du rover ({attente_liaison:.0f} s au plus)')
        debut = horloge()
        while horloge() - debut < attente_liaison:
            dormir(5.0)
            etat = rover.requete('GET', '/state')
            if etat is None:
                continue
            ctrl = etat.get('ctrl') or {}
            if ctrl.get('mode') == 'manual':
                raise ArretOperateur('rover conduit a la main pendant la coupure')
            if not pret(etat):
                continue
            dernier = ((avant.get('ctrl') or {}).get('history') or [None])[0]
            nouvelles = nouvelles_entrees(ctrl.get('history') or [], dernier)
            if any(h.get('status') in ARRETS_OPERATEUR[:2] for h in nouvelles):
                raise ArretOperateur('STOP ou conduite a la main pendant la coupure')
            journal('  liaison revenue')
            rover.recaler_apres()
            return
        raise ServeurRoverError(f'liaison perdue depuis {attente_liaison:.0f} s')

    def fin(issue, texte):
        journal(texte)
        bilan['fin'] = horloge()
        if suivi is not None:
            suivi.ecrire(ETATS_SUIVI[issue])
        return issue

    try:
        rover.placer(depart)
        attendre_pret()
        rover.recaler_apres()  # repere carte <- odometrie et cap
        traj.append(depart)
        if suivi is not None:
            suivi.traj.append(list(depart))
        if essai:
            vu = regarder() or regarder_encore()  # pas de tour de tourelle : c'est un POST
        else:
            journal('depart : tour de tourelle')
            vu = tour_de_tourelle() > 0 or regarder_encore()
        if not vu:
            return fin('perception_indisponible', 'ARRET : pas de perception, rover immobile')
        ici_tour = depart
        while True:
            pos = rover.position()
            reste = math.dist(pos, but)
            if reste <= tolerance:
                return fin('arrive', f'ARRIVE : {reste:.2f} m du but')
            dec, couloir = planifier(
                carte, pos, but, depart, demi_largeur, max(0.05, tolerance - 0.05)
            )
            chemin = [tuple(p) for p in dec.get('chemin') or []]
            ouvert = dec['action'] == 'SUIVRE' and len(chemin) >= 2
            points, raison = troncon_vu(chemin or [pos], dec['grille'], carte, segment, pos)
            a_rouler, total = longueur_chemin(points), longueur_chemin(chemin)
            journal(
                f'rover ({pos[0] - DECALAGE:+.2f}, {pos[1] - DECALAGE:+.2f}) m, cap '
                f'{math.degrees(rover.cap):+.0f} deg, reste {reste:.2f} m ; {dec["action"]} '
                f'(couloir +-{couloir:.2f} m), chemin {total:.2f} m, {a_rouler:.2f} m vus '
                f'libres devant ({raison})'
            )
            if suivi is not None:
                suivi.decision = {'action': dec['action'], 'raison': f'segment ({raison})'}
                suivi.chemin = [[float(x), float(y)] for x, y in chemin]
                suivi.ecrire()
            pres = total < SEGMENT_MIN and reste <= tolerance + 0.1
            if ouvert and pres:  # le A* s'arrete a sa tolerance pres, case de 5 cm comprise
                return fin('arrive', f'ARRIVE : {reste:.2f} m du but')
            regarde_ici = math.dist(ici_tour, pos) < 0.1
            if not (ouvert and raison == 'vu' and a_rouler >= SEGMENT_MIN) and not regarde_ici:
                if essai:
                    journal('[essai] ferait un tour de tourelle ici (aucun POST)')
                else:
                    journal('  tour de tourelle : chemin ferme ou cases pas vues devant')
                    tour_de_tourelle()
                    ici_tour = pos
                    continue
            if not ouvert:
                return fin('bloque', f'BLOQUE : plus aucun passage vers le but ({reste:.2f} m)')
            if a_rouler < SEGMENT_MIN and (raison == 'inconnu' or a_rouler < 0.05):
                return fin('bloque', f'BLOQUE : rien de vu libre devant ({a_rouler:.2f} m)')
            if essai:
                cible = points[-1]
                return fin(
                    'essai',
                    f'[essai] premier segment de {a_rouler:.2f} m vers ({cible[0] - DECALAGE:+.2f}'
                    f', {cible[1] - DECALAGE:+.2f}) m en {len(points) - 1} points : rien envoye',
                )
            if bilan['segments'] >= max_segments:
                return fin('max_segments', f'ARRET : {max_segments} segments sans arriver')
            attendre_pret()
            avant, n = odo.etat, len(requetes)
            duree = duree_prevue(points, rover.cap, vitesse)
            statut, impact = rover.suivre(points[1:], duree)
            bilan['segments'] += 1
            bilan['par_segment'].append(len(requetes) - n)
            journal(
                f'  segment {bilan["segments"]} : {a_rouler:.2f} m prevus en {duree:.0f} s, '
                f'{statut}, {len(requetes) - n} requetes'
            )
            if statut == 'liaison':
                attendre_liaison(avant)
            elif statut == 'redemarrage':
                rover.recaler_apres()
                attendre_pret()
            nouvelle = tuple(rover.position())
            bilan['metres'] += math.dist(traj[-1], nouvelle)
            traj.append(nouvelle)
            if suivi is not None:
                suivi.traj.append([float(c) for c in nouvelle])
            if impact is not None:
                journal(f'  CHOC vers ({impact[0] - DECALAGE:+.2f}, {impact[1] - DECALAGE:+.2f})')
                rover.signaler_choc(impact)
            if not regarder_encore():
                return fin('perception_indisponible', 'ARRET : plus de perception')
    except ArretOperateur as e:
        return fin('arret_operateur', f'ARRET OPERATEUR : {e}')
    except ServeurRoverError as e:
        return fin('erreur_rover', f'ERREUR DU ROVER : {e}')


def resumer(issue, bilan, journal=print):
    """Le trajet parcouru (repere du depart : m, x devant, y a gauche) et le bilan."""
    traj = bilan.get('trajet') or []
    if traj:
        points = ' '.join(f'({x - DECALAGE:+.2f},{y - DECALAGE:+.2f})' for x, y in traj)
        journal(f'Trajet : {points}')
    metres = bilan.get('metres', 0.0)
    duree = bilan.get('fin', 0.0) - bilan.get('debut', 0.0)
    ecart = max((abs(y - DECALAGE) for _, y in traj), default=0.0)
    par_metre = f'{duree / metres:.1f} s/m' if metres > 0.01 else '- s/m'
    journal(
        f'Fin : {issue} ; {metres:.2f} m roules en {bilan.get("segments", 0)} segments, '
        f'{bilan.get("tours", 0)} tours de tourelle, {len(bilan.get("requetes", []))} requetes '
        f'HTTP au rover, {duree:.0f} s ({par_metre}), ecart maximal a la ligne {ecart:.2f} m'
    )


def main(argv=None, percevoir=None):
    """`percevoir` : pour les tests (par defaut, le service de perception --perception)."""
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--rover', default='http://127.0.0.1:18080')
    p.add_argument('--perception', default='http://127.0.0.1:8095')
    p.add_argument('--distance', type=float, default=5.0, help='m droit devant')
    p.add_argument('--segment', type=float, default=1.0, help='m au plus par POST /path')
    p.add_argument('--vitesse', type=float, default=0.12, help='m/s')
    p.add_argument('--demi-largeur', type=float, default=1.5, help='m de chaque cote de la ligne')
    p.add_argument('--rayon', type=float, default=0.20, help='rayon du rover + marge (m)')
    p.add_argument('--tolerance', type=float, default=0.25, help='arrivee a moins de (m)')
    p.add_argument('--max-segments', type=int, default=40)
    p.add_argument('--delai-http', type=float, default=8.0, help='s par requete au rover')
    p.add_argument('--delai-liaison', type=float, default=15.0, help='s sans reponse = coupure')
    p.add_argument('--periode', type=float, default=1.0, help='s entre deux GET /state')
    p.add_argument('--latence-video', type=float, default=0.6, help='s apres la tourelle')
    p.add_argument('--attente-liaison', type=float, default=180.0, help='s apres une coupure')
    p.add_argument('--suivi', help='document JSON de suivi (page /mission)')
    p.add_argument('--essai', action='store_true', help='perception et plan seulement, aucun POST')
    args = p.parse_args(argv)

    def _interrompre(signum, frame):  # kill / fin de session : arreter le rover comme Ctrl-C
        raise KeyboardInterrupt

    anciens = {s: signal.signal(s, _interrompre) for s in (signal.SIGTERM, signal.SIGHUP)}
    requete = client_http(args.rover, delai=args.delai_http)
    if args.essai:
        requete = sans_post(requete)
    if percevoir is None:
        percevoir = percevoir_http(args.perception, delai=args.delai_http, age_max=3.0)
    issue, bilan = 'erreur_rover', {}
    try:
        issue = conduire(
            requete,
            percevoir,
            distance=args.distance,
            segment=args.segment,
            vitesse=args.vitesse,
            demi_largeur=args.demi_largeur,
            rayon=args.rayon,
            tolerance=args.tolerance,
            max_segments=args.max_segments,
            essai=args.essai,
            delai_liaison=args.delai_liaison,
            periode=args.periode,
            latence_video=args.latence_video,
            attente_liaison=args.attente_liaison,
            suivi=args.suivi,
            bilan=bilan,
        )
    except KeyboardInterrupt:
        issue, bilan['fin'] = 'arret_operateur', time.monotonic()
        print('Interrompu (Ctrl-C, SIGTERM ou SIGHUP)')
    except (ServeurRoverError, RuntimeError) as e:  # Odometrie sans reponse, --essai...
        bilan['fin'] = time.monotonic()
        print(f'ERREUR : {e}')
    finally:
        if not args.essai:
            # un 2e Ctrl-C, SIGTERM ou SIGHUP n'interrompt pas le /stop
            ignores = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
            avant = {s: signal.signal(s, signal.SIG_IGN) for s in ignores}
            try:
                arreter_le_rover(client_http(args.rover, delai=args.delai_http))
            finally:
                for s, h in avant.items():
                    signal.signal(s, h)
        for s, h in anciens.items():
            signal.signal(s, h)
    if bilan.get('suivi') is not None and issue != 'essai':
        bilan['suivi'].ecrire(ETATS_SUIVI.get(issue, 'erreur'))
    if bilan:
        resumer(issue, bilan)
    return issue


if __name__ == '__main__':
    main()
