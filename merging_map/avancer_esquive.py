"""
avancer_esquive.py - Le rover avance de DISTANCE m droit devant lui en contournant ce que sa
propre camera voit, sans camera du dessus. Version propre du petit pilote qui a conduit le vrai
rover le 15 novembre (20 pas, environ 3,5 m de chemin, arrete par l'operateur a 2,1 m du but).

- Position : l'odometrie du rover (GET /state), tenue pour la camera du dessus par un
  pseudo-drone (Odometrie) ; repere de la mission : origine au depart decalee de DECALAGE m,
  x droit devant au depart, y a gauche.
- Obstacles : le service de perception (rover_mapper/perception_service.py) par
  perception_rover.percevoir_http, avec le vrai pan de la tourelle lu dans GET /state.
- Carte et chemin : CarteFusion.decider(drone_disponible=False), A* sur des cases de 5 cm.
- Conduite : rover_reel.RaspRoverReel (pas bornes, STOP operateur definitif, redemarrage du
  serveur detecte).

    python avancer_esquive.py --rover http://127.0.0.1:18080 \\
        --perception http://127.0.0.1:8095 --distance 5 --essai   # plan seulement, AUCUN POST
    python avancer_esquive.py ... --distance 5    # pour de vrai, quelqu'un pret au STOP

Ce que le vrai essai a montre, et ce qui change ici :
- Le but reste « DISTANCE m devant, sur la ligne de depart », mais le A* cherche d'abord un
  chemin dans un couloir etroit autour de la ligne (demi-largeur 0,5 m, puis 1 m, puis
  --demi-largeur) : le rover ne s'ecarte que si le couloir etroit est ferme.
- Plan en OBJECTIF_DEGRADE : avant d'abandonner ou de suivre un detour, le rover (sans
  tourelle qui tourne) pivote sur place pour regarder a gauche puis a droite. RaspRoverReel n'a
  pas de pas /turn : on fait un deplacer() vers un point a ECART_REGARD m sur le cote (le
  serveur pivote vers lui puis avance de quelques cm). Le suiveur du serveur ne pivote que vers
  un point devant lui : on ne regarde donc jamais a plus de ANGLE_REGARD_MAX du cap.
- Avant un pas qui part a plus de ANGLE_AVEUGLE du cap et traverse des cases jamais vues, le
  rover se tourne d'abord vers lui pour les voir.
- Les cases jamais vues ne sont pas libres : le journal dit a chaque pas quelle longueur du
  chemin les traverse (le A* les accepte, a un cout triple).
- Fin claire : arrive, bloque, perception_indisponible, arret_operateur, max_pas (ou essai), et
  le trajet parcouru est imprime.

Limites :
- La « camera du dessus » est l'odometrie : le patinage contre un obstacle (roues qui tournent,
  rover immobile) ne se voit pas, seul un pas 'stalled' du serveur compte comme choc. La
  derive de l'odometrie n'est corrigee par rien.
- Apres un redemarrage du serveur, l'odometrie repart de zero : Odometrie reprend la derniere
  pose connue comme nouvelle origine (un deplacement pendant le redemarrage est perdu).

Ctrl-C, SIGTERM et SIGHUP arretent le rover (/stop), comme toute fin du programme.

!! Cette version n'a tourne que sur le banc (rover_server.py en simulation) ; le vrai rover a
   roule avec la version d'origine, sans les regards ni le couloir. !!
"""

import argparse
import math
import signal
import time

from cerveau import avancer
from fusion_carte import INCONNU, CarteFusion
from lancer_overhead import arreter_le_rover, sans_post
from perception_rover import percevoir_http
from rover_reel import (
    ArretOperateur,
    RaspRoverReel,
    ServeurRoverError,
    angle_normalise,
    client_http,
    pose_odom,
    serveur_redemarre,
)

DECALAGE = 10.0  # m : le depart est en (10, 10) dans le repere de la mission (coordonnees > 0)
COULOIRS = (0.5, 1.0)  # m : demi-largeurs essayees avant --demi-largeur
ECART_REGARD = 0.04  # m : pas d'un regard (le serveur finit un /path a 2 cm pres)
ANGLE_REGARD = math.radians(60.0)  # regard a gauche puis a droite
ANGLE_REGARD_MAX = math.radians(70.0)  # au-dela, le suiveur du serveur ne pivote plus
ANGLE_AVEUGLE = math.radians(35.0)  # pas trop loin du cap de la camera : se tourner d'abord
ISSUES = ('arrive', 'bloque', 'perception_indisponible', 'arret_operateur', 'max_pas', 'essai')


class Odometrie:
    """Tient lieu de camera du dessus : la pose du rover dans le repere de la mission (origine
    au depart decalee de DECALAGE, x droit devant), tiree de GET /state. Garde le dernier etat
    lu (pan de la tourelle). Si le serveur redemarre (odometrie remise a zero), la derniere
    pose connue devient la nouvelle origine."""

    def __init__(self, requete, essais=5):
        self.requete = requete
        self.etat = None
        self.derniere = (DECALAGE, DECALAGE, 0.0)
        self.redemarrages = 0
        for _ in range(essais):  # liaison lente : quelques essais avant d'abandonner
            etat = self.requete('GET', '/state')
            if etat is not None:
                break
        else:
            raise RuntimeError('le rover ne repond pas a GET /state')
        self.etat = etat
        self._ancrer(pose_odom(etat), self.derniere)

    def _ancrer(self, odo, mission):
        """L'odometrie `odo` (x, y, cap) correspond a la pose `mission` (x, y, cap)."""
        self.h0 = angle_normalise(odo[2] - mission[2])
        c, s = math.cos(self.h0), math.sin(self.h0)
        dx, dy = mission[0] - DECALAGE, mission[1] - DECALAGE
        self.x0 = odo[0] - (c * dx - s * dy)
        self.y0 = odo[1] - (s * dx + c * dy)

    def pose(self):
        etat = self.requete('GET', '/state')
        if etat is None:
            return None
        if self.etat is not None and serveur_redemarre(self.etat, etat):
            self.redemarrages += 1
            self._ancrer(pose_odom(etat), self.derniere)
        self.etat = etat
        x, y, h = pose_odom(etat)
        dx, dy = x - self.x0, y - self.y0
        c, s = math.cos(-self.h0), math.sin(-self.h0)
        self.derniere = (
            DECALAGE + c * dx - s * dy,
            DECALAGE + s * dx + c * dy,
            angle_normalise(h - self.h0),
        )
        return self.derniere

    def pan_deg(self):
        """Pan de la tourelle (deg, + = a droite pour le serveur), 0 s'il n'est pas donne."""
        return float(((self.etat or {}).get('gimbal') or {}).get('pan') or 0.0)

    def disponible(self):
        return True

    def zoomer(self, tuile):
        return []


def longueur(chemin):
    return sum(math.dist(chemin[i], chemin[i + 1]) for i in range(len(chemin) - 1))


def part_inconnue(grille, chemin):
    """Longueur (m) de `chemin` qui traverse des cases jamais vues de `grille` (niveau fin) ;
    une case hors de la grille compte comme jamais vue."""
    if grille is None:
        return longueur(chemin)
    nh, nw = grille.val.shape
    pas = grille.res / 2
    total = 0.0
    for a, b in zip(chemin, chemin[1:], strict=False):
        d = math.dist(a, b)
        n = max(1, math.ceil(d / pas))
        for k in range(n):
            t = (k + 0.5) / n
            x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            i = math.floor(y / grille.res) - grille.i0
            j = math.floor(x / grille.res) - grille.j0
            if not (0 <= i < nh and 0 <= j < nw) or grille.val[i, j] == INCONNU:
                total += d / n
    return total


def planifier(carte, pos, but, depart, demi_largeur, tolerance):
    """CarteFusion.decider dans des couloirs de plus en plus larges autour de la ligne
    depart -> but : le premier qui donne un chemin gagne. Renvoie (decision, demi-largeur)."""
    ecart = abs(pos[1] - depart[1]) + 0.3  # le couloir contient toujours le rover
    x0 = min(depart[0], pos[0]) - 0.8
    x1 = max(but[0], pos[0]) + 0.8
    largeurs = sorted({max(w, ecart) for w in COULOIRS if w < demi_largeur} | {demi_largeur})
    dec = None
    for w in largeurs:
        zone = (x0, depart[1] - w, x1, depart[1] + w)
        dec = carte.decider(pos, but, zone=zone, tolerance=tolerance, drone_disponible=False)
        if dec['action'] in ('SUIVRE', 'INSPECTER'):
            return dec, w
    return dec, largeurs[-1]


def conduire(
    requete,
    percevoir,
    distance=5.0,
    pas=0.25,
    vitesse=0.12,
    demi_largeur=2.0,
    rayon=0.20,
    tolerance=0.25,
    max_pas=60,
    essai=False,
    delai_liaison=10.0,
    periode=0.5,
    journal=print,
    dormir=time.sleep,
    trajet=None,
):
    """La boucle : percevoir (avec le vrai pan de la camera), planifier vers le but, faire un pas
    ou un regard. Renvoie (issue, trajectoire) ; issue est dans ISSUES. Les erreurs du serveur
    (ServeurRoverError) et une perte de liaison remontent. `trajet` : liste a remplir (les
    points restent a l'appelant meme si une exception interrompt la boucle)."""
    odo = Odometrie(requete)
    carte = CarteFusion(niveau_plan=4, rayon_robot=rayon)
    rover = RaspRoverReel(
        carte,
        requete,
        position_drone=odo.pose,
        percevoir=percevoir,
        drone=odo,
        rafraichir_drone=lambda: None,  # la « camera » est l'odometrie : pas d'image a attendre
        vitesse=vitesse,
        # liaison VM -> robot lente (1 a 4 s par requete) : une reponse lente n'est pas une coupure
        delai_liaison=delai_liaison,
        periode=periode,
        delai_pret=30.0,
        signaler=lambda t: journal(f'  {t}'),
        dormir=dormir,
    )
    depart = (DECALAGE, DECALAGE)
    but = (DECALAGE + distance, DECALAGE)
    rover.placer(depart)
    rover.recaler()  # repere carte <- odometrie et cap, depuis la pose d'odometrie du depart
    if rover.cap is None:
        raise RuntimeError('cap du rover inconnu apres le recalage : odometrie illisible ?')
    traj = trajet if trajet is not None else []
    traj.append(depart)
    sans_vue = 0
    regards = []  # angles (rad, relatifs au cap) des regards qui restent a faire
    regarde_ici = None  # position du dernier tour de regards

    def fin(issue, texte):
        journal(texte)
        return issue, traj

    def bouger(cible):
        """Un pas (ou un regard) vers `cible` ; False si un operateur a arrete le rover."""
        try:
            impact = rover.deplacer(cible)
        except ArretOperateur as e:
            journal(f'ARRET OPERATEUR : {e}')
            return False
        if not rover.rover_joignable():
            raise ServeurRoverError(f'rover injoignable depuis {delai_liaison:.0f} s')
        traj.append(tuple(rover.position()))
        if impact is not None:
            journal(f'  CHOC vers ({impact[0] - DECALAGE:+.2f}, {impact[1] - DECALAGE:+.2f}) m')
            rover.signaler_choc(impact)  # comme cerveau.mission : obstacle sur la carte
        return True

    for k in range(max_pas):
        pos = rover.position()
        reste = math.dist(pos, but)
        if reste <= tolerance:
            return fin('arrive', f'ARRIVE : {reste:.2f} m du but apres {k} pas')
        pan = odo.pan_deg()
        cap = None if rover.cap is None else angle_normalise(rover.cap - math.radians(pan))
        vue = percevoir(rover.est, cap)
        if vue is None:
            sans_vue += 1
            journal(f'  perception indisponible ({sans_vue}/3) : on ne bouge pas')
            if sans_vue >= 3:
                return fin('perception_indisponible', 'ARRET : pas de perception')
            dormir(1.0)
            continue
        sans_vue = 0
        rover.publier_vue(*vue)
        dec, couloir = planifier(carte, pos, but, depart, demi_largeur, tolerance)
        chemin = [tuple(p) for p in dec.get('chemin') or []]
        aveugle = part_inconnue(dec.get('grille'), chemin)
        journal(
            f'pas {k + 1} : rover ({pos[0] - DECALAGE:+.2f}, {pos[1] - DECALAGE:+.2f}) m, '
            f'cap {math.degrees(rover.cap or 0):+.0f} deg, pan {pan:+.1f} deg, '
            f'reste {reste:.2f} m ; {dec["action"]} (couloir +-{couloir:.1f} m), chemin '
            f'{longueur(chemin):.2f} m dont {aveugle:.2f} m en cases jamais vues'
        )
        if dec['action'] == 'OBJECTIF_DEGRADE':
            if not regards and (regarde_ici is None or math.dist(regarde_ici, pos) > 0.3):
                # gauche, retour au cap, droite : un regard n'a pas le droit de depasser
                # ANGLE_REGARD_MAX (le suiveur du serveur ne pivote que vers l'avant)
                regards = [ANGLE_REGARD, -ANGLE_REGARD, -ANGLE_REGARD]
                regarde_ici = pos
            if regards:
                angle = regards.pop(0)
                if essai:
                    return fin('essai', '[essai] plan degrade : regarderait autour, rien envoye')
                journal(f'  regard a {math.degrees(angle):+.0f} deg avant d\'abandonner')
                a = rover.cap + angle
                cote = (pos[0] + ECART_REGARD * math.cos(a), pos[1] + ECART_REGARD * math.sin(a))
                if not bouger(cote):
                    return fin('arret_operateur', 'ARRET : STOP operateur')
                continue
            point = dec.get('point', pos)
            if math.dist(point, but) > reste - 0.1 or len(chemin) < 2:
                return fin('bloque', f'BLOQUE : plus aucun passage vers le but ({reste:.2f} m)')
            journal(f'  detour vers ({point[0] - DECALAGE:+.2f}, {point[1] - DECALAGE:+.2f}) m')
        else:
            regards = []
        if len(chemin) < 2:
            if reste <= tolerance + 0.1:  # le A* arrive a la tolerance pres d'une case de 5 cm
                return fin('arrive', f'ARRIVE : {reste:.2f} m du but apres {k} pas')
            return fin('bloque', 'BLOQUE : pas de chemin')
        cible, _ = avancer(pos, list(chemin), pas)
        direction = math.atan2(cible[1] - pos[1], cible[0] - pos[0])
        ecart_cap = angle_normalise(direction - cap)
        if essai:
            return fin(
                'essai',
                f'[essai] prochain pas vers ({cible[0] - DECALAGE:+.2f}, '
                f'{cible[1] - DECALAGE:+.2f}) m : rien envoye',
            )
        if abs(ecart_cap) > ANGLE_AVEUGLE and part_inconnue(dec.get('grille'), [pos, cible]) > 0:
            # le pas part hors de la vue de la camera, dans des cases jamais vues : se tourner
            # d'abord vers lui (quelques cm), il sera vu au tour suivant
            a = rover.cap + max(-ANGLE_REGARD_MAX, min(ANGLE_REGARD_MAX, ecart_cap))
            journal(f'  pas hors de vue ({math.degrees(ecart_cap):+.0f} deg) : regard d\'abord')
            cible = (pos[0] + ECART_REGARD * math.cos(a), pos[1] + ECART_REGARD * math.sin(a))
        if not bouger(cible):
            return fin('arret_operateur', 'ARRET : STOP operateur')
    return fin('max_pas', f'ARRET : {max_pas} pas sans arriver')


def resumer(issue, traj, journal=print):
    """Le trajet parcouru, dans le repere du depart (m, x devant, y a gauche)."""
    points = ' '.join(f'({x - DECALAGE:+.2f},{y - DECALAGE:+.2f})' for x, y in traj)
    ecart = max(abs(y - DECALAGE) for _, y in traj)
    journal(
        f'Fin : {issue}, {len(traj) - 1} pas, {longueur(traj):.2f} m parcourus, '
        f'ecart maximal a la ligne {ecart:.2f} m'
    )
    journal(f'Trajet : {points}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--rover', default='http://127.0.0.1:18080')
    p.add_argument('--perception', default='http://127.0.0.1:8095')
    p.add_argument('--distance', type=float, default=5.0, help='m droit devant')
    p.add_argument('--pas', type=float, default=0.25, help='m par pas')
    p.add_argument('--vitesse', type=float, default=0.12, help='m/s')
    p.add_argument('--demi-largeur', type=float, default=2.0, help='m de chaque cote de la ligne')
    p.add_argument('--rayon', type=float, default=0.20, help='rayon du rover + marge (m)')
    p.add_argument('--tolerance', type=float, default=0.25, help='arrivee a moins de (m)')
    p.add_argument('--max-pas', type=int, default=60)
    p.add_argument('--delai-http', type=float, default=6.0, help='s par requete au rover')
    p.add_argument('--delai-liaison', type=float, default=10.0, help='s sans reponse = coupure')
    p.add_argument('--essai', action='store_true', help='perception et plan seulement, aucun POST')
    args = p.parse_args(argv)

    def _interrompre(signum, frame):  # kill / fin de session : arreter le rover comme Ctrl-C
        raise KeyboardInterrupt

    anciens = {s: signal.signal(s, _interrompre) for s in (signal.SIGTERM, signal.SIGHUP)}
    requete = client_http(args.rover, delai=args.delai_http)
    if args.essai:
        requete = sans_post(requete)
    percevoir = percevoir_http(args.perception, delai=6.0, age_max=3.0)
    issue, traj = 'erreur', []
    try:
        issue, _ = conduire(
            requete,
            percevoir,
            distance=args.distance,
            pas=args.pas,
            vitesse=args.vitesse,
            demi_largeur=args.demi_largeur,
            rayon=args.rayon,
            tolerance=args.tolerance,
            max_pas=args.max_pas,
            essai=args.essai,
            delai_liaison=args.delai_liaison,
            trajet=traj,
        )
    except ArretOperateur as e:
        issue = 'arret_operateur'
        print(f'ARRET OPERATEUR : {e}')
    except KeyboardInterrupt:
        print('Interrompu (Ctrl-C, SIGTERM ou SIGHUP)')
    except (ServeurRoverError, RuntimeError) as e:
        print(f'ERREUR : {e}')
    finally:
        if not args.essai:
            ancien_int = signal.signal(signal.SIGINT, signal.SIG_IGN)  # un 2e Ctrl-C attend
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            arreter_le_rover(client_http(args.rover, delai=args.delai_http))
            signal.signal(signal.SIGINT, ancien_int)
        for s, h in anciens.items():
            signal.signal(s, h)
    if traj:
        resumer(issue, traj)
    return issue


if __name__ == '__main__':
    main()
