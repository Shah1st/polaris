"""
demo_fusion.py - Demo de fusion drone + rover sur un monde simule.

Scenario 1 : une crete barre le chemin. Vue de haut (cases de 80 cm) le drone ne voit
             pas le passage de 70 cm -> le rover demande des zooms jusqu'a le trouver,
             puis decouvre en roulant un tronc d'arbre que le drone n'avait pas vu.
Scenario 2 : la victime est au fond d'un cratere ferme -> apres verification par le
             drone, le rover va au point accessible le plus proche.

Lancer :  python demo_fusion.py      (produit demo_fusion.png)
"""

import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fusion_carte import INCONNU, LIBRE, OBSTACLE, SRC_DRONE, SRC_ROVER, CarteFusion, decouper
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ZONE_M = 25.6
RES = 0.05
NC = int(round(ZONE_M / RES))  # 512 cases fines de cote
ZONE = (0.0, 0.0, ZONE_M, ZONE_M)
PORTEE_ROVER = 2.0  # portee de la camera du rover (m)
PAS = 0.3  # avance du rover par tour (m), liaison drone OK
PAS_PRUDENT = 0.15  # sans drone : deux fois plus lent
INCERTITUDE_PAR_METRE = 0.025  # sans drone, l'erreur de position possible : +2,5 cm/m
BUDGET_INCERTITUDE = 0.6  # au-dela (m), retour au dernier point vu par le drone
ATTENTE_MAX = 300.0  # temps d'attente du drone au point de retour (s), puis arret


# ----------------------------------------------------------------------------
# Monde simule (verite terrain, 5 cm)
# ----------------------------------------------------------------------------
def idx(v):
    return int(round(v / RES))


def idx_floor(v):
    return int(math.floor(v / RES))


def rect(m, x0, y0, x1, y1, v=OBSTACLE):
    m[idx(y0) : idx(y1), idx(x0) : idx(x1)] = v


def disque(m, cx, cy, r, v=OBSTACLE):
    yy, xx = np.mgrid[0:NC, 0:NC]
    m[((xx + 0.5) * RES - cx) ** 2 + ((yy + 0.5) * RES - cy) ** 2 <= r * r] = v


class Simu:
    def __init__(
        self,
        carte,
        verite,
        vu_par_drone,
        inconnu_drone=None,
        occlusion=True,
        champ=360,
        coupures=(),
        graine=0,
    ):
        self.carte, self.verite, self.vu_drone = carte, verite, vu_par_drone
        self.inconnu_drone = inconnu_drone  # zone que le drone n'a pas pu voir au 1er survol
        self.occlusion = occlusion  # False = le rover voit a travers les obstacles (irrealiste)
        self.champ = champ  # angle de vue de la camera, en degres, centre sur le cap du rover
        self.cap = math.pi / 2  # direction (estimee) du rover (radians, 0 = est, pi/2 = nord)
        self.t = 0.0
        # coupures de la liaison ordinateur <-> drone : [(debut, fin)] en s, fin=None = pour
        # toujours. Sans drone, la position du rover vient de l'odometrie, qui derive : le cap
        # derive (biais + bruit) et les roues glissent un peu.
        self.coupures = list(coupures)
        self.rng = np.random.default_rng(graine)
        self.biais_cap = 0.0015  # rad par metre parcouru (gyroscope calibre : ~2 deg sur 20 m)
        self.bruit_cap = 0.005  # rad par racine de metre
        self.glissement = 0.015  # le rover parcourt 1,5 % de plus que ce que mesurent ses roues
        self.placer((0.0, 0.0))

    # ---- liaison avec le drone et position du rover ---------------------------
    def connecte(self):
        """La liaison ordinateur <-> drone fonctionne-t-elle a l'instant t ?"""
        return not any(
            debut <= self.t < (math.inf if fin is None else fin) for debut, fin in self.coupures
        )

    def placer(self, pos):
        """Met le rover en `pos`, position connue (le drone vient de le voir)."""
        self.est = self.vraie = pos  # position estimee (celle de la carte) et reelle
        self.sigma, self.erreur_cap = 0.0, 0.0  # incertitude (m), erreur de cap (rad)
        self.dernier_fix, self.trace_fix = pos, [pos]
        self.etait_connecte = True
        self.dist_fix = 0.0  # distance parcourue depuis le dernier recalage par le drone
        # couche provisoire : observations faites sans drone, a reprojeter au recalage
        self.instantane, self.provisoires = None, []

    def deplacement_reel(self, avant_est, apres_est, avant_vraie):
        """Le rover execute le deplacement commande. Avec le drone, sa position est recalee
        a chaque pas : le reel suit l'estime. Sans drone, le cap derive et les roues glissent."""
        dx, dy = apres_est[0] - avant_est[0], apres_est[1] - avant_est[1]
        if self.connecte():
            return avant_vraie[0] + dx, avant_vraie[1] + dy
        longueur = math.hypot(dx, dy)
        self.erreur_cap += self.biais_cap * longueur + self.rng.normal(
            0.0, self.bruit_cap * math.sqrt(longueur)
        )
        c, s, k = math.cos(self.erreur_cap), math.sin(self.erreur_cap), 1 + self.glissement
        return avant_vraie[0] + k * (c * dx - s * dy), avant_vraie[1] + k * (s * dx + c * dy)

    def maj_pose(self, est, vraie, cap):
        longueur = math.dist(self.est, est)
        self.est, self.vraie, self.cap = est, vraie, cap
        if self.connecte():
            self.recaler()
        else:
            self.sigma += INCERTITUDE_PAR_METRE * longueur
            self.dist_fix += longueur
            self.trace_fix.append(est)

    def recaler(self):
        """Le drone voit le marqueur ArUco du rover : la position estimee devient la vraie.
        Les observations faites pendant la coupure sont reprojetees (voir reprojeter)."""
        self.reprojeter()
        self.est = self.vraie
        self.sigma, self.erreur_cap, self.dist_fix = 0.0, 0.0, 0.0
        self.dernier_fix, self.trace_fix = self.vraie, [self.vraie]

    def reprojeter(self):
        """Au recalage, on connait l'erreur de position finale E. Elle a grandi a peu pres
        avec la distance parcourue : une observation faite apres une distance d est decalee
        de E * d / D. On annule la couche provisoire puis on la reecrit corrigee, ce qui
        supprime les obstacles fantomes ecrits au mauvais endroit."""
        if self.instantane is None:
            return
        ex, ey = self.vraie[0] - self.est[0], self.vraie[1] - self.est[1]
        total = max(self.dist_fix, 1e-6)
        self.carte.restaurer_rover(self.instantane)
        for genre, contenu, d in self.provisoires:
            f = min(1.0, d / total)
            if genre == 'vue':
                loc, x0, y0 = contenu
                for msg in decouper(
                    loc, x0 + ex * f, y0 + ey * f, self.carte.zmax, self.carte, 'rover', self.t
                ):
                    self.carte.recevoir(msg)
            else:
                self.carte.signaler_obstacle(contenu[0] + ex * f, contenu[1] + ey * f, 0.1, self.t)
        self.instantane, self.provisoires = None, []

    def provisoire(self, genre, contenu):
        """Note une observation faite sans drone (genre 'vue' ou 'choc')."""
        self.provisoires.append((genre, contenu, self.dist_fix))

    def observer(self, tour_complet=False):
        """La camera voit le terrain reel autour de la position REELLE, mais le rover ecrit
        ses tuiles la ou il CROIT etre : sans drone, elles sont decalees de son erreur.
        tour_complet : tour d'horizon avec la tourelle pan-tilt (360 degres)."""
        decalage = (self.est[0] - self.vraie[0], self.est[1] - self.vraie[1])
        champ = 360 if tour_complet else self.champ
        if not self.connecte() and self.instantane is None:
            self.instantane = self.carte.instantane_rover()
        vue = self._voir(*self.vraie, self.cap + self.erreur_cap, champ, decalage)
        if not self.connecte():
            self.provisoire('vue', vue)

    def vue_drone(self, z):
        f = 2 ** (self.carte.zmax - z)
        n = NC // f
        return self.vu_drone.reshape(n, f, n, f).max(axis=(1, 3))

    def survol_initial(self, z=0):
        m = self.vue_drone(z).copy()
        if self.inconnu_drone:
            x0, y0, x1, y1 = self.inconnu_drone
            r = self.carte.res(z)
            m[int(y0 / r) : int(y1 / r), int(x0 / r) : int(x1 / r)] = INCONNU
        for msg in decouper(m, 0.0, 0.0, z, self.carte, 'drone', self.t):
            self.carte.recevoir(msg)

    def zoom_drone(self, cle):
        z, tx, ty = cle
        m = self.vue_drone(z)
        n = self.carte.n
        if not (0 <= ty * n < m.shape[0] and 0 <= tx * n < m.shape[1]):
            return
        self.carte.recevoir(
            {
                'source': 'drone',
                't': self.t,
                'z': z,
                'x': tx,
                'y': ty,
                'data': m[ty * n : (ty + 1) * n, tx * n : (tx + 1) * n],
            }
        )

    def capteur_rover(self, x, y, cap=None):
        """Ce que voit le rover : PORTEE_ROVER autour de lui, dans un cone de `champ` degres
        centre sur son cap. Avec occlusion, une case cachee derriere un obstacle est envoyee
        en -1 (pas vue)."""
        if cap is not None:
            self.cap = cap
        self._voir(x, y, self.cap, self.champ, (0.0, 0.0))

    def _voir(self, x, y, cap, champ, decalage):
        i0, i1 = max(0, idx(y - PORTEE_ROVER)), min(NC, idx(y + PORTEE_ROVER))
        j0, j1 = max(0, idx(x - PORTEE_ROVER)), min(NC, idx(x + PORTEE_ROVER))
        loc = self.verite[i0:i1, j0:j1].copy()
        yy, xx = np.mgrid[i0:i1, j0:j1]
        hors = ((xx + 0.5) * RES - x) ** 2 + ((yy + 0.5) * RES - y) ** 2 > PORTEE_ROVER**2
        if self.occlusion or champ < 360:
            hors |= ~self.visibles(x, y, i0, i1, j0, j1, cap, champ)
        loc[hors] = INCONNU
        x0, y0 = j0 * RES + decalage[0], i0 * RES + decalage[1]
        for msg in decouper(loc, x0, y0, self.carte.zmax, self.carte, 'rover', self.t):
            self.carte.recevoir(msg)
        return loc, x0, y0

    def visibles(self, x, y, i0, i1, j0, j1, cap=None, champ=None, n_rayons=1440):
        """Lancer de rayons depuis le rover, comme un LiDAR virtuel : sur chaque rayon, les
        cases sont vues jusqu'au premier obstacle inclus (sa face avant), et cachees au-dela.
        Une face vue en biais reste donc visible. Seuls les rayons du cone de vue comptent."""
        cap = self.cap if cap is None else cap
        champ_deg = self.champ if champ is None else champ
        pas = RES / 4
        r = np.arange(1, int(PORTEE_ROVER / pas) + 1) * pas
        champ = math.radians(min(champ_deg, 360))
        n = max(2, int(round(n_rayons * champ / (2 * math.pi))))
        a = cap - champ / 2 + np.arange(n) * (champ / n)
        if champ_deg < 360:
            a = np.append(a, cap + champ / 2)  # le bord du cone est inclus
        pi = np.floor((y + np.sin(a)[:, None] * r) / RES).astype(int)
        pj = np.floor((x + np.cos(a)[:, None] * r) / RES).astype(int)
        dedans = (pi >= 0) & (pi < NC) & (pj >= 0) & (pj < NC)
        obstacle = np.zeros(pi.shape, bool)
        if self.occlusion:
            obstacle[dedans] = self.verite[pi[dedans], pj[dedans]] == OBSTACLE
        premier = np.where(obstacle.any(axis=1), obstacle.argmax(axis=1), r.size)
        vu = (np.arange(r.size)[None, :] <= premier[:, None]) & dedans
        vu &= (pi >= i0) & (pi < i1) & (pj >= j0) & (pj < j1)
        out = np.zeros((i1 - i0, j1 - j0), bool)
        out[pi[vu] - i0, pj[vu] - j0] = True
        ri, rj = idx_floor(y) - i0, idx_floor(x) - j0
        if 0 <= ri < out.shape[0] and 0 <= rj < out.shape[1]:
            out[ri, rj] = True  # la case du rover
        return out

    def contact(self, a, b, marge=0.05):
        """Le deplacement a -> b touche-t-il un obstacle reel que le rover n'a pas vu ?
        Si oui, renvoie (point ou il s'arrete, point d'impact) : en vrai, les encodeurs
        montrent que les roues patinent sans que le robot avance."""
        d = math.dist(a, b)
        if d < 1e-9:
            return None
        n = int(d / 0.01) + 2
        for k in range(1, n):
            s = k / (n - 1)
            x, y = a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s
            i, j = idx_floor(y), idx_floor(x)
            if 0 <= i < NC and 0 <= j < NC and self.verite[i, j] == OBSTACLE:
                recul = max(0.0, s * d - marge) / d
                arret = (a[0] + (b[0] - a[0]) * recul, a[1] + (b[1] - a[1]) * recul)
                return arret, (x, y)
        return None


def longueur(chemin):
    return sum(math.dist(chemin[k], chemin[k + 1]) for k in range(len(chemin) - 1))


def avancer(pos, chemin, pas):
    x, y = pos
    reste = pas
    while chemin and reste > 0:
        nx, ny = chemin[0]
        d = math.hypot(nx - x, ny - y)
        if d <= reste:
            x, y = nx, ny
            reste -= d
            chemin.pop(0)
        else:
            x += (nx - x) / d * reste
            y += (ny - y) / d * reste
            reste = 0
    return (x, y), chemin


RESUMES = {
    'capteur': 'le rover regarde autour de lui',
    'blocage': 'BLOQUE -> zoom drone demande sur',
    'zoom': 'le drone zoome sur',
    'verifier': 'chemin par une zone inconnue -> zoom drone sur',
    'suivre': 'chemin trouve',
    'replan': 'obstacle decouvert par le rover -> replanification',
    'choc': 'CHOC contre un obstacle non vu -> signale, replanification',
    'coupure': 'LIAISON DRONE PERDUE -> mode autonome prudent',
    'inspecter': 'pas de chemin, pas de drone -> le rover va voir lui-meme',
    'analyse': "tour d'horizon camera devant l'obstacle",
    'retour': 'incertitude trop grande -> retour au dernier point vu par le drone',
    'attente': 'attente du drone',
    'reconnexion': 'drone retrouve -> position recalee',
    'degrade': 'AUCUN PASSAGE -> point le plus proche, alerte secours',
    'fin': 'arrive au point le plus proche',
    'arrivee': 'ARRIVE pres de la victime',
    'abandon': 'drone toujours absent -> arret et alerte',
    'poursuite': 'incertitude elevee, mais la victime est plus proche que le retour',
}


def mission(simu, depart, victime, tolerance=0.5, max_tours=600, journal=print, evenement=None):
    """Boucle complete, telle que l'ordinateur la fait tourner :
    decider -> (zoom drone | reconnaissance par le rover | rouler + percevoir | replanifier).

    Sans liaison drone : le rover roule deux fois plus lentement, sa position ne vient plus
    que de l'odometrie (incertitude qui grandit), les obstacles du drone sont gonfles d'autant,
    il fait la reconnaissance lui-meme, et revient au dernier point vu par le drone si
    l'incertitude depasse BUDGET_INCERTITUDE.

    `evenement(type, action, **infos)` est appele a chaque etape (tableau de bord).
    Renvoie des instantanes pour la figure de la demo."""
    carte = simu.carte
    instantanes, demandees, replans = {}, [], []
    traj, traj_vraie = [], []
    essais = {}

    def emettre(type_, action=None, **infos):
        if evenement:
            evenement(type_, action, **infos)
        if type_ != 'avance':
            detail = f' {infos["tuiles"]}' if 'tuiles' in infos else ''
            journal(f'  t={simu.t:5.1f}s  {RESUMES[type_]}{detail}')

    def noter_position():
        traj.append(simu.est)
        traj_vraie.append(simu.vraie)

    def decider():
        carte.marge_drone = 0.0 if simu.connecte() else simu.sigma
        return carte.decider(
            simu.est, victime, zone=ZONE, tolerance=tolerance, drone_disponible=simu.connecte()
        )

    def suivre_liaison():
        """Detecte une coupure ou un retour du drone. Renvoie True si l'etat a change."""
        connecte = simu.connecte()
        if connecte == simu.etait_connecte:
            return False
        simu.etait_connecte = connecte
        if connecte:
            erreur = math.dist(simu.est, simu.vraie)
            simu.recaler()
            carte.marge_drone = 0.0
            noter_position()
            emettre('reconnexion', erreur=erreur)
        else:
            emettre('coupure')
        return True

    def rouler(chemin, arrive, action, verifier=True, budget=True, finir_sur_place=True, **infos):
        """Suit le chemin pas a pas. Renvoie 'arrive', 'bloque', 'choc', 'budget' ou 'liaison'."""
        chemin = list(chemin)
        while chemin:
            if suivre_liaison():
                return 'liaison'
            if budget and not simu.connecte() and simu.sigma > BUDGET_INCERTITUDE:
                return 'budget'
            pas = PAS if simu.connecte() else PAS_PRUDENT
            avant_est, avant_vraie = simu.est, simu.vraie
            est, chemin = avancer(avant_est, chemin, pas)
            simu.t += 1.0
            cap = simu.cap
            if est != avant_est:
                cap = math.atan2(est[1] - avant_est[1], est[0] - avant_est[0])
            vraie = simu.deplacement_reel(avant_est, est, avant_vraie)
            choc = simu.contact(avant_vraie, vraie)
            impact = None
            if choc:
                arret, point = choc
                f = math.dist(avant_vraie, arret) / max(1e-9, math.dist(avant_vraie, vraie))
                est = (
                    avant_est[0] + (est[0] - avant_est[0]) * f,
                    avant_est[1] + (est[1] - avant_est[1]) * f,
                )
                vraie = arret
                # le rover situe l'impact par rapport a la ou il croit etre
                impact = (point[0] + est[0] - vraie[0], point[1] + est[1] - vraie[1])
            simu.maj_pose(est, vraie, cap)
            noter_position()
            simu.observer()
            if impact:
                carte.signaler_obstacle(*impact, rayon=0.1, t=simu.t)
                if not simu.connecte():
                    simu.provisoire('choc', impact)
                replans.append(simu.est)
                emettre('choc', action, obstacle=impact, **infos)
                return 'choc'
            if arrive(simu.est) or (finir_sur_place and not chemin):
                return 'arrive'
            if verifier:
                bloque = carte.premier_point_bloque([simu.est] + chemin, ZONE, horizon=3.0)
                if bloque:
                    replans.append(simu.est)
                    emettre('replan', action, obstacle=bloque, **infos)
                    return 'bloque'
            emettre('avance', action, **infos)
        return 'arrive'

    def revenir_et_attendre():
        """Incertitude trop grande : retour sur sa propre trace jusqu'au dernier point vu par
        le drone (le seul chemin dont on sait qu'il passe), puis attente du drone."""
        trace = list(reversed(simu.trace_fix))
        but = simu.dernier_fix
        emettre('retour', 'RETOUR', chemin=trace, point=but, sigma=simu.sigma)
        r = rouler(
            trace, lambda q: math.dist(q, trace[-1]) < 0.02, 'RETOUR', False, False, point=but
        )
        if r == 'liaison':
            return None
        return attendre_drone('RETOUR', but)

    def attendre_drone(action, point):
        """Immobile, LED allumees, le rover attend le retour du drone (au plus ATTENTE_MAX)."""
        emettre('attente', action, point=point, sigma=simu.sigma)
        attendu = 0.0
        while not simu.connecte() and attendu < ATTENTE_MAX:
            simu.t += 5.0
            attendu += 5.0
            if not simu.connecte():
                emettre('avance', action, point=point)
        if not suivre_liaison():
            emettre('abandon', action, point=point)
            return 'abandon'
        return None

    def fin():
        instantanes['fin'] = {
            'grille': carte.grille(*ZONE),
            'traj': traj,
            'traj_vraie': traj_vraie,
            'replans': replans,
            'demandees': demandees,
        }
        return instantanes

    simu.placer(depart)
    noter_position()
    simu.observer()
    emettre('capteur')
    dec = decider()
    premier = True
    for _ in range(max_tours):
        if suivre_liaison():
            dec = decider()
            continue
        action = dec['action']

        if action == 'DEMANDER_DRONE':
            tuiles = list(dec['tuiles'])
            instantanes.setdefault(
                'blocage',
                {'grille': dec['grille'], 'tuiles': tuiles, 'optimiste': dec['chemin_optimiste']},
            )
            emettre('blocage', action, tuiles=tuiles, optimiste=dec['chemin_optimiste'])
            demandees += tuiles
            for k in tuiles:
                simu.zoom_drone(k)
            emettre('zoom', action, tuiles=tuiles)
            dec = decider()
            continue

        if action == 'INSPECTER':
            secteur, cible, vue = dec['secteur'], dec['cible'], dec['point_de_vue']
            essais[secteur] = essais.get(secteur, 0) + 1
            infos = {'cible': cible, 'point_de_vue': vue}
            emettre(
                'inspecter',
                action,
                chemin=dec['chemin'],
                optimiste=dec['chemin_optimiste'],
                **infos,
            )
            r = rouler(dec['chemin'], lambda q, vue=vue: math.dist(q, vue) < 0.05, action, **infos)
            # deja assez pres pour bien voir la cible : on s'arrete et on regarde
            assez_pres = math.dist(simu.est, cible) <= PORTEE_ROVER - 0.4
            if r == 'arrive' or (r in ('bloque', 'choc') and assez_pres):
                simu.t += 2.0  # le temps de faire tourner la tourelle
                simu.observer(tour_complet=True)
                carte.inspectees.add(secteur)
                emettre('analyse', action, **infos)
            elif essais[secteur] >= 3:
                carte.inspectees.add(secteur)  # impossible d'y arriver : on passe
            if r == 'budget' and revenir_et_attendre() == 'abandon':
                break
            dec = decider()
            continue

        if action == 'OBJECTIF_DEGRADE':
            point, restant = dec['point'], dec['distance_restante']
            instantanes['degrade'] = {
                'grille': dec['grille'],
                'chemin': dec['chemin'],
                'point': point,
            }
            emettre('degrade', action, chemin=dec['chemin'], point=point, restant=restant)
            r = rouler(dec['chemin'], lambda q: False, action, False, True, False, point=point)
            if r == 'arrive' and simu.connecte():
                emettre('fin', action, point=point, restant=restant)
                break
            if r == 'arrive':
                # sans drone, rien n'est definitif : on attend qu'il revienne verifier
                if attendre_drone(action, point) == 'abandon':
                    break
                dec = decider()
                continue
            if r == 'budget' and revenir_et_attendre() == 'abandon':
                break
            dec = decider()
            continue

        # SUIVRE
        if premier:
            instantanes.setdefault(
                'premier_chemin', {'grille': dec['grille'], 'chemin': list(dec['chemin'])}
            )
        if dec['a_verifier']:
            tuiles = list(dec['a_verifier'])
            emettre('verifier', 'DEMANDER_DRONE', tuiles=tuiles, chemin=dec['chemin'])
            demandees += tuiles
            carte.demandes.update(tuiles)
            for k in tuiles:
                simu.zoom_drone(k)
            emettre('zoom', 'DEMANDER_DRONE', tuiles=tuiles)
            dec = decider()
            continue
        emettre('suivre', action, chemin=dec['chemin'], premier=premier)
        premier = False
        r = rouler(dec['chemin'], lambda q: math.dist(q, victime) <= tolerance, action)
        if r == 'budget':
            # Trop d'incertitude. Si la victime est plus proche que le dernier point vu par
            # le drone, revenir serait plus risque que finir : on continue, on reste pres
            # d'elle et on signale. Sinon, retour.
            suite = decider()
            retour = longueur(simu.trace_fix)
            if suite['action'] == 'SUIVRE' and longueur(suite['chemin']) < retour:
                emettre('poursuite', action, chemin=suite['chemin'], sigma=simu.sigma)
                r = rouler(
                    suite['chemin'],
                    lambda q: math.dist(q, victime) <= tolerance,
                    action,
                    budget=False,
                )
            elif revenir_et_attendre() == 'abandon':
                break
        if r == 'arrive':
            emettre('arrivee', action, distance=math.dist(simu.est, victime))
            return fin()
        dec = decider()
    return fin()


# ----------------------------------------------------------------------------
# Rendu
# ----------------------------------------------------------------------------
C_INCONNU = '#e1e0d9'
C_LIBRE = '#fcfcfb'
C_LIBRE_ROVER = '#d3eee2'
C_OBS_DRONE = '#52514e'
C_OBS_ROVER = '#eb6834'
C_CHEMIN = '#2a78d6'
C_ZOOM = '#6250d6'
C_ENCRE = '#0b0b0b'
C_VICTIME = '#e34948'


def hex_rgb(h):
    return [int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)]


def image(g):
    img = np.empty(g.val.shape + (3,))
    img[:] = hex_rgb(C_INCONNU)
    img[g.val == LIBRE] = hex_rgb(C_LIBRE)
    img[(g.val == LIBRE) & (g.src == SRC_ROVER)] = hex_rgb(C_LIBRE_ROVER)
    img[(g.val == OBSTACLE) & (g.src == SRC_DRONE)] = hex_rgb(C_OBS_DRONE)
    img[(g.val == OBSTACLE) & (g.src == SRC_ROVER)] = hex_rgb(C_OBS_ROVER)
    return img


def dessiner(
    ax,
    carte,
    g,
    titre,
    depart,
    victime,
    chemin=None,
    traj=None,
    tuiles=(),
    replans=(),
    point=None,
    chemin_style='-',
):
    ax.imshow(image(g), origin='lower', extent=g.etendue(), interpolation='nearest')
    for k in tuiles:
        x0, y0, cote = carte.rect_tuile(k)
        ax.add_patch(Rectangle((x0, y0), cote, cote, fill=False, ec=C_ZOOM, lw=1.6, ls='--'))
    if traj:
        ax.plot(*zip(*traj, strict=False), color=C_ENCRE, lw=1.6)
    if chemin:
        ax.plot(*zip(*chemin, strict=False), color=C_CHEMIN, lw=2, ls=chemin_style)
    for p in replans:
        ax.plot(*p, marker='o', ms=8, mfc='white', mec=C_ENCRE, mew=1.5)
    if point:
        ax.plot(*point, marker='D', ms=9, mfc=C_CHEMIN, mec='white', mew=1.5)
    ax.plot(*depart, marker='o', ms=9, mfc=C_ENCRE, mec='white', mew=1.5)
    ax.plot(*victime, marker='*', ms=16, mfc=C_VICTIME, mec='white', mew=1.2)
    ax.set_title(titre, fontsize=11, loc='left', color=C_ENCRE)
    ax.set_xlim(0, ZONE_M)
    ax.set_ylim(0, ZONE_M)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=8, colors='#898781')
    for s in ax.spines.values():
        s.set_color('#c3c2b7')


# ----------------------------------------------------------------------------
def monde_1():
    """Crete avec un passage de 70 cm + tronc invisible du drone."""
    verite = np.zeros((NC, NC), np.int8)
    rect(verite, 0, 12.0, ZONE_M, 12.6)  # crete
    rect(verite, 17.9, 12.0, 18.6, 12.6, LIBRE)  # passage de 70 cm
    for cx, cy, r in [
        (5, 16, 0.6),
        (13, 19.5, 0.8),
        (21, 5, 1.0),
        (15, 3, 0.5),
        (11.5, 15.6, 0.35),
    ]:
        disque(verite, cx, cy, r)
    vu_drone = verite.copy()
    rect(verite, 4, 6.0, 14, 6.5)  # tronc d'arbre sous les feuilles : invisible du drone
    carte = CarteFusion()
    simu = Simu(carte, verite, vu_drone, inconnu_drone=(9.6, 14.4, 14.4, 17.6))
    return carte, simu, (3.0, 3.0), (8.0, 20.0)


def monde_2():
    """Victime au fond d'un cratere ferme."""
    verite = np.zeros((NC, NC), np.int8)
    disque(verite, 19, 19, 1.7)
    disque(verite, 19, 19, 1.2, LIBRE)
    for cx, cy, r in [(8, 10, 1.0), (14, 6, 0.7), (6, 19, 0.8)]:
        disque(verite, cx, cy, r)
    carte = CarteFusion()
    simu = Simu(carte, verite, verite.copy())
    return carte, simu, (3.0, 3.0), (19.0, 19.0)


def monde_3():
    """Foret : le drone voit les falaises qui la bordent, mais rien sous les arbres
    (troncs couches et rochers). Le rover doit tout decouvrir avec sa camera."""
    verite = np.zeros((NC, NC), np.int8)
    rect(verite, 3.6, 5.0, 4.0, 21.0)  # falaise ouest, vue du drone
    rect(verite, 22.0, 5.0, 22.4, 21.0)  # falaise est, vue du drone
    vu_drone = verite.copy()
    # sous les arbres, invisible du drone : un slalom de troncs + des rochers
    rect(verite, 4.0, 9.0, 15.0, 9.4)
    rect(verite, 11.0, 12.6, 22.0, 13.0)
    rect(verite, 4.0, 16.2, 15.0, 16.6)
    for cx, cy, r in [(17.2, 10.6, 0.45), (8.6, 14.3, 0.45), (17.0, 18.0, 0.5), (12.5, 7.2, 0.4)]:
        disque(verite, cx, cy, r)
    carte = CarteFusion()
    simu = Simu(carte, verite, vu_drone)
    return carte, simu, (13.0, 3.0), (13.0, 22.0)


def monde_4(coupures=((0.0, 110.0),)):
    """Crete et tronc cache, mais la liaison avec le drone est coupee juste apres son premier
    survol (cases de 80 cm) : le rover doit trouver le passage lui-meme."""
    carte, simu, depart, victime = monde_1()
    simu.coupures = list(coupures)
    return carte, simu, depart, victime


def monde_5(coupures=((0.0, None),)):
    """Crete avec le passage plus pres (x = 9,3 a 10 m), toujours invisible du drone a 80 cm,
    et liaison drone coupee pour toujours : le rover doit le trouver seul."""
    carte, simu, depart, victime = monde_1()
    v = simu.verite
    rect(v, 17.9, 12.0, 18.6, 12.6)  # on referme le passage d'origine...
    rect(v, 9.3, 12.0, 10.0, 12.6, LIBRE)  # ...et on en ouvre un plus pres
    rect(simu.vu_drone, 17.9, 12.0, 18.6, 12.6)
    rect(simu.vu_drone, 9.3, 12.0, 10.0, 12.6, LIBRE)
    simu.coupures = list(coupures)
    return carte, simu, depart, victime


def scenario_1():
    print('\n=== Scenario 1 : crete avec un passage etroit + tronc cache ===')
    carte, simu, depart, victime = monde_1()
    simu.survol_initial(z=0)
    print(f'  drone : 1er survol, cases de {carte.res(0) * 100:.0f} cm')
    return carte, depart, victime, mission(simu, depart, victime)


def scenario_2():
    print("\n=== Scenario 2 : victime au fond d'un cratere ferme ===")
    carte, simu, depart, victime = monde_2()
    simu.survol_initial(z=0)
    return carte, depart, victime, mission(simu, depart, victime)


def main():
    c1, d1, v1, s1 = scenario_1()
    c2, d2, v2, s2 = scenario_2()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12.6), facecolor='#fcfcfb')
    b = s1['blocage']
    dessiner(
        axes[0, 0],
        c1,
        b['grille'],
        '1. Vue drone (80 cm) : aucun chemin.\nZoom demandé là où un passage peut se cacher',
        d1,
        v1,
        chemin=b['optimiste'],
        tuiles=b['tuiles'],
        chemin_style=':',
    )
    p = s1['premier_chemin']
    dessiner(
        axes[0, 1],
        c1,
        p['grille'],
        '2. Après les zooms (20 cm) : le passage apparaît,\npremier chemin planifié',
        d1,
        v1,
        chemin=p['chemin'],
        tuiles=s1['fin']['demandees'],
    )
    f = s1['fin']
    dessiner(
        axes[1, 0],
        c1,
        f['grille'],
        '3. Carte fusionnée finale : tronc découvert\npar le rover, replanification, arrivée',
        d1,
        v1,
        traj=f['traj'],
        replans=f['replans'],
    )
    dg = s2['degrade']
    dessiner(
        axes[1, 1],
        c2,
        dg['grille'],
        '4. Cratère fermé confirmé par le drone :\nobjectif dégradé (point le plus proche)',
        d2,
        v2,
        chemin=dg['chemin'],
        tuiles=s2['fin']['demandees'],
        point=dg['point'],
    )

    legende = [
        Patch(fc=C_OBS_DRONE, label='Obstacle (drone)'),
        Patch(fc=C_OBS_ROVER, label='Obstacle (rover)'),
        Patch(fc=C_LIBRE, ec='#c3c2b7', label='Libre (drone)'),
        Patch(fc=C_LIBRE_ROVER, label='Libre (vu par le rover)'),
        Patch(fc=C_INCONNU, label='Inconnu (-1)'),
        Line2D([], [], color=C_CHEMIN, lw=2, label='Chemin planifié'),
        Line2D([], [], color=C_CHEMIN, lw=2, ls=':', label='Chemin « si le passage existait »'),
        Line2D([], [], color=C_ENCRE, lw=1.6, label='Trajet du rover'),
        Patch(fc='none', ec=C_ZOOM, ls='--', lw=1.6, label='Tuile zoomée par le drone'),
        Line2D([], [], marker='o', ls='', mfc='white', mec=C_ENCRE, ms=8, label='Replanification'),
        Line2D([], [], marker='*', ls='', mfc=C_VICTIME, mec='white', ms=14, label='Victime'),
        Line2D(
            [],
            [],
            marker='D',
            ls='',
            mfc=C_CHEMIN,
            mec='white',
            ms=9,
            label='Point le plus proche',
        ),
    ]
    fig.legend(handles=legende, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig('demo_fusion.png', dpi=110, facecolor=fig.get_facecolor())
    print('\nFigure : demo_fusion.png')


if __name__ == '__main__':
    main()
