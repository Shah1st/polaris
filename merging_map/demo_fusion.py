"""
demo_fusion.py - Demo de fusion drone + rover sur un monde simule.

Scenario 1 : une crete barre le chemin. Vue de haut (cases de 80 cm) le drone ne voit
             pas le passage de 70 cm -> le rover demande des zooms jusqu'a le trouver,
             puis decouvre en roulant un tronc d'arbre que le drone n'avait pas vu.
Scenario 2 : la victime est au fond d'un cratere ferme -> apres verification par le
             drone, le rover va au point accessible le plus proche.

La mission elle-meme est dans cerveau.py ; ce fichier fournit le monde simule et Simu,
l'implementation simulee de l'interface robot.Robot.

Lancer :  python demo_fusion.py      (produit demo_fusion.png)
"""

import math

import cerveau
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fusion_carte import INCONNU, LIBRE, OBSTACLE, SRC_DRONE, SRC_ROVER, CarteFusion, decouper
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from robot import RobotOdometrie

ZONE_M = 25.6
RES = 0.05
NC = int(round(ZONE_M / RES))  # 512 cases fines de cote
ZONE = (0.0, 0.0, ZONE_M, ZONE_M)
PORTEE_ROVER = 2.0  # portee de la camera du rover (m)


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


class Simu(RobotOdometrie):
    """Le rover et le drone simules : implementation de robot.Robot pour les tests et le
    tableau de bord. Connait la verite terrain (position reelle, obstacles reels)."""

    portee_camera = PORTEE_ROVER

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
        zones_blanches=(),
    ):
        self.verite, self.vu_drone = verite, vu_par_drone
        self.inconnu_drone = inconnu_drone  # zone que le drone n'a pas pu voir au 1er survol
        self.occlusion = occlusion  # False = le rover voit a travers les obstacles (irrealiste)
        self.champ = champ  # angle de vue de la camera, en degres, centre sur le cap du rover
        self.cap = math.pi / 2  # direction (estimee) du rover (radians, 0 = est, pi/2 = nord)
        self.t = 0.0
        # coupures de la liaison ordinateur <-> drone : [(debut, fin)] en s, fin=None = pour
        # toujours. Sans drone, la position du rover vient de l'odometrie, qui derive : le cap
        # derive (biais + bruit) et les roues glissent un peu.
        self.coupures = list(coupures)
        # zones blanches : ou la liaison ordinateur <-> rover ne passe pas (x, y, rayon)
        self.zones_blanches = list(zones_blanches)
        self.rng = np.random.default_rng(graine)
        self.biais_cap = 0.0015  # rad par metre parcouru (gyroscope calibre : ~2 deg sur 20 m)
        self.bruit_cap = 0.005  # rad par racine de metre
        self.glissement = 0.015  # le rover parcourt 1,5 % de plus que ce que mesurent ses roues
        super().__init__(carte)

    # ---- interface robot.Robot ------------------------------------------------
    def temps(self):
        return self.t

    def attendre(self, duree):
        self.t += duree

    def drone_disponible(self):
        return self.connecte()

    def connecte(self):
        """La liaison ordinateur <-> drone fonctionne-t-elle a l'instant t ?"""
        return not any(
            debut <= self.t < (math.inf if fin is None else fin) for debut, fin in self.coupures
        )

    def rover_joignable(self):
        """L'ordinateur peut-il parler au rover, la ou il est reellement ?"""
        x, y = self.vraie
        return not any(
            (x - zx) ** 2 + (y - zy) ** 2 <= zr * zr for zx, zy, zr in self.zones_blanches
        )

    def position_reelle(self):
        return self.vraie

    def placer(self, pos):
        """Met le rover en `pos`, position connue (le drone vient de le voir)."""
        super().placer(pos)
        self.vraie = pos  # position reelle (l'estimee est self.est)
        self.erreur_cap = 0.0  # erreur de cap de l'odometrie (rad)

    def recaler(self):
        """Le drone voit le marqueur ArUco du rover : la position estimee devient la vraie.
        Les observations faites pendant la coupure sont reprojetees."""
        erreur = math.dist(self.est, self.vraie)
        self._recaler(self.vraie)
        self.erreur_cap = 0.0
        return erreur

    def deplacer(self, cible):
        """Le rover execute le deplacement commande (une seconde). Sans drone, sa position
        reelle derive de l'estimee ; s'il touche un obstacle reel, il s'arrete avant."""
        avant_est, avant_vraie = self.est, self.vraie
        est = cible
        self.t += 1.0
        cap = self.cap
        if est != avant_est:
            cap = math.atan2(est[1] - avant_est[1], est[0] - avant_est[0])
        vraie = self.deplacement_reel(avant_est, est, avant_vraie)
        choc = self.contact(avant_vraie, vraie)
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
        self.maj_pose(est, vraie, cap)
        return impact

    def demander_zoom(self, tuile):
        self.zoom_drone(tuile)

    def observer(self, tour_complet=False):
        """La camera voit le terrain reel autour de la position REELLE, mais le rover ecrit
        ses tuiles la ou il CROIT etre : sans drone, elles sont decalees de son erreur.
        tour_complet : tour d'horizon avec la tourelle pan-tilt (360 degres)."""
        decalage = (self.est[0] - self.vraie[0], self.est[1] - self.vraie[1])
        champ = 360 if tour_complet else self.champ
        self._voir(*self.vraie, self.cap + self.erreur_cap, champ, decalage)

    # ---- physique simulee ------------------------------------------------------
    def deplacement_reel(self, avant_est, apres_est, avant_vraie):
        """Avec le drone, la position est recalee a chaque pas : le reel suit l'estime.
        Sans drone, le cap derive et les roues glissent."""
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
        self.vraie, self.cap = vraie, cap
        if self.connecte():
            self.est = est
            self.recaler()
        else:
            self._avancer_estimation(est)

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
        return self.publier_vue(loc, x0, y0)

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


def mission(simu, depart, victime, tolerance=0.5, max_tours=600, journal=print, evenement=None):
    """La mission du cerveau (cerveau.mission) sur le rover simule, dans la zone de la demo."""
    return cerveau.mission(
        simu,
        simu.carte,
        depart,
        victime,
        zone=ZONE,
        tolerance=tolerance,
        max_tours=max_tours,
        journal=journal,
        evenement=evenement,
    )


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


def monde_6(zones_blanches=((15.2, 16.2, 1.6),)):
    """Crete et tronc cache, avec une zone sans reseau (la liaison ordinateur <-> rover ne
    passe pas) en plein sur le chemin normal, apres le passage."""
    carte, simu, depart, victime = monde_1()
    simu.zones_blanches = list(zones_blanches)
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
