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
PAS = 0.3  # avance du rover par tour (m)


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
    def __init__(self, carte, verite, vu_par_drone, inconnu_drone=None, occlusion=True):
        self.carte, self.verite, self.vu_drone = carte, verite, vu_par_drone
        self.inconnu_drone = inconnu_drone  # zone que le drone n'a pas pu voir au 1er survol
        self.occlusion = occlusion  # False = le rover voit a travers les obstacles (irrealiste)
        self.t = 0.0

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

    def capteur_rover(self, x, y):
        """Ce que voit le rover : un disque de PORTEE_ROVER autour de lui.
        Avec occlusion, une case cachee derriere un obstacle est envoyee en -1 (pas vue)."""
        i0, i1 = max(0, idx(y - PORTEE_ROVER)), min(NC, idx(y + PORTEE_ROVER))
        j0, j1 = max(0, idx(x - PORTEE_ROVER)), min(NC, idx(x + PORTEE_ROVER))
        loc = self.verite[i0:i1, j0:j1].copy()
        yy, xx = np.mgrid[i0:i1, j0:j1]
        hors = ((xx + 0.5) * RES - x) ** 2 + ((yy + 0.5) * RES - y) ** 2 > PORTEE_ROVER**2
        if self.occlusion:
            hors |= self.cache(x, y, yy, xx)
        loc[hors] = INCONNU
        for msg in decouper(loc, j0 * RES, i0 * RES, self.carte.zmax, self.carte, 'rover', self.t):
            self.carte.recevoir(msg)

    def cache(self, x, y, yy, xx):
        """Lancer de rayons : une case est cachee si la ligne rover -> case traverse un
        obstacle avant de l'atteindre. La face avant d'un obstacle reste visible, pas ce qu'il
        y a derriere."""
        cx, cy = (xx + 0.5) * RES, (yy + 0.5) * RES
        n_pas = int(math.ceil(PORTEE_ROVER / (RES / 2)))
        s = np.arange(1, n_pas)[:, None, None] / n_pas  # fractions du rayon, sans les extremites
        px, py = x + (cx - x) * s, y + (cy - y) * s
        pi = np.clip(np.floor(py / RES).astype(int), 0, NC - 1)
        pj = np.clip(np.floor(px / RES).astype(int), 0, NC - 1)
        sur_la_case = (pi == yy) & (pj == xx)
        depart = (pi == idx_floor(y)) & (pj == idx_floor(x))
        bloque = (self.verite[pi, pj] == OBSTACLE) & ~sur_la_case & ~depart
        return bloque.any(axis=0)


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


def mission(simu, depart, victime, tolerance=0.5, max_tours=400, journal=print):
    """Boucle complete : decider -> (zoom drone | rouler + percevoir | replanifier)."""
    carte = simu.carte
    pos, traj, replans, demandees = depart, [depart], [], []
    instantanes = {}
    simu.capteur_rover(*pos)
    dec = carte.decider(pos, victime, zone=ZONE, tolerance=tolerance)
    chemin = None
    for _ in range(max_tours):
        a = dec['action']
        if a == 'DEMANDER_DRONE':
            journal(f'  t={simu.t:5.1f}s  BLOQUE -> zoom drone demande sur {dec["tuiles"]}')
            instantanes.setdefault(
                'blocage',
                {
                    'grille': dec['grille'],
                    'tuiles': list(dec['tuiles']),
                    'optimiste': dec.get('chemin_optimiste'),
                },
            )
            demandees += dec['tuiles']
            for k in dec['tuiles']:
                simu.zoom_drone(k)
            dec = carte.decider(pos, victime, zone=ZONE, tolerance=tolerance)
            continue
        if a == 'OBJECTIF_DEGRADE':
            journal(
                f'  t={simu.t:5.1f}s  AUCUN PASSAGE -> point le plus proche '
                f'({dec["point"][0]:.1f}, {dec["point"][1]:.1f}), '
                f"reste {dec['distance_restante']:.1f} m jusqu'a la victime. Alerte secours."
            )
            instantanes['degrade'] = {
                'grille': dec['grille'],
                'chemin': dec['chemin'],
                'point': dec['point'],
            }
            break
        # SUIVRE
        if chemin is None:
            instantanes.setdefault(
                'premier_chemin', {'grille': dec['grille'], 'chemin': list(dec['chemin'])}
            )
        chemin = list(dec['chemin'])
        if dec['a_verifier']:
            journal(
                f'  t={simu.t:5.1f}s  chemin par une zone inconnue -> '
                f'zoom drone sur {dec["a_verifier"]}'
            )
            demandees += dec['a_verifier']
            carte.demandes.update(dec['a_verifier'])
            for k in dec['a_verifier']:
                simu.zoom_drone(k)
            dec = carte.decider(pos, victime, zone=ZONE, tolerance=tolerance)
            continue
        # rouler jusqu'a ce que le chemin devienne invalide ou qu'on arrive
        while True:
            pos, chemin = avancer(pos, chemin, PAS)
            simu.t += PAS / 0.3
            traj.append(pos)
            simu.capteur_rover(*pos)
            if math.hypot(pos[0] - victime[0], pos[1] - victime[1]) <= tolerance or not chemin:
                journal(
                    f'  t={simu.t:5.1f}s  ARRIVE pres de la victime ({pos[0]:.1f}, {pos[1]:.1f})'
                )
                instantanes['fin'] = {
                    'grille': carte.grille(*ZONE),
                    'traj': traj,
                    'replans': replans,
                    'demandees': demandees,
                }
                return instantanes
            if not carte.chemin_valide([pos] + chemin, ZONE, horizon=3.0):
                journal(f'  t={simu.t:5.1f}s  obstacle decouvert par le rover -> replanification')
                replans.append(pos)
                break
        dec = carte.decider(pos, victime, zone=ZONE, tolerance=tolerance)
    instantanes['fin'] = {
        'grille': carte.grille(*ZONE),
        'traj': traj,
        'replans': replans,
        'demandees': demandees,
    }
    return instantanes


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
