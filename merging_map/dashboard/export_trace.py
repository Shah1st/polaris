"""
export_trace.py - Rejoue les deux scenarios de la demo et enregistre tout ce qui se passe
(tuiles recues, decisions, chemins, position du rover) dans trace.js, lu par index.html.

Lancer depuis n'importe ou :  python merging_map/dashboard/export_trace.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))

from demo_fusion import PAS, PORTEE_ROVER, ZONE, ZONE_M, avancer, monde_1, monde_2  # noqa: E402
from fusion_carte import INCONNU, LIBRE, OBSTACLE, SRC_ROVER  # noqa: E402


# ----------------------------------------------------------------------------
# Encodage compact
# ----------------------------------------------------------------------------
def encoder(data):
    """Matrice -1/0/1 -> texte compresse par plages, avec u = inconnu, l = libre, x = obstacle.
    'u37' = 37 cases inconnues (des lettres, pour ne pas confondre valeur et compteur)."""
    a = np.asarray(data, np.int8).ravel()
    car = np.full(a.shape, 'u', dtype='<U1')
    car[a == LIBRE] = 'l'
    car[a == OBSTACLE] = 'x'
    out, n = [], len(car)
    i = 0
    while i < n:
        j = i
        while j < n and car[j] == car[i]:
            j += 1
        out.append(car[i] + (str(j - i) if j - i > 1 else ''))
        i = j
    return ''.join(out)


def simplifier(chemin):
    """Garde seulement les points ou le chemin change de direction."""
    if not chemin:
        return []
    pts = [chemin[0]]
    for k in range(1, len(chemin) - 1):
        ax, ay = chemin[k][0] - chemin[k - 1][0], chemin[k][1] - chemin[k - 1][1]
        bx, by = chemin[k + 1][0] - chemin[k][0], chemin[k + 1][1] - chemin[k][1]
        if abs(ax * by - ay * bx) > 1e-9 or ax * bx + ay * by < 0:
            pts.append(chemin[k])
    if len(chemin) > 1:
        pts.append(chemin[-1])
    return [[round(x, 3), round(y, 3)] for x, y in pts]


def longueur(chemin):
    return sum(math.dist(chemin[k], chemin[k + 1]) for k in range(len(chemin) - 1))


def premier_point_bloque(carte, chemin, horizon=3.0):
    """Meme test que CarteFusion.chemin_valide, mais renvoie le point bloque."""
    plan = carte.grille_plan(carte.grille(*ZONE))
    parcouru = 0.0
    for n, (x, y) in enumerate(chemin):
        if n:
            parcouru += math.dist(chemin[n - 1], (x, y))
        if parcouru > horizon:
            break
        i, j = plan.ij(x, y)
        if n > 0 and plan.dedans(i, j) and plan.bloque[i, j]:
            return x, y
    return None


# ----------------------------------------------------------------------------
# Enregistreur
# ----------------------------------------------------------------------------
class Traceur:
    def __init__(self, carte, simu, depart):
        self.carte, self.simu = carte, simu
        self.msgs, self.frames, self.chemins = [], [], []
        self.attente = []
        self.pos, self.cap, self.traj = depart, math.pi / 2, [depart]
        self.chemin_id = None
        recevoir = carte.recevoir

        def tracer(msg):
            recevoir(msg)
            self.attente.append(len(self.msgs))
            source = 0 if msg['source'] == 'drone' else 1
            self.msgs.append([source, msg['z'], msg['x'], msg['y'], encoder(msg['data'])])

        carte.recevoir = tracer

    def ajouter_chemin(self, chemin):
        self.chemins.append(simplifier(chemin))
        return len(self.chemins) - 1

    def deplacer(self, pos):
        dx, dy = pos[0] - self.pos[0], pos[1] - self.pos[1]
        if math.hypot(dx, dy) > 1e-6:
            self.cap = math.atan2(dy, dx)
        self.pos = pos
        self.traj.append(pos)

    def frame(self, type_, titre, texte, action=None, **extra):
        f = {
            'type': type_,
            'titre': titre,
            'texte': texte,
            'action': action,
            'msgs': self.attente,
            'pos': [round(self.pos[0], 3), round(self.pos[1], 3)],
            'cap': round(self.cap, 3),
            't': round(self.simu.t, 1),
            'chemin': self.chemin_id,
            'traj': len(self.traj),
        }
        f.update(extra)
        self.frames.append(f)
        self.attente = []


def mission_tracee(carte, simu, depart, victime, tolerance=0.5, max_tours=600):
    tr = Traceur(carte, simu, depart)
    simu.survol_initial(z=0)
    tr.frame(
        'survol',
        'Premier survol du drone',
        f'Le drone survole toute la zone en altitude, avec des cases de '
        f'{carte.res(0) * 100:.0f} cm. Il voit les gros obstacles, mais un passage étroit '
        f'ou un tronc sous les arbres peut lui échapper.',
    )
    simu.capteur_rover(*tr.pos)
    tr.frame(
        'capteur',
        'Le rover regarde autour de lui',
        f'Le rover envoie ce que voit sa caméra dans un rayon de {PORTEE_ROVER:.0f} m, '
        f"en cases de {carte.res_fine * 100:.0f} cm. Ce qu'il voit l'emporte sur le drone.",
    )
    dec = carte.decider(tr.pos, victime, zone=ZONE, tolerance=tolerance)
    premier = True
    for _ in range(max_tours):
        action = dec['action']

        if action == 'DEMANDER_DRONE':
            optimiste = tr.ajouter_chemin(dec['chemin_optimiste'])
            tuiles = [list(k) for k in dec['tuiles']]
            tr.frame(
                'blocage',
                'Bloqué : aucun chemin',
                'Sur la carte fusionnée, aucun chemin ne mène à la victime. Le module se demande '
                ': '
                '« si les obstacles vus seulement de loin par le drone étaient franchissables, '
                'par où passerais-je ? » (pointillés). Il demande au drone de zoomer sur les '
                'tuiles que ce chemin traverse.',
                action,
                tuiles=tuiles,
                optimiste=optimiste,
            )
            for k in dec['tuiles']:
                simu.zoom_drone(k)
            res_zoom = carte.res(dec['tuiles'][0][0]) * 100
            tr.frame(
                'zoom',
                'Le drone zoome',
                f'Le drone descend et renvoie {len(tuiles)} tuile(s) en cases de '
                f'{res_zoom:.0f} cm. Dans ces tuiles, la vue fine remplace la vue grossière.',
                action,
                tuiles=tuiles,
            )
            dec = carte.decider(tr.pos, victime, zone=ZONE, tolerance=tolerance)
            continue

        if action == 'OBJECTIF_DEGRADE':
            tr.chemin_id = tr.ajouter_chemin(dec['chemin'])
            point = [round(dec['point'][0], 3), round(dec['point'][1], 3)]
            tr.frame(
                'degrade',
                'Aucun passage : objectif dégradé',
                'Le drone a vérifié toutes les zones douteuses : la victime est inaccessible. '
                f'Le rover va au point accessible le plus proche, à '
                f"{dec['distance_restante']:.1f} m d'elle, et alerte les secours.",
                action,
                point=point,
                restant=round(dec['distance_restante'], 2),
            )
            chemin = list(dec['chemin'])
            while chemin:
                pos, chemin = avancer(tr.pos, chemin, PAS)
                simu.t += 1.0
                tr.deplacer(pos)
                simu.capteur_rover(*pos)
                tr.frame('avance', 'Le rover avance', '', action, point=point)
            tr.frame(
                'fin',
                'Arrivé au point le plus proche',
                f"Le rover s'arrête à {dec['distance_restante']:.1f} m de la victime et transmet "
                'sa position et la carte aux secours.',
                action,
                point=point,
            )
            break

        # SUIVRE
        tr.chemin_id = tr.ajouter_chemin(dec['chemin'])
        if dec['a_verifier']:
            tuiles = [list(k) for k in dec['a_verifier']]
            tr.frame(
                'verifier',
                'Chemin par une zone inconnue',
                "Le meilleur chemin traverse une zone que personne n'a vue. Avant de s'y "
                "engager, le module demande au drone d'y jeter un œil.",
                'DEMANDER_DRONE',
                tuiles=tuiles,
            )
            carte.demandes.update(dec['a_verifier'])
            for k in dec['a_verifier']:
                simu.zoom_drone(k)
            tr.frame(
                'zoom',
                'Le drone zoome',
                'Le drone renvoie la zone inconnue.',
                'DEMANDER_DRONE',
                tuiles=tuiles,
            )
            dec = carte.decider(tr.pos, victime, zone=ZONE, tolerance=tolerance)
            continue

        tr.frame(
            'suivre',
            'Chemin trouvé' if premier else 'Nouveau chemin',
            f'Le A* trouve un chemin de {longueur(dec["chemin"]):.1f} m sur la carte fusionnée '
            '(obstacles gonflés de 15 cm pour la largeur du robot). Le rover le suit et continue '
            "d'observer à chaque pas.",
            action,
        )
        premier = False
        chemin = list(dec['chemin'])
        while True:
            pos, chemin = avancer(tr.pos, chemin, PAS)
            simu.t += 1.0
            tr.deplacer(pos)
            simu.capteur_rover(*pos)
            if math.dist(pos, victime) <= tolerance or not chemin:
                tr.frame(
                    'arrivee',
                    'Arrivé près de la victime',
                    f'Le rover est à {math.dist(pos, victime):.1f} m de la victime, après '
                    f'{longueur(tr.traj):.1f} m parcourus.',
                    action,
                )
                return tr
            bloque = premier_point_bloque(carte, [pos] + chemin)
            if bloque:
                tr.frame(
                    'replan',
                    'Obstacle découvert : replanification',
                    'La caméra du rover voit un obstacle sur le chemin prévu (cercle), que le '
                    "drone n'avait pas vu. Le chemin n'est plus valide : le module replanifie "
                    'sur la carte fusionnée.',
                    action,
                    obstacle=[round(bloque[0], 3), round(bloque[1], 3)],
                )
                break
            tr.frame('avance', 'Le rover avance', '', action)
        dec = carte.decider(tr.pos, victime, zone=ZONE, tolerance=tolerance)
    return tr


def exporter(nom, description, monde):
    carte, simu, depart, victime = monde()
    tr = mission_tracee(carte, simu, depart, victime)
    g = carte.grille(*ZONE)
    return {
        'nom': nom,
        'description': description,
        'zone': ZONE_M,
        'n': carte.n,
        'zmax': carte.zmax,
        'res_fine': carte.res_fine,
        'rayon_robot': carte.rayon_robot,
        'portee': PORTEE_ROVER,
        'depart': list(depart),
        'victime': list(victime),
        'verite': encoder(simu.verite),
        'msgs': tr.msgs,
        'chemins': tr.chemins,
        'traj': [[round(x, 3), round(y, 3)] for x, y in tr.traj],
        'frames': tr.frames,
        # pour verifier que la page refait la meme fusion que Python
        'controle': {
            'obstacle': int((g.val == OBSTACLE).sum()),
            'libre': int((g.val == LIBRE).sum()),
            'inconnu': int((g.val == INCONNU).sum()),
            'rover': int((g.src == SRC_ROVER).sum()),
        },
    }


def main():
    trace = {
        'scenarios': [
            exporter(
                'Crête et tronc caché',
                'Une crête barre la route, avec un passage de 70 cm que le drone ne voit pas '
                'de haut. Un tronc couché sous les arbres est invisible depuis le ciel.',
                monde_1,
            ),
            exporter(
                'Cratère fermé',
                "La victime est au fond d'un cratère complètement fermé. Aucun chemin "
                "n'existe : le rover doit s'approcher au plus près.",
                monde_2,
            ),
        ]
    }
    sortie = ICI / 'trace.js'
    texte = 'window.TRACE = ' + json.dumps(trace, ensure_ascii=False, separators=(',', ':'))
    sortie.write_text(texte + ';\n', encoding='utf-8')
    for s in trace['scenarios']:
        print(f'{s["nom"]} : {len(s["frames"])} etapes, {len(s["msgs"])} tuiles')
    print(f'{sortie} : {sortie.stat().st_size / 1e6:.2f} Mo')


if __name__ == '__main__':
    main()
