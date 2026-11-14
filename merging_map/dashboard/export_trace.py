"""
export_trace.py - Rejoue les scenarios de la demo et enregistre tout ce qui se passe
(tuiles recues, decisions, chemins, positions estimee et reelle, liaison avec le drone) dans
trace.js, lu par index.html.

La mission est celle de demo_fusion.mission : ce script ne fait que l'ecouter.

Lancer depuis n'importe ou :  python merging_map/dashboard/export_trace.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))

from demo_fusion import (  # noqa: E402
    BUDGET_INCERTITUDE,
    PORTEE_ROVER,
    ZONE,
    ZONE_M,
    longueur,
    mission,
    monde_1,
    monde_2,
    monde_3,
    monde_4,
    monde_5,
)
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


def r3(p):
    return [round(float(p[0]), 3), round(float(p[1]), 3)]


def cm(v):
    return f'{v * 100:.0f} cm'


# ----------------------------------------------------------------------------
# Textes affiches pour chaque etape
# ----------------------------------------------------------------------------
def textes(type_, simu, carte, infos, distance):
    if type_ == 'avance':
        return 'Le rover avance', ''
    if type_ == 'survol':
        return 'Premier survol du drone', (
            f'Le drone survole toute la zone en altitude, avec des cases de {cm(carte.res(0))}. '
            'Il voit les gros obstacles, mais un passage étroit ou un tronc sous les arbres peut '
            'lui échapper.'
        )
    if type_ == 'capteur':
        return 'Le rover regarde autour de lui', (
            f'Le rover envoie ce que voit sa caméra (champ de {simu.champ}°, portée '
            f'{PORTEE_ROVER:.0f} m), en cases de {cm(carte.res_fine)}. '
            "Ce qu'il voit l'emporte sur le drone."
        )
    if type_ == 'blocage':
        return 'Bloqué : aucun chemin', (
            'Sur la carte fusionnée, aucun chemin ne mène à la victime. Le module se demande : '
            '« si les obstacles vus seulement de loin par le drone étaient franchissables, par '
            'où passerais-je ? » (pointillés). Il demande au drone de zoomer sur les tuiles que '
            'ce chemin traverse.'
        )
    if type_ == 'zoom':
        z = infos['tuiles'][0][0]
        return 'Le drone zoome', (
            f'Le drone descend et renvoie {len(infos["tuiles"])} tuile(s) en cases de '
            f'{cm(carte.res(z))}. Dans ces tuiles, la vue fine remplace la vue grossière.'
        )
    if type_ == 'verifier':
        return 'Chemin par une zone inconnue', (
            "Le meilleur chemin traverse une zone que personne n'a vue. Avant de s'y engager, "
            "le module demande au drone d'y jeter un œil."
        )
    if type_ == 'suivre':
        marge = ''
        if carte.marge_drone > 0:
            marge = (
                f', et de {cm(carte.marge_drone)} de plus pour ceux du drone situés loin de ce '
                'que le rover a vu lui-même'
            )
        return 'Chemin trouvé' if infos.get('premier') else 'Nouveau chemin', (
            f'Le A* trouve un chemin de {longueur(infos["chemin"]):.1f} m sur la carte fusionnée '
            f'(obstacles gonflés de {cm(carte.rayon_robot)} pour la largeur du robot{marge}). '
            "Le rover le suit et continue d'observer à chaque pas."
        )
    if type_ == 'replan':
        return 'Obstacle découvert : replanification', (
            'La caméra du rover voit un obstacle sur le chemin prévu (cercle). Le chemin '
            "n'est plus valide : le module replanifie sur la carte fusionnée."
        )
    if type_ == 'choc':
        return 'Choc : obstacle pas vu', (
            'Le rover a touché un obstacle qui était hors de son champ de vision. Ses roues '
            "patinent (les encodeurs ne montrent plus d'avance) : il recule de 5 cm, marque "
            "l'obstacle sur la carte (cercle) et replanifie."
        )
    if type_ == 'degrade':
        if simu.connecte():
            return 'Aucun passage : objectif dégradé', (
                'Le drone a vérifié toutes les zones douteuses : la victime est inaccessible. '
                'Le rover va au point accessible le plus proche, à '
                f"{infos['restant']:.1f} m d'elle, et alerte les secours."
            )
        return 'Aucun passage trouvé par le rover', (
            "Le rover a inspecté tout ce qu'il pouvait atteindre sans trouver de passage. Il "
            f'va au point accessible le plus proche, à {infos["restant"]:.1f} m de la victime, '
            "et y attendra le drone : sans lui, ce n'est pas une conclusion définitive."
        )
    if type_ == 'fin':
        return 'Arrivé au point le plus proche', (
            f"Le rover s'arrête à {infos['restant']:.1f} m de la victime et transmet sa position "
            'et la carte aux secours.'
        )
    if type_ == 'arrivee':
        texte = (
            f'Le rover est à {infos["distance"]:.1f} m de la victime, après {distance:.1f} m '
            'parcourus.'
        )
        if not simu.connecte():
            texte += (
                f" Sans drone, il n'est sûr de sa position qu'à ±{cm(simu.sigma)} près : il reste "
                'sur place et signale sa présence (LED, synthèse vocale).'
            )
        return 'Arrivé près de la victime', texte
    if type_ == 'coupure':
        return 'Liaison avec le drone perdue', (
            "L'ordinateur ne reçoit plus rien du drone : plus de nouvelles tuiles, plus de zoom, "
            'et surtout plus de position absolue du rover. Le rover continue avec la carte '
            'figée et sa caméra, deux fois plus lentement. Sa position ne vient plus que de '
            "l'odométrie, qui dérive : l'incertitude grandit avec la distance (cercle rouge)."
        )
    if type_ == 'inspecter':
        return 'Reconnaissance par le rover', (
            'Aucun chemin, et le drone ne répond pas. Le module cherche où un passage pourrait '
            'se cacher sous un obstacle vu seulement de loin (pointillés) et envoie le rover '
            "regarder lui-même : il s'approche à environ 1 m de la cible (cercle violet), en "
            'restant hors de la marge de sécurité.'
        )
    if type_ == 'analyse':
        return "Analyse de l'obstacle", (
            "Arrivé au point d'observation, le rover fait un tour d'horizon avec sa caméra "
            "(tourelle pan-tilt). Ce qu'il voit remplace la vue grossière du drone. Ce secteur "
            'est noté comme inspecté : si rien ne passe, la recherche continue plus loin.'
        )
    if type_ == 'retour':
        return "Trop d'incertitude : retour", (
            f"Sans drone, l'incertitude sur la position du rover atteint {cm(infos['sigma'])}, "
            f'au-delà de la limite de {cm(BUDGET_INCERTITUDE)}. Il rebrousse chemin sur sa '
            "propre trace (le seul chemin dont on sait qu'il passe) jusqu'au dernier point où "
            "le drone l'a vu."
        )
    if type_ == 'poursuite':
        return 'Incertitude élevée, mais on continue', (
            f"L'incertitude atteint {cm(infos['sigma'])}, au-delà de la limite de "
            f'{cm(BUDGET_INCERTITUDE)}. Mais la victime est plus proche que le dernier point vu '
            'par le drone : revenir serait plus risqué que finir. Le rover continue prudemment.'
        )
    if type_ == 'attente':
        return 'En attente du drone', (
            "Le rover s'arrête, LED allumées, et attend que le drone revienne recaler sa "
            'position et vérifier la zone.'
        )
    if type_ == 'reconnexion':
        return 'Drone retrouvé : position recalée', (
            "Le drone revoit le marqueur du rover. L'erreur de position accumulée était de "
            f'{cm(infos["erreur"])}. La position est recalée, et tout ce que le rover a vu '
            "pendant la coupure est réécrit au bon endroit (l'erreur est répartie le long du "
            'trajet) : les obstacles fantômes disparaissent.'
        )
    if type_ == 'abandon':
        return 'Drone toujours absent', (
            "Après 5 minutes sans drone, le rover reste à l'arrêt en lieu sûr et l'ordinateur "
            "alerte l'équipe."
        )
    raise ValueError(type_)


# ----------------------------------------------------------------------------
# Enregistreur
# ----------------------------------------------------------------------------
class Traceur:
    """Ecoute la mission et la carte : chaque tuile recue devient un message, chaque
    evenement de la mission une etape (frame)."""

    def __init__(self, carte, simu):
        self.carte, self.simu = carte, simu
        self.msgs, self.frames, self.chemins, self.attente = [], [], [], []
        self.traj, self.traj_vraie = [], []
        self.chemin_id = None
        recevoir = carte.recevoir
        instantane = carte.instantane_rover
        restaurer = carte.restaurer_rover

        def tracer(msg):
            recevoir(msg)
            self.attente.append(len(self.msgs))
            source = 0 if msg['source'] == 'drone' else 1
            self.msgs.append([source, msg['z'], msg['x'], msg['y'], encoder(msg['data'])])

        def tracer_instantane():
            self.attente.append(len(self.msgs))
            self.msgs.append([-1])  # la page copie la couche rover
            return instantane()

        def tracer_restauration(inst):
            restaurer(inst)
            self.attente.append(len(self.msgs))
            self.msgs.append([-2])  # la page remet cette copie

        carte.recevoir = tracer
        carte.instantane_rover = tracer_instantane
        carte.restaurer_rover = tracer_restauration

    def ajouter_chemin(self, chemin):
        self.chemins.append(simplifier(chemin))
        return len(self.chemins) - 1

    def evenement(self, type_, action=None, **infos):
        simu = self.simu
        est, vraie = r3(simu.est), r3(simu.vraie)
        if not self.traj or self.traj[-1] != est or self.traj_vraie[-1] != vraie:
            self.traj.append(est)
            self.traj_vraie.append(vraie)
        titre, texte = textes(type_, simu, self.carte, infos, longueur(self.traj))
        if infos.get('chemin'):
            self.chemin_id = self.ajouter_chemin(infos['chemin'])
        f = {
            'type': type_,
            'titre': titre,
            'texte': texte,
            'action': action,
            'msgs': self.attente,
            'pos': est,
            'vraie': vraie,
            'sigma': round(simu.sigma, 3),
            'liaison': simu.connecte(),
            'cap': round(simu.cap, 3),
            't': round(simu.t, 1),
            'chemin': self.chemin_id,
            'traj': len(self.traj),
        }
        if infos.get('optimiste'):
            f['optimiste'] = self.ajouter_chemin(infos['optimiste'])
        for cle in ('obstacle', 'point', 'cible', 'point_de_vue'):
            if infos.get(cle) is not None:
                f[cle] = r3(infos[cle])
        if 'tuiles' in infos:
            f['tuiles'] = [list(t) for t in infos['tuiles']]
        if 'restant' in infos:
            f['restant'] = round(infos['restant'], 2)
        self.frames.append(f)
        self.attente = []


def resume(tr):
    types = [f['type'] for f in tr.frames]
    return {
        'replans': types.count('replan'),
        'chocs': types.count('choc'),
        'zooms': types.count('zoom'),
        'analyses': types.count('analyse'),
        'distance': round(longueur(tr.traj), 1),
        'duree': tr.frames[-1]['t'],
        'issue': tr.frames[-1]['type'],
    }


def exporter(terrain, monde, champ, occlusion=True):
    carte, simu, depart, victime = monde()
    simu.occlusion, simu.champ = occlusion, champ
    tr = Traceur(carte, simu)
    simu.placer(depart)
    simu.survol_initial(z=0)
    tr.evenement('survol')
    mission(simu, depart, victime, journal=lambda _: None, evenement=tr.evenement)
    g = carte.grille(*ZONE)
    vu = round(float((g.src == SRC_ROVER).sum()) * carte.res_fine**2)
    return {
        'terrain': terrain,
        'champ': champ,
        'rayons_x': not occlusion,
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
        'traj': tr.traj,
        'traj_vraie': tr.traj_vraie,
        'frames': tr.frames,
        'resume': {**resume(tr), 'vu': vu},
        # pour verifier que la page refait la meme fusion que Python
        'controle': {
            'obstacle': int((g.val == OBSTACLE).sum()),
            'libre': int((g.val == LIBRE).sum()),
            'inconnu': int((g.val == INCONNU).sum()),
            'rover': int((g.src == SRC_ROVER).sum()),
        },
    }


TOUS = [60, 90, 120, 160, 240, 360, 'X']  # 'X' = vision a rayons X, pour comparer
SANS_DRONE = [90, 160, 360]
TERRAINS = {
    'crete': {
        'nom': 'Crête et tronc caché',
        'description': 'Une crête barre la route, avec un passage de 70 cm que le drone ne voit '
        'pas de haut. Un tronc couché sous les arbres est invisible depuis le ciel.',
        'monde': monde_1,
        'champs': TOUS,
    },
    'foret': {
        'nom': 'Forêt',
        'description': 'Le drone voit les falaises qui bordent la forêt, mais rien sous les '
        'arbres : un slalom de troncs couchés et des rochers. Le rover doit tout découvrir '
        'avec sa caméra.',
        'monde': monde_3,
        'champs': TOUS,
    },
    'cratere': {
        'nom': 'Cratère fermé',
        'description': "La victime est au fond d'un cratère complètement fermé. Aucun chemin "
        "n'existe : le rover doit s'approcher au plus près.",
        'monde': monde_2,
        'champs': TOUS,
    },
    'coupure': {
        'nom': 'Drone perdu 110 s',
        'description': 'Même crête, mais la liaison avec le drone est coupée juste après son '
        "premier survol, et ne revient qu'au bout de 110 s. Le rover fait la reconnaissance "
        'lui-même, prudemment, puis le drone recale sa position.',
        'monde': monde_4,
        'champs': SANS_DRONE,
    },
    'perdu': {
        'nom': 'Drone perdu pour toujours',
        'description': 'Même crête, et le drone ne revient jamais. Le rover longe la crête et '
        "l'inspecte lui-même jusqu'à trouver le passage, alors que l'incertitude sur sa "
        'position ne cesse de grandir.',
        'monde': lambda: monde_4(((0.0, None),)),
        'champs': SANS_DRONE,
    },
    'passage': {
        'nom': 'Sans drone, passage proche',
        'description': 'Le passage dans la crête est plus près (x = 9,3 à 10 m), toujours '
        'invisible depuis le ciel, et le drone ne revient jamais.',
        'monde': monde_5,
        'champs': SANS_DRONE,
    },
}


def main():
    scenarios = []
    for cle, t in TERRAINS.items():
        for champ in t['champs']:
            if champ == 'X':
                scenarios.append(exporter(cle, t['monde'], 360, occlusion=False))
            else:
                scenarios.append(exporter(cle, t['monde'], champ))
            print(cle, champ, scenarios[-1]['resume'], flush=True)
    terrains = {k: {'nom': t['nom'], 'description': t['description']} for k, t in TERRAINS.items()}
    trace = {'terrains': terrains, 'scenarios': scenarios}
    sortie = ICI / 'trace.js'
    texte = 'window.TRACE = ' + json.dumps(trace, ensure_ascii=False, separators=(',', ':'))
    sortie.write_text(texte + ';' + chr(10), encoding='utf-8', newline=chr(10))
    print(f'{len(scenarios)} simulations -> {sortie} : {sortie.stat().st_size / 1e6:.2f} Mo')


if __name__ == '__main__':
    main()
