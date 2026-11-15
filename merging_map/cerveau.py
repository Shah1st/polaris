"""
cerveau.py - La mission, telle que l'ordinateur la fait tourner.

decider -> (zoom drone | reconnaissance par le rover | rouler + percevoir | replanifier),
avec la gestion des pertes de liaison (drone ou rover). Le cerveau ne parle qu'a une
interface robot.Robot : le meme code pilote le rover simule (demo_fusion.Simu) et le vrai
RaspRover (rover_reel.RaspRoverReel).
"""

import math

from mission_settings import FIELD

# The field values under their former names (the dashboard reads them). The distances of a
# mission now come from its `settings` (mission_settings.FIELD or ARENA).
PAS, PAS_PRUDENT = FIELD.step, FIELD.careful_step
BUDGET_INCERTITUDE = FIELD.uncertainty_budget
ATTENTE_MAX = FIELD.drone_wait_max
RECUL_RESEAU, RAYON_ZONE = FIELD.link_backoff, FIELD.no_go_radius

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
    'perte_rover': 'LIAISON ROVER PERDUE -> retour au dernier point connecte',
    'reprise_rover': 'liaison rover retrouvee -> zone interdite, replanification',
    'link_back': 'liaison rover retrouvee -> replanification, sans zone interdite',
    'erreur_robot': 'ERREUR du rover -> arret, nouvel essai',
    'rover_injoignable': 'abandon: rover injoignable -> arret et alerte',
    'sans_progres': "abandon: le rover n'avance plus -> arret et alerte",
    'max_tours': 'abandon: trop de tours sans arriver -> arret et alerte',
}

# Issue d'une mission (instantanes['issue']) -> etat pour une page de suivi (interface 6 :
# en_cours | arrive | arret_operateur | abandon | erreur). Seule 'arrivee' est un succes.
ETAT_SUIVI = {
    'arrivee': 'arrive',
    'objectif_degrade': 'abandon',  # point le plus proche atteint, victime inaccessible
    'abandon_drone': 'abandon',
    'rover_injoignable': 'abandon',
    'sans_progres': 'abandon',
    'max_tours': 'abandon',
    'erreur_robot': 'erreur',
}


class _FinMission(Exception):  # noqa: N818 - signal interne, pas une erreur
    """Fin anticipee de la mission depuis une fonction imbriquee (issue dans ETAT_SUIVI)."""

    def __init__(self, issue, raison=''):
        super().__init__(issue)
        self.issue, self.raison = issue, raison


def _noms_classes(erreur):
    return {c.__name__ for c in type(erreur).__mro__}


def erreur_rover_reessayable(erreur):
    """Une erreur du serveur du rover (rover_reel.ServeurRoverError) apres laquelle le rover
    est a l'arret : la mission peut l'arreter et reessayer. Reconnue par son nom, pour ne pas
    importer rover_reel (qui importe robot et la carte). Un arret de l'operateur
    (rover_reel.ArretOperateur) n'en est jamais une : il doit remonter tout de suite."""
    noms = _noms_classes(erreur)
    return 'ServeurRoverError' in noms and 'ArretOperateur' not in noms


def longueur(chemin):
    return sum(math.dist(chemin[k], chemin[k + 1]) for k in range(len(chemin) - 1))


def avancer(pos, chemin, pas):
    """Point atteint en parcourant `pas` m le long de `chemin` depuis `pos` (les points
    depasses sont retires du chemin)."""
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


def mission(
    robot,
    carte,
    depart,
    victime,
    zone=None,
    tolerance=0.5,
    max_tours=600,
    journal=print,
    evenement=None,
    settings=FIELD,
):
    """Mene le rover de `depart` a `victime`.

    Sans liaison drone : le rover roule deux fois plus lentement, sa position ne vient plus
    que de l'odometrie (incertitude qui grandit), les obstacles du drone sont gonfles d'autant,
    il fait la reconnaissance lui-meme, et revient au dernier point vu par le drone si
    l'incertitude depasse settings.uncertainty_budget (sauf si la victime est plus proche).
    Sans liaison rover : retour au dernier point connecte, zone interdite, replanification.

    `zone` : rectangle (x0, y0, x1, y1) ou planifier (defaut : zone survolee par le drone).
    `evenement(type, action, **infos)` est appele a chaque etape (tableau de bord).
    `settings` : distances and delays of the mission (mission_settings.FIELD, the default, or
    ARENA for the 2.4 m arena); the map's own inspection settings are set from it too.

    Renvoie des instantanes pour la figure de la demo, avec l'issue de la mission :
    instantanes['issue'] (une cle de ETAT_SUIVI), ['succes'] (True seulement pour
    'arrivee') et ['raison']. Une erreur du serveur du rover (ServeurRoverError) arrete le
    rover et fait reessayer, settings.robot_error_retries fois de suite au plus ; un arret de
    l'operateur (ArretOperateur) remonte tout de suite a l'appelant."""
    settings.apply_to_map(carte)
    instantanes, demandees, replans = {}, [], []
    traj, traj_vraie = [], []
    essais = {}
    etat = {'drone': True}
    # Non-progres : positions apres les derniers pas commandes, serie de chocs au meme
    # endroit, erreurs du rover a la suite.
    suivi = {'pas': [], 'choc0': None, 'chocs': 0, 'erreurs': 0}

    def emettre(type_, action=None, **infos):
        if evenement:
            evenement(type_, action, **infos)
        if type_ != 'avance':
            detail = f' {infos["tuiles"]}' if 'tuiles' in infos else ''
            if 'message' in infos:
                detail += f' : {infos["message"]}'
            journal(f'  t={robot.temps():5.1f}s  {RESUMES[type_]}{detail}')

    def abandonner(issue, action, raison='', **infos):
        emettre(issue, action, message=raison, **infos)
        raise _FinMission(issue, raison)

    def noter_position():
        traj.append(robot.position())
        traj_vraie.append(robot.position_reelle() or robot.position())

    def surveiller_progres(avant, cible, impact, action):
        """Arrete la mission si le rover ne bouge plus de plus de settings.stall_distance m
        sur settings.stall_steps pas commandes a la suite, ou s'il se cogne
        settings.same_bump_max fois au meme endroit (faux 'stalled', roues bloquees)."""
        ici = robot.position()
        if math.dist(avant, cible) > settings.stall_distance:
            suivi['pas'] = (suivi['pas'] + [ici])[-(settings.stall_steps + 1) :]
            fenetre = suivi['pas']
            if len(fenetre) > settings.stall_steps and all(
                math.dist(fenetre[0], p) <= settings.stall_distance for p in fenetre[1:]
            ):
                abandonner(
                    'sans_progres',
                    action,
                    f'immobile (< {settings.stall_distance:.2f} m) sur '
                    f'{settings.stall_steps} pas',
                )
        if impact is None:
            if suivi['choc0'] and math.dist(ici, suivi['choc0']) > 2 * settings.same_bump_radius:
                suivi['choc0'], suivi['chocs'] = None, 0
            return
        if suivi['choc0'] and math.dist(impact, suivi['choc0']) <= settings.same_bump_radius:
            suivi['chocs'] += 1
        else:
            suivi['choc0'], suivi['chocs'] = tuple(impact), 1
        if suivi['chocs'] >= settings.same_bump_max:
            abandonner('sans_progres', action, f'{suivi["chocs"]} chocs au meme endroit')

    def deplacer(cible, action):
        """robot.deplacer, avec reprise apres une erreur du serveur du rover. Renvoie
        (impact, erreur)."""
        try:
            impact = robot.deplacer(cible)
        except Exception as e:
            if not erreur_rover_reessayable(e):
                raise
            suivi['erreurs'] += 1
            if suivi['erreurs'] > settings.robot_error_retries:
                abandonner('erreur_robot', action, str(e))
            emettre('erreur_robot', action, message=str(e))
            try:
                robot.attendre(1.0)  # arret du rover, puis nouvel essai
            except Exception as e2:
                if not erreur_rover_reessayable(e2):
                    raise
            return None, True
        suivi['erreurs'] = 0
        return impact, False

    def decider():
        carte.expirer_chocs(robot.temps())
        carte.marge_drone = 0.0 if robot.drone_disponible() else robot.incertitude()
        return carte.decider(
            robot.position(),
            victime,
            zone=zone,
            tolerance=tolerance,
            drone_disponible=robot.drone_disponible(),
        )

    def suivre_liaison():
        """Detecte une coupure ou un retour du drone. Renvoie True si l'etat a change."""
        connecte = robot.drone_disponible()
        if connecte == etat['drone']:
            return False
        etat['drone'] = connecte
        if connecte:
            erreur = robot.recaler()
            carte.marge_drone = 0.0
            noter_position()
            emettre('reconnexion', erreur=erreur)
        else:
            emettre('coupure')
        return True

    def rouler(
        chemin,
        arrive,
        action,
        verifier=True,
        budget=True,
        finir_sur_place=True,
        surveiller_rover=True,
        **infos,
    ):
        """Suit le chemin pas a pas. Renvoie 'arrive', 'bloque', 'choc', 'budget',
        'liaison' (drone), 'perte_rover' ou 'erreur' (erreur du rover, a replanifier)."""
        chemin = list(chemin)
        while chemin:
            if suivre_liaison():
                return 'liaison'
            if (
                budget
                and not robot.drone_disponible()
                and robot.incertitude() > settings.uncertainty_budget
            ):
                return 'budget'
            pas = settings.step if robot.drone_disponible() else settings.careful_step
            avant = robot.position()
            cible, chemin = avancer(avant, chemin, pas)
            impact, erreur = deplacer(cible, action)
            if erreur:
                return 'erreur'
            noter_position()
            if surveiller_rover and not robot.rover_joignable():
                robot.garder_a_bord()  # l'ordinateur ne l'entend plus : il garde ce qu'il voit
                robot.observer()
                secours_rover()
                return 'perte_rover'
            if surveiller_rover:
                surveiller_progres(avant, cible, impact, action)
            robot.observer()
            if impact:
                robot.signaler_choc(impact)
                replans.append(robot.position())
                emettre('choc', action, obstacle=impact, **infos)
                return 'choc'
            if arrive(robot.position()) or (finir_sur_place and not chemin):
                return 'arrive'
            if verifier:
                bloque = carte.premier_point_bloque(
                    [robot.position()] + chemin, zone, horizon=settings.check_horizon
                )
                if bloque:
                    replans.append(robot.position())
                    emettre('replan', action, obstacle=bloque, **infos)
                    return 'bloque'
            emettre('avance', action, **infos)
        return 'arrive'

    def revenir_et_attendre():
        """Incertitude trop grande : retour sur sa propre trace jusqu'au dernier point vu par
        le drone (le seul chemin dont on sait qu'il passe), puis attente du drone."""
        but, trace = robot.trace_depuis_recalage()
        trace = list(reversed(trace))
        emettre('retour', 'RETOUR', chemin=trace, point=but, sigma=robot.incertitude())
        r = rouler(
            trace, lambda q: math.dist(q, trace[-1]) < 0.02, 'RETOUR', False, False, point=but
        )
        if r == 'liaison':
            return None
        return attendre_drone('RETOUR', but)

    def attendre_drone(action, point):
        """Immobile, LED allumees, le rover attend le retour du drone (au plus
        settings.drone_wait_max)."""
        emettre('attente', action, point=point, sigma=robot.incertitude())
        attendu = 0.0
        while not robot.drone_disponible() and attendu < settings.drone_wait_max:
            robot.attendre(5.0)
            attendu += 5.0
            if not robot.drone_disponible():
                emettre('avance', action, point=point)
        if not suivre_liaison():
            emettre('abandon', action, point=point)
            return 'abandon'
        return None

    def secours_rover():
        """Plus d'ordre de l'ordinateur depuis 1 s. Le programme de secours embarque sur le
        rover enregistre l'instant et le dernier point ou la liaison marchait, puis le ramene
        sur sa propre trace jusqu'a settings.link_backoff m avant ce point. La liaison
        revient : le rover renvoie ce qu'il a vu, l'ordinateur marque la zone perdue comme
        interdite (une zone de settings.no_go_radius m devant ce point ; aucune si c'est 0) et
        replanifie avec la carte globale et le drone. Pendant tout ce temps, le drone continue
        de voir le rover."""
        instant = robot.temps()
        dernier_ok = traj[-2] if len(traj) >= 2 else traj[-1]
        ici = robot.position()
        dx, dy = ici[0] - dernier_ok[0], ici[1] - dernier_ok[1]
        norme = math.hypot(dx, dy) or 1.0
        emettre('perte_rover', 'SECOURS', point=dernier_ok, instant=instant)
        points = list(reversed(traj[:-1]))
        chemin, d = [points[0]], 0.0
        for a, b in zip(points, points[1:], strict=False):
            if d >= settings.link_backoff:
                break
            chemin.append(b)
            d += math.dist(a, b)
        if robot.retour_embarque:
            # le vrai rover revient tout seul : l'ordinateur ne peut que l'attendre, pendant
            # un temps borne (delai de l'adaptateur), puis la mission s'arrete
            if not robot.attendre_retour_embarque(chemin):
                abandonner('rover_injoignable', 'SECOURS', 'pas de reponse du rover', point=ici)
            noter_position()
        else:
            rouler(
                chemin,
                lambda q: math.dist(q, chemin[-1]) < 0.02,
                'SECOURS',
                verifier=False,
                budget=False,
                surveiller_rover=False,
                point=dernier_ok,
            )
        if not robot.rover_joignable():
            # jamais 'liaison retrouvee' si elle ne l'est pas
            abandonner('rover_injoignable', 'SECOURS', 'liaison pas revenue', point=ici)
        robot.vider_tampon()
        radius = settings.no_go_radius
        if radius <= 0:  # the link loss does not come from a place: just plan again
            emettre('link_back', 'SECOURS', point=dernier_ok, instant=instant)
            return
        zone_interdite = (
            dernier_ok[0] + dx / norme * radius,
            dernier_ok[1] + dy / norme * radius,
            radius,
        )
        carte.interdire_zone(*zone_interdite)
        emettre('reprise_rover', 'SECOURS', point=dernier_ok, zone=zone_interdite, instant=instant)

    def fin(issue, raison=''):
        instantanes['fin'] = {
            'grille': carte.grille(*zone) if zone else None,
            'traj': traj,
            'traj_vraie': traj_vraie,
            'replans': replans,
            'demandees': demandees,
        }
        instantanes['issue'], instantanes['raison'] = issue, raison
        instantanes['succes'] = issue == 'arrivee'
        return instantanes

    def boucle(dec):
        """La mission elle-meme ; renvoie (issue, raison)."""
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
                    {
                        'grille': dec['grille'],
                        'tuiles': tuiles,
                        'optimiste': dec['chemin_optimiste'],
                    },
                )
                emettre('blocage', action, tuiles=tuiles, optimiste=dec['chemin_optimiste'])
                demandees.extend(tuiles)
                for k in tuiles:
                    robot.demander_zoom(k)
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
                r = rouler(
                    dec['chemin'], lambda q, vue=vue: math.dist(q, vue) < 0.05, action, **infos
                )
                # deja assez pres pour bien voir la cible : on s'arrete et on regarde
                assez_pres = math.dist(robot.position(), cible) <= robot.portee_camera - 0.4
                if r == 'arrive' or (r in ('bloque', 'choc') and assez_pres):
                    robot.attendre(2.0)  # le temps de faire tourner la tourelle
                    robot.observer(tour_complet=True)
                    carte.inspectees.add(secteur)
                    emettre('analyse', action, **infos)
                elif essais[secteur] >= 3:
                    carte.inspectees.add(secteur)  # impossible d'y arriver : on passe
                if r == 'budget' and revenir_et_attendre() == 'abandon':
                    return 'abandon_drone', 'drone toujours absent'
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
                if r == 'arrive' and robot.drone_disponible():
                    emettre('fin', action, point=point, restant=restant)
                    return 'objectif_degrade', f'point le plus proche, a {restant:.2f} m'
                if r == 'arrive':
                    # sans drone, rien n'est definitif : on attend qu'il revienne verifier
                    if attendre_drone(action, point) == 'abandon':
                        return 'abandon_drone', 'drone toujours absent'
                    dec = decider()
                    continue
                if r == 'budget' and revenir_et_attendre() == 'abandon':
                    return 'abandon_drone', 'drone toujours absent'
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
                demandees.extend(tuiles)
                carte.demandes.update(tuiles)
                for k in tuiles:
                    robot.demander_zoom(k)
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
                retour = longueur(robot.trace_depuis_recalage()[1])
                if suite['action'] == 'SUIVRE' and longueur(suite['chemin']) < retour:
                    emettre('poursuite', action, chemin=suite['chemin'], sigma=robot.incertitude())
                    r = rouler(
                        suite['chemin'],
                        lambda q: math.dist(q, victime) <= tolerance,
                        action,
                        budget=False,
                    )
                elif revenir_et_attendre() == 'abandon':
                    return 'abandon_drone', 'drone toujours absent'
            if r == 'arrive':
                emettre('arrivee', action, distance=math.dist(robot.position(), victime))
                return 'arrivee', f'a {math.dist(robot.position(), victime):.2f} m'
            dec = decider()
        abandonner('max_tours', None, f'{max_tours} tours')

    robot.placer(depart)
    noter_position()
    robot.observer()
    emettre('capteur')
    try:
        return fin(*boucle(decider()))
    except _FinMission as f:
        return fin(f.issue, f.raison)
