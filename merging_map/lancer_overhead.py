"""
lancer_overhead.py - Mission sur le vrai RaspRover, avec la camera au-dessus de la scene
(`guidance_global_map.drone.OverheadDrone`) dans le role du drone.

    PYTHONPATH=merging_map:src/guidance_calib:src/guidance_global_map:src/guidance_rover_pose \\
        python merging_map/lancer_overhead.py --camera 0 --metres-par-pixel 0.0021 \\
        --victime 1.2 0.9 --rover http://<IP du rover>:8080 --dry-run

D'abord --dry-run : aucun POST ne part, le programme affiche la zone, le rover vu par la
camera, l'echelle, la premiere decision et les premiers pas, puis s'arrete. Ensuite la meme
commande sans --dry-run fait rouler le rover. Ctrl-C, une erreur ou un STOP de l'operateur
(page du rover, tableau de bord) arretent la mission et envoient POST /stop au rover.

Il faut guidance_global_map (src/), opencv-contrib-python et numpy, et sur le rover
rover/rover_server.py (POST /path). --camera a besoin de guidance_global_map.drone.camera_reader
(camera en direct) ; --image (une photo fixe) n'est accepte qu'avec --dry-run : une photo ne
voit pas le rover bouger.

Le repere (point 4 de docs/overhead-camera.md) : celui de la camera, origine au coin en bas a
gauche de l'image, x vers la droite, y vers le haut, en metres. Le depart, la victime et les
tuiles de la camera du rover doivent y etre exprimes. Il reste valable tant que la camera ne
bouge pas.

--suivi FICHIER ecrit apres chaque decision et chaque pas un document JSON (etat de la
mission, rover, chemin, trajectoire, journal, grille) pour une page de suivi en direct.
"""

import argparse
import collections
import json
import math
import os
import sys
import tempfile
import time

from cerveau import RESUMES, avancer, mission
from fusion_carte import INCONNU, OBSTACLE, SRC_ROVER, CarteFusion
from mission_settings import ARENA
from rover_reel import ArretOperateur, RaspRoverReel, ServeurRoverError, client_http

# Reglages pour une petite zone (point 3) : planification en cases de 5 cm, objectif atteint a
# 15 cm. Les valeurs par defaut (cases de 10 cm, arrivee a 50 cm) bloquaient 45 % du sol de la
# photo IMG_4099.
# rayon_robot est le rayon du cercle qui englobe tout le rover (fusion_carte) : pour un rover
# de 0.20 x 0.23 m (rover_size de la camera), hypot(0.20, 0.23) / 2 = 0.152 m, arrondi a
# 0.155. La doc de la camera donnait 0.115, la demi-longueur : un coin du rover pouvait alors
# toucher un obstacle en pivotant.
# Provisoire : a ajuster apres un essai sur le vrai robot.
RAYON_ROVER = 0.155
REGLAGES_PETITE_ZONE = {'niveau_plan': 4, 'rayon_robot': RAYON_ROVER}
TOLERANCE_PETITE_ZONE = 0.15
PAS_AFFICHES = 3  # --dry-run : pas affiches apres la premiere decision


def dans_la_zone(point, zone, marge=0.0):
    x0, y0, x1, y1 = zone
    return x0 - marge <= point[0] <= x1 + marge and y0 - marge <= point[1] <= y1 + marge


def preparer(camera, requete, percevoir=None, options_rover=None, **reglages_carte):
    """Carte remplie par le premier survol de la camera, et rover pret a partir.

    `camera` : un OverheadDrone (ou tout objet avec tiles(), disponible(), zoomer(),
    rover_pose() et zone()). La position du rover vient de `camera.rover_pose`, avec ou sans
    cap ; sans cap, le rover le mesure d'apres son mouvement. `requete` : le serveur du rover
    (rover_reel.client_http). `options_rover` : arguments de RaspRoverReel (vitesse,
    attente_image, horloge...)."""
    carte = CarteFusion(**{**REGLAGES_PETITE_ZONE, **reglages_carte})
    for message in camera.tiles():
        carte.recevoir(message)
    options = dict(options_rover or {})
    if hasattr(camera, 'refresh'):  # OverheadDrone : une image tout de suite, rover arrete
        options.setdefault('rafraichir_drone', lambda: camera.refresh(force=True))
    rover = RaspRoverReel(
        carte,
        requete,
        position_drone=camera.rover_pose,
        percevoir=percevoir or (lambda position, cap: None),
        drone=camera,
        **options,
    )
    return carte, rover


def arreter_le_rover(requete, essais=3, journal=print, dormir=time.sleep):
    """POST /stop, en insistant : c'est le dernier ordre de la mission. True s'il est recu."""
    for k in range(essais):
        try:
            if requete('POST', '/stop', {}) is not None:
                return True
        except Exception as e:  # noqa: BLE001 - un /stop refuse ne doit pas masquer la cause
            journal(f'  /stop refuse : {e}')
        if k + 1 < essais:
            dormir(0.3)
    journal('  ATTENTION : /stop sans reponse, le rover finit peut-etre son pas tout seul')
    return False


def sans_post(requete, journal=print):
    """--dry-run : les GET passent, aucun POST ne part (le rover ne peut pas bouger)."""

    def requete_lecture(methode, chemin, corps=None):
        if methode != 'GET':
            journal(f'  [dry-run] {methode} {chemin} {corps} NON envoye')
            raise RuntimeError(f'--dry-run : {methode} {chemin} ne doit pas partir')
        return requete(methode, chemin, corps)

    return requete_lecture


def decrire_echelle(camera):
    """L'echelle de la camera du dessus, en texte, pour la verifier avant de rouler."""
    reglages = getattr(camera, 'settings', None)
    if reglages is None:
        return 'echelle inconnue (camera sans settings)'
    mpp = getattr(reglages, 'metres_per_pixel', None)
    if mpp is not None:
        return f'{mpp * 1000:.3f} mm/pixel (--metres-par-pixel)'
    return (
        f'deduite de la hauteur {reglages.camera_height:.2f} m, focale 35 mm '
        f'{reglages.focal_length_35mm:g} mm, capteur {reglages.sensor_size[0]}x'
        f'{reglages.sensor_size[1]} px : a verifier sur la taille de la zone'
    )


class Suivi:
    """--suivi : un document JSON reecrit (atomiquement) apres chaque decision et chaque pas,
    pour une page de suivi en direct. Branche sur journal, evenement, rover.deplacer et
    rover.observer : cerveau.py n'en sait rien."""

    def __init__(self, fichier, carte, rover, victime, zone, journal=print):
        self.fichier, self.carte, self.rover = fichier, carte, rover
        self.victime, self.zone, self.journal_sortie = victime, zone, journal
        self.lignes = collections.deque(maxlen=30)
        self.decision, self.chemin, self.traj = None, [], []
        self.etat = 'en_cours'

    def journal(self, texte):
        self.lignes.append(str(texte))
        self.journal_sortie(texte)

    def evenement(self, type_, action, **infos):
        if type_ == 'avance':
            return
        self.decision = {'action': action or type_, 'raison': RESUMES.get(type_, type_)}
        if infos.get('chemin') is not None:
            self.chemin = [[float(x), float(y)] for x, y in infos['chemin']]
        if type_ == 'arrivee':
            self.etat = 'arrive'
        self.ecrire()

    def brancher(self):
        deplacer, observer = self.rover.deplacer, self.rover.observer

        def deplacer_et_ecrire(cible):
            try:
                return deplacer(cible)
            finally:
                self.traj.append([float(c) for c in self.rover.position()])
                self.ecrire()

        def observer_et_ecrire(tour_complet=False):
            observer(tour_complet)
            self.ecrire()

        self.rover.deplacer, self.rover.observer = deplacer_et_ecrire, observer_et_ecrire

    def grille(self):
        g = self.carte.grille(*self.zone)
        lignes = []
        for i in range(g.val.shape[0]):  # ligne 0 = sud, comme la grille de la carte
            caracteres = []
            for v, s in zip(g.val[i], g.src[i], strict=True):
                if v == INCONNU:
                    caracteres.append('?')
                elif v == OBSTACLE:
                    caracteres.append('2' if s == SRC_ROVER else '1')
                else:
                    caracteres.append('0')
            lignes.append(''.join(caracteres))
        return {'x0': g.j0 * g.res, 'y0': g.i0 * g.res, 'res': g.res, 'lignes': lignes}

    def document(self):
        x, y = self.rover.position()
        cap = self.rover.cap
        return {
            't': time.time(),
            'etat': self.etat,
            'rover': [float(x), float(y), None if cap is None else float(cap)],
            'victime': [float(c) for c in self.victime],
            'zone': [float(c) for c in self.zone],
            'decision': self.decision,
            'chemin': self.chemin,
            'traj': self.traj,
            'journal': list(self.lignes),
            'grille': self.grille(),
        }

    def ecrire(self, etat=None):
        if etat is not None:
            self.etat = etat
        if self.fichier is None:
            return
        dossier = os.path.dirname(os.path.abspath(self.fichier))
        # ecrire a cote puis renommer : la page ne lit jamais un fichier a moitie ecrit
        fd, tmp = tempfile.mkstemp(dir=dossier, prefix='.suivi-', suffix='.json')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.document(), f)
            os.replace(tmp, self.fichier)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def lancer(
    camera,
    requete,
    victime,
    depart=None,
    percevoir=None,
    tolerance=TOLERANCE_PETITE_ZONE,
    journal=print,
    options_rover=None,
    settings=ARENA,
    suivi=None,
    dry_run=False,
    **reglages_carte,
):
    """Lance la mission. `depart` par defaut : la ou la camera voit le rover.
    `settings`: the mission's distances, ARENA by default (mission_settings).
    `suivi` : fichier JSON de suivi (voir Suivi). `dry_run` : `requete` ne doit envoyer aucun
    POST (voir sans_post) ; on s'arrete apres la premiere decision, sans rouler.

    Renvoie les instantanes de cerveau.mission, plus 'etat' : 'arrive', 'abandon',
    'arret_operateur' (avec 'raison' et 'fin' = {'traj'}) ou 'dry_run' (avec 'decision').
    POST /stop est envoye en sortie dans tous les cas (sauf dry_run), meme sur Ctrl-C."""
    etat_rover = requete('GET', '/state')
    if etat_rover is None and not dry_run:
        raise RuntimeError(
            'Le rover ne repond pas a GET /state : verifier --rover (adresse, port 8080), que '
            "rover_server.py tourne et que le reseau passe. Rien n'a ete envoye."
        )
    if etat_rover is not None:
        odom = etat_rover.get('odom') or {}
        journal(
            f'Rover joint : liaison ESP32 {(etat_rover.get("link") or {}).get("connected")}, '
            f'gyroscope pret {odom.get("gyro_ready")}, boot_id {etat_rover.get("boot_id")}'
        )
    else:
        journal('Rover non joint (dry-run) : seule la camera est verifiee.')
    zone = camera.zone()
    if zone is None:
        raise RuntimeError("La camera ne donne pas d'image : impossible de connaitre la zone.")
    pose = camera.rover_pose()
    journal(
        f'Zone vue par la camera : {zone[2] - zone[0]:.2f} x {zone[3] - zone[1]:.2f} m '
        f'(x {zone[0]:.2f}..{zone[2]:.2f}, y {zone[1]:.2f}..{zone[3]:.2f}) ; '
        f'echelle {decrire_echelle(camera)}'
    )
    if pose is None:
        journal('Rover vu par la camera : NON')
    else:
        cap = 'inconnu' if pose[2] is None else f'{math.degrees(pose[2]):.0f} deg'
        journal(f'Rover vu par la camera : ({pose[0]:.2f}, {pose[1]:.2f}) m, cap {cap}')
    if depart is None:
        if pose is None:
            raise RuntimeError('La camera ne voit pas le rover : donne --depart x y.')
        depart = (pose[0], pose[1])
    for nom, point in (('depart', depart), ('victime', victime)):
        if not dans_la_zone(point, zone):
            raise ValueError(
                f'{nom} {point} hors de la zone vue par la camera {zone}. Repere attendu : '
                "origine au coin en bas a gauche de l'image, x vers la droite, y vers le "
                'haut, en metres.'
            )
    options = dict(options_rover or {})
    options.setdefault('signaler', lambda texte: journal(f'  {texte}'))
    carte, rover = preparer(camera, requete, percevoir, options, **reglages_carte)
    suivi_ = Suivi(suivi, carte, rover, victime, zone, journal)
    rover.placer(depart)  # pour le suivi ; mission() le refait
    if dry_run:
        return _premiere_decision(carte, rover, depart, victime, zone, tolerance, settings, suivi_)
    suivi_.brancher()
    try:
        instantanes = mission(
            rover,
            carte,
            depart,
            victime,
            zone=zone,
            tolerance=tolerance,
            journal=suivi_.journal,
            evenement=suivi_.evenement,
            settings=settings,
        )
    except ArretOperateur as e:
        suivi_.journal(f'ARRET OPERATEUR : {e} -> mission terminee, envoi de /stop')
        suivi_.ecrire('arret_operateur')
        return {'etat': 'arret_operateur', 'raison': str(e), 'fin': {'traj': suivi_.traj}}
    except BaseException as e:
        try:  # le suivi ne doit pas masquer la vraie cause
            suivi_.journal(f'MISSION INTERROMPUE : {type(e).__name__} {e}')
            suivi_.ecrire('abandon' if isinstance(e, KeyboardInterrupt) else 'erreur')
        except Exception:  # noqa: BLE001, S110
            pass
        raise
    finally:
        arreter_le_rover(requete, journal=journal)
    etat = 'arrive' if suivi_.etat == 'arrive' else 'abandon'
    suivi_.ecrire(etat)
    instantanes['etat'] = etat
    return instantanes


def _premiere_decision(carte, rover, depart, victime, zone, tolerance, settings, suivi):
    """--dry-run : la decision que prendrait le cerveau au depart, et ses premiers pas."""
    settings.apply_to_map(carte)
    dec = carte.decider(
        depart, victime, zone=zone, tolerance=tolerance, drone_disponible=rover.drone_disponible()
    )
    chemin = [tuple(p) for p in dec.get('chemin') or []]
    suivi.decision = {'action': dec['action'], 'raison': 'dry-run : premiere decision'}
    suivi.chemin = [[float(x), float(y)] for x, y in chemin]
    longueur = sum(math.dist(chemin[k], chemin[k + 1]) for k in range(len(chemin) - 1))
    suivi.journal(f'[dry-run] decision : {dec["action"]}, chemin de {longueur:.2f} m')
    if not rover.cap_fiable and (rover.position_drone() or (0, 0, None))[2] is None:
        suivi.journal(
            f"[dry-run] cap du rover inconnu : il avancera d'abord tout droit de "
            f'{rover.pas_cap:.2f} m pour le mesurer (marqueur ArUco : --marqueur)'
        )
    pos, reste = tuple(depart), list(chemin)
    for k in range(PAS_AFFICHES):
        if not reste:
            break
        cible, reste = avancer(pos, reste, settings.step)
        suivi.journal(
            f'[dry-run] pas {k + 1} : ({pos[0]:.2f}, {pos[1]:.2f}) -> '
            f'({cible[0]:.2f}, {cible[1]:.2f}) m, {math.dist(pos, cible):.2f} m'
        )
        pos = cible
    suivi.journal("[dry-run] arret apres la premiere decision : le rover n'a pas bouge")
    suivi.ecrire('en_cours')
    return {'etat': 'dry_run', 'decision': dec}


def _lecteur_camera(source):
    try:
        from guidance_global_map.drone import camera_reader
    except ImportError as e:
        sys.exit(
            f'--camera : {e}\nIl faut guidance_global_map.drone.camera_reader (camera en '
            'direct, src/guidance_global_map). Sans lui : --image photo.jpg --dry-run.'
        )
    return camera_reader(source)


def _percevoir(url):
    try:
        from perception_rover import percevoir_http
    except ImportError as e:
        sys.exit(f'--perception : {e}\nIl faut merging_map/perception_rover.py (percevoir_http).')
    return percevoir_http(url)


def analyser(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--camera',
        help='camera du dessus en direct : indice local (0), URL http(s) MJPEG ou snapshot, '
        'rtsp://, ou fichier (guidance_global_map.drone.camera_reader)',
    )
    source.add_argument(
        '--image',
        help='photo fixe de la camera du dessus : seulement avec --dry-run (une photo ne voit '
        'pas le rover bouger)',
    )
    p.add_argument(
        '--rover',
        help='serveur du rover (rover/rover_server.py), par exemple http://192.168.1.42:8080 ; '
        'facultatif avec --dry-run',
    )
    p.add_argument('--victime', type=float, nargs=2, required=True, metavar=('X', 'Y'))
    p.add_argument('--depart', type=float, nargs=2, metavar=('X', 'Y'))
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='aucun POST au rover : affiche zone, rover, echelle, premiere decision et '
        "premiers pas, puis s'arrete",
    )
    p.add_argument('--hauteur', type=float, default=1.9, help='hauteur de la camera (m)')
    p.add_argument(
        '--metres-par-pixel',
        type=float,
        help="taille au sol d'un pixel (m), mesuree a la regle sur l'image : prioritaire sur "
        '--hauteur/--focale-35mm/--capteur',
    )
    p.add_argument('--focale-35mm', type=float, help='focale equivalente 35 mm (mm)')
    p.add_argument('--capteur', type=int, nargs=2, metavar=('L', 'H'), help='capteur (pixels)')
    p.add_argument('--marqueur', type=int, help='id ArUco sur le rover, pour avoir son cap')
    p.add_argument('--perception', help='service de perception du rover (percevoir_http)')
    p.add_argument('--suivi', help='fichier JSON de suivi, reecrit a chaque decision et pas')
    p.add_argument(
        '--tolerance', type=float, default=TOLERANCE_PETITE_ZONE, help='arrivee a moins de (m)'
    )
    p.add_argument('--vitesse', type=float, help='vitesse de croisiere du rover (m/s, 0.2)')
    p.add_argument(
        '--delai-http', type=float, default=1.0, help="delai d'une requete au rover (s)"
    )
    args = p.parse_args(argv)
    if args.image is not None and not args.dry_run:
        p.error('--image (photo fixe) seulement avec --dry-run : il faut --camera pour rouler')
    if args.rover is None and not args.dry_run:
        p.error('--rover est obligatoire sans --dry-run')
    return args


def reglages_camera(args):
    reglages = {'camera_height': args.hauteur}
    if args.metres_par_pixel is not None:
        reglages['metres_per_pixel'] = args.metres_par_pixel
    if args.focale_35mm is not None:
        reglages['focal_length_35mm'] = args.focale_35mm
    if args.capteur is not None:
        reglages['sensor_size'] = tuple(args.capteur)
    if args.marqueur is not None:
        reglages['rover_marker_id'] = args.marqueur
    return reglages


def main(argv=None):
    args = analyser(argv)
    try:
        from guidance_global_map.drone import OverheadDrone, image_file_reader
        from guidance_global_map.pipeline import OverheadSettings
    except ImportError as e:
        sys.exit(
            f'{e}\nIl faut guidance_global_map sur le PYTHONPATH (src/guidance_calib, '
            'src/guidance_global_map, src/guidance_rover_pose) et opencv-contrib-python.'
        )
    lecteur = _lecteur_camera(args.camera) if args.camera else image_file_reader(args.image)
    percevoir = _percevoir(args.perception) if args.perception else None
    if args.rover:
        requete = client_http(args.rover, delai=args.delai_http)
    else:
        requete = lambda methode, chemin, corps=None: None  # noqa: E731 - dry-run sans rover
    if args.dry_run:
        requete = sans_post(requete)
    options = {} if args.vitesse is None else {'vitesse': args.vitesse}
    camera = OverheadDrone(lecteur, OverheadSettings(**reglages_camera(args)))
    try:
        resultat = lancer(
            camera,
            requete,
            tuple(args.victime),
            tuple(args.depart) if args.depart else None,
            percevoir=percevoir,
            tolerance=args.tolerance,
            options_rover=options,
            suivi=args.suivi,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        sys.exit('Interrompu (Ctrl-C) : /stop envoye au rover.')
    except (ServeurRoverError, RuntimeError, ValueError) as e:
        sys.exit(f'ERREUR : {e}')
    finally:
        fermer = getattr(lecteur, 'close', None)
        if fermer is not None:
            fermer()
    if resultat['etat'] == 'dry_run':
        return resultat
    traj = resultat['fin']['traj']
    distance = sum(math.dist(traj[k], traj[k + 1]) for k in range(len(traj) - 1))
    print(f'Fin de mission ({resultat["etat"]}) : {distance:.2f} m parcourus.')
    return resultat


if __name__ == '__main__':
    main()
