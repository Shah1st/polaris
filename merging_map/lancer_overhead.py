"""
lancer_overhead.py - Mission sur le vrai RaspRover, avec la camera au-dessus de la scene
(branche overhead-camera, `guidance_global_map.drone.OverheadDrone`) dans le role du drone.

    PYTHONPATH=merging_map:src/guidance_calib:src/guidance_global_map:src/guidance_rover_pose \\
        python merging_map/lancer_overhead.py --camera http://<IP du telephone>:8080/video \\
        --metres-par-pixel 0.0012 --victime 1.2 0.9 --rover http://<IP du rover>:8080

--camera : la camera en direct (flux d'une application camera IP, RTSP, ou numero de
peripherique). Son echelle n'est pas celle d'une photo : il faut --metres-par-pixel, mesure sur
une image de ce flux (un objet de longueur connue pose au sol). --image a la place : une photo
fixe, pour essayer sans camera ; elle ne voit pas le rover bouger.

--image ... --photo-unique : une seule photo du dessus, prise avec le rover a son depart. Elle
donne la carte des obstacles, la zone et la position de depart ; ensuite le rover avance sur
son odometrie (roues + gyroscope, a calibrer : rover/README.md) et le cerveau gonfle les
obstacles de l'incertitude qui grandit. Le cap de depart ne se voit pas sur une photo (le rover
est presque carre) : --cap (rover pose face a une direction connue de la photo) ou --marqueur.
L'echelle : --largeur-photo (photo recadree sur une zone de largeur connue) ou
--metres-par-pixel.

    PYTHONPATH=merging_map:src/guidance_calib:src/guidance_global_map:src/guidance_rover_pose \\
        python merging_map/lancer_overhead.py --image parcours.jpg --photo-unique --cap 90 \\
        --largeur-photo 2.4 --victime 2.0 1.6 --rover http://<IP du rover>:8080

Il faut les modules de la branche overhead-camera, avec opencv-contrib-python et numpy, et
sur le rover le serveur de la branche rover-teleop (rover/rover_server.py, POST /path).

Le repere (point 4 de docs/overhead-camera.md) : celui de la camera, origine au coin en bas a
gauche de l'image, x vers la droite, y vers le haut, en metres. Le depart, la victime et les
tuiles de la camera du rover doivent y etre exprimes. Il reste valable tant que la camera ne
bouge pas.
"""

import argparse
import dataclasses
import math
import sys
import time

from cerveau import mission
from fusion_carte import CarteFusion
from mission_settings import ARENA
from rover_reel import RaspRoverReel, client_http

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
# Photo unique : le cerveau attend le drone (settings.drone_wait_max, 300 s) quand il ne peut
# plus avancer sans lui ; aucune nouvelle image ne viendra, on n'attend donc que quelques s.
PHOTO_UNIQUE_ATTENTE = 5.0


class PhotoUnique:
    """Une seule photo du dessus a la place du drone.

    Elle sert une fois : la carte et la zone (camera.tiles(), zone()), puis la position de
    depart du rover, donnee avec son cap a la premiere mesure du rover (rover arrete a son
    depart). Ensuite plus aucune image : disponible() est faux, le rover avance sur son
    odometrie et le cerveau fait des pas prudents et gonfle les obstacles de l'incertitude.
    Relire la photo apres un pas ferait croire au rover qu'il n'a pas bouge (faux chocs).

    `camera` : un OverheadDrone sur la photo. `cap` : cap du rover dans le repere de la photo
    (rad, depuis l'axe x vers la droite, sens trigonometrique), ou None pour celui du marqueur.
    """

    def __init__(self, camera, cap=None):
        self.camera = camera
        self.cap = cap
        self.mesures = 0  # images demandees par le rover (rafraichir_drone)

    def prise(self):
        """rafraichir_drone de RaspRoverReel : appele avant chaque mesure, rover arrete."""
        self.mesures += 1

    @property
    def epuisee(self):
        return self.mesures > 1

    def disponible(self):
        return not self.epuisee and self.camera.disponible()

    def rover_pose(self):
        if self.epuisee:
            return None
        pose = self.camera.rover_pose()
        if pose is None:
            return None
        return (pose[0], pose[1], self.cap if self.cap is not None else pose[2])

    def tiles(self, force=False):
        return self.camera.tiles(force)

    def zone(self):
        return self.camera.zone()

    def zoomer(self, tuile):
        return [] if self.epuisee else self.camera.zoomer(tuile)


def verifier_photo(camera, reglages, hauteur_rover=0.18, journal=print):
    """Ce que la photo donne, pour verifier l'echelle avant de partir. Le rover est detecte
    comme la tache de sa taille (0,20 x 0,23 m) : s'il n'est pas trouve, l'echelle est
    probablement fausse. Vu de haut il parait un peu plus grand que son empreinte au sol (son
    dessus est plus pres de la camera), d'environ hauteur / (hauteur - hauteur_rover)."""
    resultat = camera.refresh(force=True)
    if resultat is None:
        raise RuntimeError('Photo illisible.')
    b = resultat.bounds
    journal(
        f'Photo : {b.x_max - b.x_min:.2f} x {b.y_max - b.y_min:.2f} m, '
        f'{reglages.scale() * 1000:.2f} mm par pixel.'
    )
    if resultat.rover_position is None:
        raise RuntimeError(
            'Rover introuvable sur la photo : aucune tache de 0,20 x 0,23 m (+-20 %). '
            "L'echelle est sans doute fausse (--largeur-photo / --metres-par-pixel), ou le "
            'rover se confond avec le sol.'
        )
    taches = [t for t in resultat.blobs if t.mask is not None and resultat.rover_mask is not None]
    vu = next((t.size for t in taches if (t.mask & resultat.rover_mask).any()), None)
    x, y = resultat.rover_position
    message = f'Rover vu en ({x:.2f}, {y:.2f}) m'
    if vu is not None:
        attendu = sorted(reglages.rover_size)
        h = reglages.camera_height
        perspective = h / (h - hauteur_rover) if h > hauteur_rover else 1.0
        rapport = (vu[0] / attendu[0] + vu[1] / attendu[1]) / 2
        message += (
            f', taille {vu[0] * 100:.0f} x {vu[1] * 100:.0f} cm pour {attendu[0] * 100:.0f} x '
            f'{attendu[1] * 100:.0f} cm ({(rapport - 1) * 100:+.0f} % ; la perspective en '
            f'explique environ {(perspective - 1) * 100:+.0f} %)'
        )
    journal(message + '.')
    return resultat


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
    **reglages_carte,
):
    """Lance la mission. `depart` par defaut : la ou la camera voit le rover.
    `settings`: the mission's distances, ARENA by default (mission_settings)."""
    zone = camera.zone()
    if zone is None:
        raise RuntimeError("La camera ne donne pas d'image : impossible de connaitre la zone.")
    if depart is None:
        pose = camera.rover_pose()
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
    carte, rover = preparer(camera, requete, percevoir, options_rover, **reglages_carte)
    return mission(
        rover,
        carte,
        depart,
        victime,
        zone=zone,
        tolerance=tolerance,
        journal=journal,
        settings=settings,
    )


def reglages_camera(args):
    """Arguments de OverheadSettings d'apres la ligne de commande."""
    reglages = {'camera_height': args.hauteur}
    if args.metres_par_pixel is not None:
        reglages['metres_per_pixel'] = args.metres_par_pixel
    if args.marqueur is not None:
        reglages['rover_marker_id'] = args.marqueur
    return reglages


def rafraichir_apres(camera, latence, dormir):
    """Pour RaspRoverReel(rafraichir_drone=...) avec une camera en direct : son flux a du
    retard, donc on attend `latence` s avant de prendre l'image, pour qu'elle montre bien le
    rover arrete."""

    def rafraichir():
        dormir(latence)
        camera.refresh(force=True)

    return rafraichir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--camera',
        help='camera du dessus en direct : URL du flux (application camera IP du telephone, '
        'RTSP) ou numero de peripherique. Demande --metres-par-pixel',
    )
    source.add_argument(
        '--image',
        help='photo de la camera du dessus, relue a chaque image : une photo fixe ne voit pas '
        'le rover bouger, a garder pour les essais sans camera',
    )
    p.add_argument(
        '--metres-par-pixel',
        type=float,
        help="taille au sol d'un pixel de l'image, mesuree sur une image de la source. Sans "
        "elle, l'echelle vient de --hauteur et de l'objectif d'une photo d'iPhone",
    )
    p.add_argument(
        '--latence',
        type=float,
        default=0.5,
        help='retard du flux de --camera (s) : attente avant de regarder le rover arrete',
    )
    p.add_argument(
        '--rover',
        required=True,
        help='serveur du rover (rover/rover_server.py, branche rover-teleop), '
        'par exemple http://192.168.1.42:8080',
    )
    p.add_argument('--victime', type=float, nargs=2, required=True, metavar=('X', 'Y'))
    p.add_argument('--depart', type=float, nargs=2, metavar=('X', 'Y'))
    p.add_argument('--hauteur', type=float, default=1.9, help='hauteur de la camera (m)')
    p.add_argument('--marqueur', type=int, help='id ArUco sur le rover, pour avoir son cap')
    p.add_argument(
        '--photo-unique',
        action='store_true',
        help='avec --image : la photo ne sert qu au depart (carte + position), ensuite le rover '
        'avance sur son odometrie. Demande --cap ou --marqueur',
    )
    p.add_argument(
        '--cap',
        type=float,
        help='cap du rover au depart dans le repere de la photo, en degres depuis la droite de '
        "l'image, sens trigonometrique : 0 = face a droite, 90 = face au haut de la photo",
    )
    p.add_argument(
        '--largeur-photo',
        type=float,
        help="largeur reelle (m) de la zone couverte par la photo (photo recadree sur l'arene "
        'par exemple) : donne l echelle, a la place de --metres-par-pixel',
    )
    args = p.parse_args(argv)
    if args.photo_unique:
        if args.image is None:
            p.error('--photo-unique va avec --image.')
        if args.cap is None and args.marqueur is None:
            p.error(
                '--photo-unique demande --cap (ou --marqueur) : le cap de depart ne se voit pas '
                'sur une photo, et le rover ne peut plus le mesurer en bougeant.'
            )
    if args.largeur_photo is not None:
        if args.metres_par_pixel is not None:
            p.error('--largeur-photo ou --metres-par-pixel, pas les deux.')
        if args.image is None:
            p.error('--largeur-photo va avec --image.')
        import cv2

        photo = cv2.imread(args.image)
        if photo is None:
            sys.exit(f'Photo illisible : {args.image}')
        args.metres_par_pixel = args.largeur_photo / photo.shape[1]
    try:
        from guidance_global_map.drone import LiveCamera, OverheadDrone, image_file_reader
        from guidance_global_map.pipeline import OverheadSettings
    except ImportError as e:
        sys.exit(
            f'{e}\nIl faut les modules de la branche overhead-camera sur le PYTHONPATH '
            '(src/guidance_calib, src/guidance_global_map, src/guidance_rover_pose) et '
            'opencv-contrib-python.'
        )
    options_rover = {}
    if args.camera is not None:
        if args.metres_par_pixel is None:
            p.error(
                "--camera demande --metres-par-pixel : l'echelle d'un flux video n'est pas "
                "celle d'une photo."
            )
        source = int(args.camera) if args.camera.isdigit() else args.camera
        lire = LiveCamera(source)
        if not lire.wait_for_image(15.0):
            sys.exit(f'Aucune image de la camera {args.camera} en 15 s.')
    else:
        lire = image_file_reader(args.image)
    reglages = OverheadSettings(**reglages_camera(args))
    camera = OverheadDrone(lire, reglages)
    if args.camera is not None:
        options_rover['rafraichir_drone'] = rafraichir_apres(camera, args.latence, time.sleep)
    if args.photo_unique:
        try:
            verifier_photo(camera, reglages)
        except RuntimeError as e:
            sys.exit(str(e))
        camera = PhotoUnique(camera, None if args.cap is None else math.radians(args.cap))
        options_rover['rafraichir_drone'] = camera.prise
    settings = ARENA
    if args.photo_unique:  # no drone will ever come back: do not wait 5 minutes for it
        settings = dataclasses.replace(ARENA, drone_wait_max=PHOTO_UNIQUE_ATTENTE)
    instantanes = lancer(
        camera,
        client_http(args.rover),
        tuple(args.victime),
        tuple(args.depart) if args.depart else None,
        options_rover=options_rover,
        settings=settings,
    )
    traj = instantanes['fin']['traj']
    distance = sum(math.dist(traj[k], traj[k + 1]) for k in range(len(traj) - 1))
    print(f'Fin de mission : {distance:.2f} m parcourus.')


if __name__ == '__main__':
    main()
