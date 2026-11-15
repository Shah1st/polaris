"""
lancer_overhead.py - Mission sur le vrai RaspRover, avec la camera au-dessus de la scene
(branche overhead-camera, `guidance_global_map.drone.OverheadDrone`) dans le role du drone.

    PYTHONPATH=merging_map:src/guidance_calib:src/guidance_global_map:src/guidance_rover_pose \\
        python merging_map/lancer_overhead.py --image overhead.jpg --victime 1.2 0.9 \\
        --rover http://<IP du rover>:8080

Il faut les modules de la branche overhead-camera, avec opencv-contrib-python et numpy, et
sur le rover le serveur de la branche rover-teleop (rover/rover_server.py, POST /path).

Le repere (point 4 de docs/overhead-camera.md) : celui de la camera, origine au coin en bas a
gauche de l'image, x vers la droite, y vers le haut, en metres. Le depart, la victime et les
tuiles de la camera du rover doivent y etre exprimes. Il reste valable tant que la camera ne
bouge pas.
"""

import argparse
import math
import sys

from cerveau import mission
from fusion_carte import CarteFusion
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
    **reglages_carte,
):
    """Lance la mission. `depart` par defaut : la ou la camera voit le rover."""
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
    return mission(rover, carte, depart, victime, zone=zone, tolerance=tolerance, journal=journal)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument(
        '--image',
        required=True,
        help='photo de la camera du dessus, relue a chaque image : une photo fixe ne voit pas '
        'le rover bouger, il faudra la camera en direct pour une vraie mission',
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
    args = p.parse_args(argv)
    try:
        from guidance_global_map.drone import OverheadDrone, image_file_reader
        from guidance_global_map.pipeline import OverheadSettings
    except ImportError as e:
        sys.exit(
            f'{e}\nIl faut les modules de la branche overhead-camera sur le PYTHONPATH '
            '(src/guidance_calib, src/guidance_global_map, src/guidance_rover_pose) et '
            'opencv-contrib-python.'
        )
    reglages = {'camera_height': args.hauteur}
    if args.marqueur is not None:
        reglages['rover_marker_id'] = args.marqueur
    camera = OverheadDrone(image_file_reader(args.image), OverheadSettings(**reglages))
    instantanes = lancer(
        camera,
        client_http(args.rover),
        tuple(args.victime),
        tuple(args.depart) if args.depart else None,
    )
    traj = instantanes['fin']['traj']
    distance = sum(math.dist(traj[k], traj[k + 1]) for k in range(len(traj) - 1))
    print(f'Fin de mission : {distance:.2f} m parcourus.')


if __name__ == '__main__':
    main()
