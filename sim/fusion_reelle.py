"""
fusion_reelle.py - The maps of the REAL rover for the dashboard (teleop/server.py --fusion), in the
same format as sim/fusion_live.py, and a mission to a point picked on the map.

    GET  /rover.png /drone.png /fused.png /status.json   the maps and the state
    POST /reset              empty maps; new frame at the rover's current position
    POST /objectif           {"panneau": "fused"|"drone", "fx": 0..1, "fy": 0..1}: the point
                             clicked on a map (fractions of its width and height, from the top
                             left corner); {} clears the point
    POST /aller              the rover drives to the point by itself (bounded steps, one look of
                             its camera at each stop, around the obstacles it saw)
    POST /arreter            end of the mission and POST /stop to the rover
    POST /carte_locale       {"carte_locale": true|false}: build (true) or not (false) the
                             rover's local map from its own camera, while running; same
                             as --carte-locale / --no-carte-locale. Off empties it.
    POST /marqueur           {"sans_marqueur": true|false}: ignore (true) or trust (false)
                             the rover's marker in the drone's picture, while running;
                             same as --drone-sans-marqueur / --no-drone-sans-marqueur

Outside a mission: read only (GET /state of the rover, GET /obstacles of the perception service).
Frame: the odometry frame at start (or at the last /reset), origin at the start point, x towards
the front of the rover at that time. A view is fused only while the rover stands still.
The drone (--drone-nom: e.g. the iPhone above the area) is declared as the real drone source;
its picture is not fused into the rover's map until the rover is located in it. Two ways, both
passive (this program still sends no motor command outside a mission):
- the rover's ArUco marker, seen while the rover stands still: one picture is enough. Off
  for now: --drone-sans-marqueur is on by default (the marker is hard to detect), so only the
  rover's move places the map; --no-drone-sans-marqueur brings the marker back;
- no marker: the rover's own move. The camera's pictures at two stops at least
  --drone-pas-min apart show one object that moved, the rover; the same move in its odometry
  gives the direction of the camera's frame and corrects its scale. Drive the rover a little
  (the dashboard's keys) and stop: the drone map is then placed and fused.

    python sim/fusion_reelle.py --rover http://127.0.0.1:18080 \\
        --perception http://127.0.0.1:8095 --port 8096 [--drone-nom "iPhone (CamStream)"]
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE + '/merging_map')
sys.path.insert(0, RACINE + '/sim')
for _paquet in ('guidance_calib', 'guidance_global_map', 'guidance_rover_pose'):
    sys.path.insert(0, f'{RACINE}/src/{_paquet}')  # the overhead camera pipeline (--drone-source)

from cerveau import avancer  # noqa: E402
from fusion_carte import CarteFusion, decouper  # noqa: E402
from lancer_overhead import arreter_le_rover  # noqa: E402
from perception_rover import percevoir_http  # noqa: E402
from rover_reel import ArretOperateur, RaspRoverReel, ServeurRoverError, client_http  # noqa: E402

import fusion_feed as ff  # noqa: E402

DECALAGE = 20.0  # m: positive coordinates for the tile grid
DEMI = 2.5  # m: window of the rover map around it
TAILLE = 480  # px: side of the pictures


def angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def sans_traits(m, cases):
    """The drone's matrix without what is thinner than `cases` cells: a line painted or taped
    on the floor differs from the floor's colour but is not an obstacle. Those cells become
    free; so do the specks of fewer than cases * cases cells that are left."""
    k = int(round(cases))
    if k < 2:
        return m
    obstacle = (m == 1).astype(np.uint8)
    garde = cv2.morphologyEx(obstacle, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    # give back the border cells of what survived (the opening rounds the corners off)
    n, etiquettes = cv2.connectedComponents(obstacle, connectivity=8)
    vivants = np.unique(etiquettes[garde == 1])
    gros = np.isin(etiquettes, vivants[vivants > 0])
    large = cv2.dilate(garde, np.ones((3, 3), np.uint8)) == 1
    out = m.copy()
    out[(m == 1) & ~(gros & large)] = 0
    return out


def sans_rover(m, res, rover, rayon):
    """The drone's matrix with the cells within `rayon` m of the rover (x, y, ...) free: seen
    from above without a marker, the rover looks like any obstacle."""
    h, w = m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    pres = np.hypot((xs + 0.5) * res - rover[0], (ys + 0.5) * res - rover[1]) <= rayon
    out = m.copy()
    out[pres & (m == 1)] = 0
    return out


def marqueurs_tolerants(detecter, identifiant):
    """`detect_markers` of the overhead pipeline, with a second look for the rover's marker
    when the standard one misses it: the marker on the rover is small in the phone's picture
    (20 px or less), tilted and a little blurred. The second look enlarges the picture and
    loosens the detector, and only keeps the marker `identifiant`. On simulated markers of 14
    to 20 px it reads 16 of 16 where the standard detector reads 11; none found in 8 real
    pictures without a marker."""

    def detecter_mieux(image, dictionary_name='DICT_4X4_50'):
        trouves = detecter(image, dictionary_name)
        ancien = not hasattr(cv2.aruco, 'ArucoDetector')  # OpenCV 4.6: no second look
        if identifiant is None or identifiant in trouves or ancien:
            return trouves
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        dico = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        for echelle in (1, 2, 3):
            g = gris
            if echelle > 1:
                g = cv2.resize(gris, None, fx=echelle, fy=echelle, interpolation=cv2.INTER_CUBIC)
            for souple in (False, True):
                p = cv2.aruco.DetectorParameters()
                p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax = 5, 45
                p.adaptiveThreshWinSizeStep = 8
                p.minMarkerPerimeterRate = 0.01
                p.polygonalApproxAccuracyRate = 0.06
                p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                if souple:
                    p.errorCorrectionRate = 0.9
                    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
                coins, ids, _ = cv2.aruco.ArucoDetector(dico, p).detectMarkers(g)
                if ids is None:
                    continue
                for i, quad in zip(ids.flatten(), coins, strict=True):
                    if int(i) == identifiant:
                        trouves[identifiant] = (
                            np.asarray(quad, dtype=np.float64).reshape(4, 2) / echelle
                        )
                        return trouves
        return trouves

    return detecter_mieux


def part_ombre(image, mpp, forme, res=0.05):
    """For every cell of the drone's grid (`forme`: rows, columns; row 0 = south), the share of
    its pixels that look like a SHADOW on the floor: darker than the floor around, but still
    of the floor's colour (same hue, chroma that fell with the light). A black object has no
    colour left, a box or a railing has another colour: they are not shadows."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0
    L0 = float(np.median(L))
    clair = L >= L0 - 6.0  # the lit floor: where the floor's colour is read
    a0, b0 = float(np.median(a[clair])), float(np.median(b[clair]))
    c0 = max(1.0, float(np.hypot(a0, b0)))
    c = np.hypot(a, b)
    cos = (a * a0 + b * b0) / (np.maximum(c, 1e-3) * c0)  # 1: the floor's hue
    ombre = (L < L0 - 8.0) & (L > 0.25 * L0) & (cos > 0.9) & (c > 0.45 * c0 * L / L0)
    k = mpp / res
    petit = cv2.resize(ombre.astype(np.float32), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)[
        ::-1
    ]
    out = np.zeros(forme, np.float32)
    h, w = min(forme[0], petit.shape[0]), min(forme[1], petit.shape[1])
    out[:h, :w] = petit[:h, :w]
    return out


def mesure_du_mouvement(px_de, pose_de, px_a, pose_a, mini_px=8.0, mini_m=0.08):
    """What one move of the rover says of the camera, from PIXELS: (rotation of the camera's
    frame to the maps', metres per pixel), or None when the move is too small to tell. In
    pixels (x right, y up) so that a wrong scale of the pictures cannot bend it."""
    vu = (px_a[0] - px_de[0], px_a[1] - px_de[1])
    roule = (pose_a[0] - pose_de[0], pose_a[1] - pose_de[1])
    if math.hypot(*vu) < mini_px or math.hypot(*roule) < mini_m:
        return None
    rotation = angle(math.atan2(roule[1], roule[0]) - math.atan2(vu[1], vu[0]))
    return rotation, math.hypot(*roule) / math.hypot(*vu)


def troncon(chemin, grille, carte, longueur_max, depart):
    """The start of the planned path, `longueur_max` m at most, in a few points: each chord
    stays out of the obstacles grown by the rover's radius. Cells nobody has seen may be
    crossed, as the planner itself allows (avancer_esquive.troncon_vu is the strict version:
    only cells seen free)."""
    import avancer_esquive as ae

    points = ae.couper(chemin, longueur_max)
    plan = carte.grille_plan(grille)
    rayon = carte.rayon_effectif

    def libre(a, b):
        for _, p in ae.echantillons([a, b], grille.res / 2):
            if math.dist(p, depart) <= rayon:
                continue  # under the rover at the start
            i, j = plan.ij(*p)
            if not plan.dedans(i, j) or plan.bloque[i, j]:
                return False
        return True

    simples, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not libre(points[i], points[j]):
            j -= 1
        simples.append(points[j])
        i = j
    return simples


class Odometrie:
    """Stands in for the overhead camera of RaspRoverReel: the rover pose in the frame of the
    maps, from GET /state."""

    def __init__(self, fusion, requete):
        self.fusion, self.requete = fusion, requete

    def pose(self):
        etat = self.requete('GET', '/state')
        if etat is None:
            return None
        self.fusion.dernier_etat = etat
        return self.fusion.pose_carte(etat)

    def disponible(self):
        return True

    def zoomer(self, tuile):
        return []


class FusionReelle:
    def __init__(self, args, requete=None, percevoir=None, percevoir_frais=None):
        self.args = args
        self.verrou = threading.RLock()  # the map, the frame and the trail
        self.lock = threading.Lock()  # the pictures and the state served
        self.images = {}
        self.vues_img = {}
        self.status = {'etat': 'demarrage', 'genere_a': time.time()}
        self.resets = 0
        self.mission = {'etat': 'aucune', 'texte': '', 'pas': 0}
        self.fil_mission = None
        self.arret_demande = threading.Event()
        self.requete = requete or client_http(args.rover, delai=args.delai_http)
        self.percevoir = percevoir or percevoir_http(
            args.perception, delai=6.0, age_max=args.age_max, fraiche=False
        )
        self.percevoir_frais = percevoir_frais or percevoir_http(
            args.perception, delai=8.0, age_max=args.age_max, fraiche=True
        )
        self.camera = None  # the overhead camera (OverheadDrone), with --drone-source
        self.ia = None  # the vision model that finds the rover, with --drone-detecteur ia
        self.drone = {'etat': 'no drone camera (--drone-source)'}
        self.reinitialiser()
        if args.drone_source:
            self.ouvrir_camera()

    # ---- the drone: the overhead camera, read only ------------------------------------------
    def ouvrir_camera(self):
        from guidance_global_map import pipeline
        from guidance_global_map.drone import OverheadDrone, camera_reader
        from guidance_global_map.pipeline import OverheadSettings

        if not getattr(pipeline.detect_markers, 'tolerant', False):
            pipeline.detect_markers = marqueurs_tolerants(
                pipeline.detect_markers, self.args.drone_marqueur
            )
            pipeline.detect_markers.tolerant = True

        a = self.args
        mpp = a.drone_mpp
        self.mpp_mesure = mpp is not None  # measured by hand: the rover's move does not change it
        if mpp is None:  # width seen on the floor / width of the picture, set at the first one
            mpp = 1.0
            self.mpp_auto = True
        else:
            self.mpp_auto = False
        self.reglages_camera = OverheadSettings(
            metres_per_pixel=mpp,
            camera_height=a.drone_hauteur,
            rover_marker_id=a.drone_marqueur,
            marker_height=a.drone_hauteur_marqueur,
        )
        self.lecteur = camera_reader(a.drone_source, max_age=5.0)  # a slow Wi-Fi link
        self.ia = None
        if a.drone_detecteur == 'ia':
            from rover_du_dessus import DetecteurRover

            self.drone = {'etat': 'loading the rover detector (OWLv2)'}
            self.ia = DetecteurRover(marqueur_devant=a.drone_marqueur_devant)
        self.camera = OverheadDrone(self.lecteur, self.reglages_camera, period=0.0)
        self.drone = {'etat': 'waiting for the first picture'}
        threading.Thread(target=self.boucle_drone, daemon=True).start()

    def boucle_drone(self):
        while True:
            t0 = time.monotonic()
            try:
                self.tour_drone()
            except Exception as e:  # noqa: BLE001 - a bad picture must not end the thread
                self.drone = {'etat': f'{type(e).__name__}: {e}'[:160]}
            time.sleep(max(0.2, self.args.drone_periode - (time.monotonic() - t0)))

    def tour_drone(self):
        import dataclasses

        from guidance_global_map.segmentation import grid_to_fusion

        if self.mpp_auto:  # the phone app can change its picture size: follow it
            image = self.lecteur()
            if image is None:
                self.drone = {'etat': 'no picture from the drone camera'}
                return
            a = self.args
            largeur = 2 * a.drone_hauteur * math.tan(math.radians(a.drone_champ) / 2)
            # the field is the long side's; the rover's move may have corrected the scale
            mpp = largeur / max(image.shape[:2]) * self.correction_echelle
            if abs(mpp / self.camera.settings.metres_per_pixel - 1.0) > 1e-6:
                self.camera.settings = dataclasses.replace(
                    self.camera.settings, metres_per_pixel=mpp
                )
        if self.ia is not None:
            trouve = self.tour_drone_ia()
            if trouve is None:
                return
            m, vu, score = trouve
            objets = None  # the vision model sets the scale per picture: no placing by a move
        else:
            score = None
            r = self.camera.refresh(force=True)
            if r is None:
                probleme = self.camera.problem or 'no picture'
                self.drone = {'etat': f'picture not usable: {probleme}'}
                return
            m = np.asarray(grid_to_fusion(r.grid), dtype=np.int8)  # row 0 = south, 5 cm cells
            reglages = self.camera.settings
            image = self.lecteur() if self.args.drone_ombres else None
            if image is not None:  # a shadow on the floor is not an obstacle
                ombre = part_ombre(image, reglages.metres_per_pixel, m.shape, reglages.cell_size)
                m[(m == 1) & (ombre > 0.6)] = 0
            m = sans_traits(m, self.args.drone_largeur_min / self.camera.settings.cell_size)
            vu = None
            if r.rover_position is not None:
                vu = (r.rover_position[0], r.rover_position[1], r.rover_heading)
            objets = self.camera.seen_in(r)
        res = self.camera.settings.cell_size
        if self.args.drone_sans_marqueur and vu is not None:
            # no marker to trust: the heading read in the picture (the ArUco code, or any
            # white patch taken for the marker's paper) is dropped; the rover's move gives
            # the direction instead
            vu = (vu[0], vu[1], None)
        with self.verrou:
            pose = self.pose
            # The camera may be held by hand: a picture is placed in the frame of the maps by
            # the marker seen in that same picture, while the rover stands still. Without a
            # marker, by a move of the rover, which holds for the later pictures only as long
            # as the camera itself has not moved. Otherwise the last placed view stays.
            calage, par, echelle_changee = None, None, False
            if vu is not None and vu[2] is not None and pose is not None and self.immobile:
                calage, par = (angle(pose[2] - vu[2]), vu[0], vu[1], pose[0], pose[1]), 'marker'
            elif objets is not None:
                if self.calage_mvt is not None and not self.camera_immobile(objets):
                    self.calage_mvt, self.appui = None, None
                    self.info_mouvement = 'the camera moved: waiting for a new move of the rover'
                if pose is not None and self.immobile:
                    echelle_changee = self.caler_sur_mouvement(objets, pose)
                if self.calage_mvt is not None and not echelle_changee and pose is not None:
                    calage, par = self.calage_mvt, 'movement'
                    # no marker: the rover is where its odometry says, in the camera's frame.
                    # It must not be an obstacle under itself.
                    vu = self.rover_dans_camera(calage, pose)
                    m = sans_rover(m, res, vu, self.camera.settings.rover_radius)
            elif vu is not None and pose is not None and self.immobile:
                # the vision model found the rover but not which way it faces (the marker's
                # paper is hidden or too small): two stops of the rover give the direction
                calage = self.caler_ia_sur_mouvement(vu, pose)
                par = None if calage is None else 'movement'
            if calage is not None:
                self.calage, self.t_calage, self.cale_par = calage, time.time(), par
            place = None if self.vue_drone is None else self.vue_drone['place']
            if calage is not None:
                place = self.placer_vue(m, res, calage)
                if not self.args.drone_cumul:
                    self.carte.tuiles['drone'] = {}  # a fixed camera sees its whole area
                    self.carte.maj['drone'] = {}
                for msg in decouper(
                    place[0], place[1], place[2], 4, self.carte, 'drone', t=time.monotonic()
                ):
                    self.carte.recevoir(msg)
                if self.args.drone_cumul:
                    # a camera that moves: each placed picture adds to the drone map (what it
                    # sees again is replaced, what it no longer sees is kept)
                    place = self.matrice_drone()
            self.vue_drone = {'m': m, 'res': res, 'rover': vu, 'place': place}
        self.drone = {
            'etat': 'ok',
            't': time.time(),
            'cale': place is not None,
            'cale_sur_cette_image': calage is not None,
            'cale_par': self.cale_par if place is not None else None,
            'age_calage_s': None if self.calage is None else round(time.time() - self.t_calage),
            'mouvement': self.info_mouvement,
            'rover_vu': vu is not None,
            'detecteur': 'marker' if self.ia is None else 'OWLv2',
            'score_rover': score,
            'cap_vu': vu is not None and vu[2] is not None,
            'obstacles': int((m == 1).sum()),
            'libres': int((m == 0).sum()),
            'metres_par_pixel': self.camera.settings.metres_per_pixel,
            'hauteur_estimee_m': self.hauteur_estimee(),
            'zone_m': [round(m.shape[1] * res, 2), round(m.shape[0] * res, 2)],
        }

    def regler_carte_locale(self, corps):
        """POST /carte_locale: build the rover's local map or not, while running. (code, answer)

        Off: the rover's camera is no longer asked, outside a mission and at the stops of a
        mission, and what it had put on the maps is removed, so the planner works from the
        drone's map alone. What the rover bumped into stays out too: it is on the same map."""
        valeur = corps.get('carte_locale')
        if not isinstance(valeur, bool):
            return 400, {'error': '{"carte_locale": true or false} is expected'}
        with self.verrou:
            self.args.carte_locale = valeur
            if not valeur:
                self.carte.tuiles['rover'] = {}
                self.carte.maj['rover'] = {}
                self.vues = 0
                self.derniere_vue = None
                self.n_scan = {'points': 0, 'free': 0, 'obstacle': 0}
        print(time.strftime('%H:%M:%S'), 'local map', 'on' if valeur else 'off', flush=True)
        return 200, {'ok': True, 'carte_locale': valeur}

    def regler_marqueur(self, corps):
        """POST /marqueur: turn --drone-sans-marqueur on or off while running. (code, answer)

        On: the heading read in the drone's picture is ignored and the rover's move places
        the drone map. Off: the marker is trusted again. What the rover's moves have found so
        far is kept either way."""
        valeur = corps.get('sans_marqueur')
        if not isinstance(valeur, bool):
            return 400, {'error': '{"sans_marqueur": true or false} is expected'}
        with self.verrou:
            self.args.drone_sans_marqueur = valeur
        print(
            time.strftime('%H:%M:%S'),
            'drone map placed by the rover move only' if valeur else 'rover marker trusted',
            flush=True,
        )
        return 200, {'ok': True, 'sans_marqueur': valeur}

    def hauteur_estimee(self):
        """The camera height that the scale in use means for --drone-champ: once the rover's
        move corrected the scale, the height found, whatever --drone-hauteur said."""
        taille = getattr(self.camera, '_image_size', None)
        if taille is None:
            return None
        vu = self.camera.settings.metres_per_pixel * max(taille)  # long side on the floor
        return round(vu / (2 * math.tan(math.radians(self.args.drone_champ) / 2)), 2)

    def caler_sur_mouvement(self, objets, pose):
        """Place the camera's frame from a move of the rover, without a marker. Call with the
        rover standing still (and self.verrou held).

        The first call keeps what the camera sees and where the odometry puts the rover. At a
        later stop at least --drone-pas-min away, the one object that is at another place is
        the rover: its displacement in the picture and in the odometry gives the rotation
        between the two frames, and the camera's scale (corrected unless --drone-mpp gave a
        measured one). The result is self.calage_mvt. Returns True when the scale was changed:
        the picture in hand is then at the old scale and must not be placed; the next one is.
        """
        from guidance_global_map.scale_check import find_mover, frame_from_move, size_matches

        if self.appui is None:
            self.appui = (objets, pose)
            self.info_mouvement = 'waiting for the rover to move and stop (no marker seen)'
            return False
        objets0, pose0 = self.appui
        if math.dist(pose0[:2], pose[:2]) < self.args.drone_pas_min:
            return False
        self.appui = (objets, pose)  # the next move starts from here
        paire = find_mover(objets0, objets)
        if paire is None:
            self.info_mouvement = (
                'the rover moved but the camera did not see exactly one object move: is the '
                'rover in view, away from the obstacles, with nothing else moving?'
            )
            return False
        avant, apres = paire
        rotation, correction = frame_from_move(
            (avant.x, avant.y), (apres.x, apres.y), pose0[:2], pose[:2]
        )
        taille = self.camera.settings.rover_size
        # the height given may be far off: what tells the rover from anything else that
        # moved is its real size, at the scale its own move gives
        if not 0.2 <= correction <= 5.0 or not size_matches(apres, correction, taille, 0.4):
            self.info_mouvement = (
                f'an object moved, but not like the rover (scale x{correction:.2f}, '
                f'{apres.short * correction * 100:.0f} x {apres.long * correction * 100:.0f} cm)'
            )
            return False
        changee = False
        if self.mpp_mesure or abs(correction - 1.0) <= 0.03:
            correction = 1.0
        else:
            import dataclasses

            reglages = self.camera.settings
            self.camera.settings = dataclasses.replace(
                reglages, metres_per_pixel=reglages.metres_per_pixel * correction
            )
            self.correction_echelle *= correction
            self.appui = None  # what was kept is at the old scale
            changee = True
        # the picture's origin (bottom left) stays put when the scale changes
        self.calage_mvt = (
            rotation,
            apres.x * correction,
            apres.y * correction,
            pose[0],
            pose[1],
        )
        self.repere_fixe = [(o.x * correction, o.y * correction) for o in objets if o is not apres]
        self.info_mouvement = (
            f'placed by the rover move ({math.dist(pose0[:2], pose[:2]) * 100:.0f} cm)'
            + (f', scale corrected by {(correction - 1) * 100:+.0f} %' if changee else '')
        )
        return changee

    def caler_ia_sur_mouvement(self, vu, pose):
        """With the vision model: place a picture in which the rover was found without its
        heading. Call with the rover standing still (and self.verrou held).

        The model gives where the rover is and the picture's scale, in every picture. What is
        missing is the direction of the camera's frame. Two stops of the rover at least
        --drone-pas-min apart give it: the same move seen in the pictures and in the odometry
        (scale_check.frame_from_move). The direction is kept for the next pictures and
        measured again at every new stop, so it follows a camera that turns slowly; it is
        refused when the two distances disagree by more than 40 %, which is what a camera
        moved between the two stops looks like. Returns the placement, or None."""
        from guidance_global_map.scale_check import frame_from_move

        if self.appui_ia is None:
            self.appui_ia = (vu[:2], pose)
        else:
            vu0, pose0 = self.appui_ia
            if math.dist(pose0[:2], pose[:2]) >= self.args.drone_pas_min:
                self.appui_ia = (vu[:2], pose)
                rotation, correction = frame_from_move(vu0, vu[:2], pose0[:2], pose[:2])
                if 1 / 1.4 <= correction <= 1.4:
                    self.rotation_ia = rotation
                    self.info_mouvement = (
                        f'direction from the rover move '
                        f'({math.dist(pose0[:2], pose[:2]) * 100:.0f} cm)'
                    )
                else:
                    self.info_mouvement = (
                        f'the rover move and the pictures disagree (x{correction:.2f}): did '
                        'the camera move?'
                    )
        if self.rotation_ia is None:
            self.info_mouvement = self.info_mouvement or (
                'rover found without its heading: drive it a little and stop'
            )
            return None
        return (self.rotation_ia, vu[0], vu[1], pose[0], pose[1])

    def camera_immobile(self, objets, ecart=0.06):
        """True while what stood still when the rover's move placed the camera is still at the
        same place in its picture (most of it: the rover may hide a thing or two). With
        nothing but floor in view there is nothing to check, and the answer is True."""
        if not self.repere_fixe:
            return True
        retrouves = sum(
            any(math.dist(point, (o.x, o.y)) <= ecart for o in objets)
            for point in self.repere_fixe
        )
        return retrouves >= 0.6 * len(self.repere_fixe)

    @staticmethod
    def rover_dans_camera(calage, pose):
        """The rover (x, y, heading) in the camera's frame, from its pose in the maps' frame."""
        th, cx, cy, mx, my = calage
        c, s = math.cos(th), math.sin(th)
        dx, dy = pose[0] - mx, pose[1] - my
        return cx + c * dx + s * dy, cy - s * dx + c * dy, angle(pose[2] - th)

    def tour_drone_ia(self):
        """One picture with the vision model: (matrix, rover seen or None, score) or None.
        The rover's box gives where it is, the scale of the picture (the box is about
        --drone-taille-rover wide) and, with the marker's white paper, its heading."""
        import dataclasses

        from guidance_global_map.pipeline import frame_problem, process_image
        from guidance_global_map.segmentation import grid_to_fusion

        a = self.args
        image = self.lecteur()
        probleme = 'no picture' if image is None else frame_problem(image)
        if probleme is not None:
            self.drone = {'etat': f'picture not usable: {probleme}'}
            return None
        r = self.ia(image)
        reglages = self.camera.settings
        if r is not None and a.drone_mpp is None and not self.mpp_calibre:
            mpp = a.drone_taille_rover / r.taille
            reglages = dataclasses.replace(reglages, metres_per_pixel=mpp)
            self.camera.settings = reglages  # kept for the pictures where it is not found
            self.mpp_auto = False  # the rover's size is a better scale than the lens and height
        reglages = dataclasses.replace(reglages, rover_marker_id=None)
        mpp, res = reglages.metres_per_pixel, reglages.cell_size
        m = np.asarray(grid_to_fusion(process_image(image, reglages).grid), dtype=np.int8)
        if a.drone_ombres:  # a shadow on the floor is not an obstacle
            m[(m == 1) & (part_ombre(image, mpp, m.shape, res) > 0.6)] = 0
        m = sans_traits(m, a.drone_largeur_min / res)
        if r is None:
            return m, None, None
        x, y = r.x * mpp, (image.shape[0] - r.y) * mpp  # camera frame: y up from the bottom
        cap = None if r.cap is None else angle(-r.cap)
        # the rover is not an obstacle on its own map
        ys = (np.arange(m.shape[0]) + 0.5) * res
        xs = (np.arange(m.shape[1]) + 0.5) * res
        pres = np.hypot(xs[None, :] - x, ys[:, None] - y) <= 0.7 * a.drone_taille_rover + res
        m[pres & (m == 1)] = 0
        return m, (x, y, cap), round(r.score, 2)

    def matrice_drone(self):
        """Everything the drone has seen so far, as one matrix: (matrix, x0, y0)."""
        tuiles = {k: v for k, v in self.carte.tuiles.get('drone', {}).items() if k[0] == 4}
        n, res = self.carte.n, self.carte.res(4)
        tx0, tx1 = min(k[1] for k in tuiles), max(k[1] for k in tuiles) + 1
        ty0, ty1 = min(k[2] for k in tuiles), max(k[2] for k in tuiles) + 1
        m = np.full(((ty1 - ty0) * n, (tx1 - tx0) * n), -1, np.int8)
        for (_, tx, ty), data in tuiles.items():
            m[(ty - ty0) * n : (ty - ty0 + 1) * n, (tx - tx0) * n : (tx - tx0 + 1) * n] = data
        return m, tx0 * n * res, ty0 * n * res

    @staticmethod
    def placer_vue(m, res, calage):
        """The camera's matrix (its own frame: origin bottom left of the picture) resampled on
        the grid of the maps: (matrix, x0, y0)."""
        th, cx, cy, mx, my = calage
        c, s = math.cos(th), math.sin(th)
        h, w = m.shape
        coins = [(0, 0), (w * res, 0), (0, h * res), (w * res, h * res)]
        xs = [mx + c * (x - cx) - s * (y - cy) for x, y in coins]
        ys = [my + s * (x - cx) + c * (y - cy) for x, y in coins]
        j0, j1 = math.floor(min(xs) / res), math.ceil(max(xs) / res)
        i0, i1 = math.floor(min(ys) / res), math.ceil(max(ys) / res)
        X = (np.arange(j0, j1) + 0.5) * res - mx
        Y = (np.arange(i0, i1) + 0.5) * res - my
        XX, YY = np.meshgrid(X, Y)
        jc = np.floor((cx + c * XX + s * YY) / res).astype(int)
        ic = np.floor((cy - s * XX + c * YY) / res).astype(int)
        ok = (jc >= 0) & (jc < w) & (ic >= 0) & (ic < h)
        out = np.full(XX.shape, -1, np.int8)
        out[ok] = m[ic[ok], jc[ok]]
        return out, j0 * res, i0 * res

    def reinitialiser(self):
        """Empty map, trail cleared, new frame: its origin is the rover pose at the next round
        (POST /reset)."""
        with self.verrou:
            self.carte = CarteFusion(niveau_plan=4, rayon_robot=self.args.rayon)
            self.origine = None
            self.pose = None
            self.trail = []
            self.vues = 0
            self.derniere_vue = None
            self.dernier_etat = None
            self.precedente = None
            self.objectif = None
            self.decision = None
            self.n_scan = {'points': 0, 'free': 0, 'obstacle': 0}
            self.immobile = False
            self.calage = None  # camera frame -> frame of the maps, of the last placed picture
            self.t_calage = 0.0
            self.cale_par = None  # 'marker' or 'movement'
            self.calage_mvt = None  # the placement found from a move of the rover
            self.repere_fixe = []  # what stood still in the camera's view when it was found
            self.appui = None  # (objects the camera saw, rover pose) at the last stop
            self.appui_ia = None  # (rover in the picture, rover pose) at the last stop
            self.mpp_calibre = False  # the scale was measured by /calibrer: kept as it is
            self.rotation_ia = None  # camera frame -> maps, from a move (vision model)
            self.info_mouvement = None
            self.correction_echelle = 1.0  # what the rover's move said of the camera's scale
            self.vue_drone = None
            self.local = {'source': self.args.source_locale, 'age_s': None, 'erreur': None}

    def en_mission(self):
        return self.fil_mission is not None and self.fil_mission.is_alive()

    # ---- frame -----------------------------------------------------------------------
    def pose_carte(self, etat):
        o = etat['odom']
        x, y, h = float(o['x']), float(o['y']), math.radians(float(o['heading_deg']))
        with self.verrou:
            if self.origine is None:
                self.origine = (x, y, h)
            x0, y0, h0 = self.origine
        c, s = math.cos(-h0), math.sin(-h0)
        dx, dy = x - x0, y - y0
        return DECALAGE + c * dx - s * dy, DECALAGE + s * dx + c * dy, angle(h - h0)

    @staticmethod
    def pan(etat):
        """Pan of the camera turret (deg, + = to the right for the rover server)."""
        return float((((etat or {}).get('gimbal') or {}).get('pan')) or 0.0)

    def publier(self, vue):
        m, x0, y0 = vue
        with self.verrou:
            for msg in decouper(m, x0, y0, 4, self.carte, 'rover', t=time.monotonic()):
                self.carte.recevoir(msg)
            self.vues += 1
            self.derniere_vue = time.time()
            self.n_scan = {
                'points': int(m.size),
                'free': int((m == 0).sum()),
                'obstacle': int((m == 1).sum()),
            }

    def vue_locale(self):
        """What the rover's own obstacle grid shows (its GET /local_grid: the depth camera on
        the rover, rover/local_map.py), on the maps; None when it has none or it is too old."""
        import suivi_continu as sc

        rep = self.requete('GET', '/local_grid')
        with self.verrou:
            origine = self.origine
        self.local = {
            'source': 'rover',
            'age_s': (rep or {}).get('age_s'),
            'erreur': 'no answer to GET /local_grid' if rep is None else rep.get('error'),
        }
        if rep is None or origine is None:
            return None
        x0, y0, h0 = origine  # the odometry pose that is (DECALAGE, DECALAGE, 0) on the maps
        c, s = math.cos(-h0), math.sin(-h0)
        zero = (DECALAGE - (c * x0 - s * y0), DECALAGE - (s * x0 + c * y0))
        return sc.grille_locale(rep, zero, -h0, self.args.age_max)

    def regard(self, position, cap):
        """One look of the rover at a stop: its own grid (--source-locale rover) or the
        perception service's picture taken after the stop."""
        if self.args.source_locale == 'rover':
            return self.vue_locale()
        return self.percevoir_frais(position, cap)

    # ---- outside a mission: observe --------------------------------------------------------
    def tour(self):
        etat = self.requete('GET', '/state')
        if etat is None:
            raise RuntimeError('the rover does not answer GET /state')
        pose = self.pose_carte(etat)
        with self.verrou:
            self.pose, self.dernier_etat = pose, etat
            if not self.trail or math.dist(self.trail[-1], pose[:2]) > 0.03:
                self.trail.append(pose[:2])
                self.trail = self.trail[-2000:]
        v = abs(float(etat['odom'].get('v') or 0.0))
        immobile = (
            v < 0.02
            and self.precedente is not None
            and math.dist(self.precedente[:2], pose[:2]) < 0.03
            and abs(angle(self.precedente[2] - pose[2])) < math.radians(3)
        )
        self.precedente = pose
        self.immobile = immobile
        if not self.args.carte_locale:
            info = "local map off: the rover's camera is not used"
        elif self.en_mission():
            info = 'mission running: the rover looks at each stop'
        elif self.args.source_locale == 'rover':  # its own grid: valid while it moves too
            vue = self.vue_locale()
            if vue is None:
                info = "local map: no recent grid from the rover's depth camera"
            else:
                self.publier(vue)
                info = "rover's grid fused"
        elif not immobile:
            info = 'rover moving: view not fused'
        else:
            vue = self.percevoir(pose[:2], angle(pose[2] - math.radians(self.pan(etat))))
            if vue is None:
                info = 'perception: no recent picture'
            else:
                self.publier(vue)
                info = 'view fused'
        self.dessiner(pose, info, etat)

    # ---- mission ---------------------------------------------------------------------
    def zone_mission(self, pos):
        ob = self.objectif
        m = self.args.marge_zone
        return (
            min(pos[0], ob[0]) - m,
            min(pos[1], ob[1]) - m,
            max(pos[0], ob[0]) + m,
            max(pos[1], ob[1]) + m,
        )

    def journal_mission(self, etat=None, texte=None, pas=None):
        with self.lock:
            if etat is not None:
                self.mission['etat'] = etat
            if texte is not None:
                self.mission['texte'] = texte
                print(time.strftime('%H:%M:%S'), 'mission:', texte, flush=True)
            if pas is not None:
                self.mission['pas'] = pas

    def aller(self):
        if self.objectif is None:
            return 409, {'error': 'no goal: click on a map first'}
        if self.en_mission():
            return 409, {'error': 'a mission is already running'}
        self.arret_demande.clear()
        self.journal_mission('en_cours', 'start', 0)
        self.fil_mission = threading.Thread(target=self._mission, daemon=True)
        self.fil_mission.start()
        return 200, {'ok': True}

    def arreter(self):
        self.arret_demande.set()
        ok = arreter_le_rover(self.requete, journal=lambda t: print(t, flush=True))
        if self.fil_mission is not None:
            self.fil_mission.join(timeout=1.0)
        self.journal_mission('arret', 'stopped from the dashboard (/stop sent)')
        return 200, {'ok': True, 'stop_recu': ok}

    def _mission(self):
        a = self.args
        issue = 'erreur'
        try:
            if a.continu:
                issue = self._mission_continue()
                return
            if a.segment > 0:
                issue = self._mission_segments()
                return
            odo = Odometrie(self, self.requete)
            rover = RaspRoverReel(
                self.carte,
                self.requete,
                position_drone=odo.pose,
                # no local map: the rover's camera is not asked, at any stop
                percevoir=lambda position, cap: (
                    self.regard(position, cap) if self.args.carte_locale else None
                ),
                drone=odo,
                vitesse=a.vitesse,
                delai_liaison=a.delai_liaison,
                periode=a.periode_pas,
                delai_pret=30.0,
                signaler=lambda t: self.journal_mission(texte=t),
            )
            depart = odo.pose()
            if depart is None:
                raise RuntimeError('the rover does not answer GET /state')
            rover.placer(depart[:2])
            rover.recaler()
            issue = self._boucle_mission(rover)
        except ArretOperateur as e:
            issue = 'arret_operateur'
            self.journal_mission(texte=f'OPERATOR STOP: {e}')
        except (ServeurRoverError, RuntimeError) as e:
            issue = 'erreur'
            self.journal_mission(texte=f'ERROR: {e}')
        except Exception as e:  # noqa: BLE001 - the mission ends, the rover stops
            issue = 'erreur'
            self.journal_mission(texte=f'ERROR: {type(e).__name__}: {e}')
        finally:
            if issue != 'arrive':
                arreter_le_rover(self.requete, journal=lambda t: print(t, flush=True))
            self.journal_mission(etat=issue)

    # ---- calibration: the rover drives forward and back, seen by the drone ------------------
    def calibrer(self):
        """POST /calibrer: the rover drives --calibrage-distance forward, then back. The vision
        model finds it in the drone's picture at the three stops; the two moves give the
        direction of the camera's frame and its scale. (code, answer)"""
        if self.en_mission():
            return 409, {'error': 'a mission or a calibration is already running'}
        if self.camera is None or self.ia is None:
            return 409, {'error': 'needs the drone camera with the vision model (ia)'}
        self.arret_demande.clear()
        self.journal_mission('en_cours', 'calibration: start', 0)
        self.fil_mission = threading.Thread(target=self._calibrage, daemon=True)
        self.fil_mission.start()
        return 200, {'ok': True}

    def _rover_dans_l_image(self, apres, delai=12.0):
        """(rover in the picture in PIXELS (x right, y up), rover pose on the maps) from a drone
        picture taken after `apres` (time.time()) with the rover standing still and found by
        the model; None after `delai` s without one."""
        fin = time.time() + delai
        while time.time() < fin and not self.arret_demande.is_set():
            d = self.drone
            if d.get('etat') == 'ok' and d.get('t', 0.0) > apres and d.get('rover_vu'):
                with self.verrou:
                    vd, pose, immobile = self.vue_drone, self.pose, self.immobile
                mpp = d.get('metres_par_pixel')
                if immobile and vd is not None and vd['rover'] is not None and pose and mpp:
                    return (vd['rover'][0] / mpp, vd['rover'][1] / mpp), pose
            time.sleep(0.2)
        return None

    def _rouler(self, distance, delai=20.0):
        """POST /move {distance} (+ forward, - backward) and wait for the rover to stand still.
        Raises when the rover does not take it or does not finish."""
        import avancer_esquive as ae

        etat = self.requete('GET', '/state')
        if etat is None or not ae.pret(etat):
            raise RuntimeError('rover not ready (GET /state): link, gyro, or a move running')
        if self.requete('POST', '/move', {'distance': distance}) is None:
            raise RuntimeError('the rover did not answer POST /move')
        debut = time.monotonic()
        time.sleep(0.5)
        while time.monotonic() - debut < delai:
            if self.arret_demande.is_set():
                raise ArretOperateur('calibration stopped')
            etat = self.requete('GET', '/state')
            if etat is not None and ae.pret(etat):
                return
            time.sleep(0.5)
        raise RuntimeError(f'the rover did not finish {distance:+.2f} m in {delai:.0f} s')

    def _calibrage(self):
        a = self.args
        d = a.calibrage_distance
        issue = 'erreur'
        try:
            p0 = self._rover_dans_l_image(time.time())
            if p0 is None:
                raise RuntimeError('the drone does not see the rover standing still')
            self.journal_mission(texte=f'calibration: {d * 100:.0f} cm forward', pas=1)
            self._rouler(d)
            p1 = self._rover_dans_l_image(time.time() + 1.0)
            self.journal_mission(texte=f'calibration: {d * 100:.0f} cm back', pas=2)
            self._rouler(-d)
            p2 = self._rover_dans_l_image(time.time() + 1.0)
            if p1 is None or p2 is None:
                raise RuntimeError('the drone lost the rover during the move')
            mesures = [mesure_du_mouvement(*p0, *p1), mesure_du_mouvement(*p1, *p2)]
            if any(m is None for m in mesures):
                raise RuntimeError('the rover hardly moved (in the picture or on its odometry)')
            (r1, k1), (r2, k2) = mesures
            ecart = abs(angle(r1 - r2))
            if ecart > math.radians(a.calibrage_ecart) or not 0.75 <= k1 / k2 <= 1.33:
                raise RuntimeError(
                    f'forward and back disagree ({math.degrees(ecart):.0f} deg, scales '
                    f'{k1 * 1000:.1f} and {k2 * 1000:.1f} mm per pixel): hold the camera still'
                )
            rotation = angle(r1 + angle(r2 - r1) / 2)
            mpp = (k1 + k2) / 2
            self.fixer_calibrage(rotation, mpp, p2)
            issue = 'arrive'
            self.journal_mission(
                texte=f'CALIBRATED: direction {math.degrees(rotation):+.0f} deg, '
                f'{mpp * 1000:.1f} mm per pixel (forward and back agree within '
                f'{math.degrees(ecart):.0f} deg)'
            )
        except ArretOperateur as e:
            issue = 'arret_operateur'
            self.journal_mission(texte=f'OPERATOR STOP: {e}')
        except Exception as e:  # noqa: BLE001 - the calibration ends, the rover stops
            self.journal_mission(texte=f'CALIBRATION FAILED: {e}')
        finally:
            if issue != 'arrive':
                arreter_le_rover(self.requete, journal=lambda t: print(t, flush=True))
            self.journal_mission(etat=issue)

    def fixer_calibrage(self, rotation, mpp, dernier):
        """The direction and the scale measured by the calibration become the camera's: the
        scale is no longer read off the rover's box, the drone map starts again."""
        import dataclasses

        px, pose = dernier
        with self.verrou:
            self.camera.settings = dataclasses.replace(self.camera.settings, metres_per_pixel=mpp)
            self.mpp_calibre, self.mpp_auto = True, False
            self.rotation_ia = rotation
            self.appui_ia = ((px[0] * mpp, px[1] * mpp), pose)
            self.info_mouvement = 'direction and scale from the calibration (forward and back)'
            self.calage, self.vue_drone = None, None
            self.carte.tuiles['drone'] = {}  # placed with the old direction and scale
            self.carte.maj['drone'] = {}

    def _mission_segments(self):
        """The mission in long segments (--segment): one POST /path carries several points of
        the planned path and the rover follows them without stopping (the driver of
        merging_map/avancer_esquive.py), instead of one stop every --pas m."""
        rover, ae = self._preparer_segments()
        return self._boucle_segments(rover, ae)

    def _mission_continue(self):
        """The mission without stops (--continu): the path is planned again about twice a
        second while the rover drives, and sent again when it changed; the rover takes it
        over at speed. The rover's own guard (its depth camera) slows it down and stops it."""
        rover, ae = self._preparer_segments()
        config = self.requete('GET', '/config') or {}
        if not config.get('local_guard'):
            self.journal_mission(
                texte="WARNING: the rover's local guard is off (POST /config "
                '{"local_guard": 1} on the rover): it will not stop by itself for an obstacle'
            )
        return self._boucle_continue(rover, ae)

    def _preparer_segments(self):
        """The rover driver of merging_map/avancer_esquive.py, placed on the maps and ready."""
        import avancer_esquive as ae

        a = self.args
        odo = ae.Odometrie(self.requete)
        # its frame is the frame of the maps: the same pose for the same GET /state
        odo._ancrer(ae.pose_odom(odo.etat), self.pose_carte(odo.etat))
        rover = ae.RoverSegments(
            self.carte,
            self.requete,
            odo,
            lambda position, cap: (self.regard(position, cap) if self.args.carte_locale else None),
            vitesse=a.vitesse,
            delai_liaison=a.delai_liaison,
            periode=a.periode_pas,
            delai_pret=30.0,
            signaler=lambda t: self.journal_mission(texte=t),
        )
        rover.placer(odo.pose()[:2])
        debut = time.monotonic()
        while not ae.pret(odo.etat):  # ESP32, gyroscope, nothing running
            if (odo.etat.get('ctrl') or {}).get('mode') == 'manual':
                raise ArretOperateur('rover driven by hand (manual command /cmd)')
            if time.monotonic() - debut > 30.0:
                raise ServeurRoverError('rover not ready after 30 s (GET /state)')
            time.sleep(1.0)
            rover.requete('GET', '/state')
        rover.recaler_apres()
        return rover, ae

    def _regarder_autour(self, rover):
        """The rover's depth camera is fixed to its body: to look around, the rover turns on
        the spot (left, right, back) and each view goes on the maps."""
        for angle_deg in (40.0, -80.0, 40.0):
            if self.arret_demande.is_set():
                return
            rover.pivoter(angle_deg)
            vue = self.vue_locale()
            if vue is not None:
                self.publier(vue)

    def _boucle_continue(self, rover, ae):
        import suivi_continu as sc

        a = self.args
        portee = a.segment if a.segment > 0 else 2.0
        strict = a.segment_vu or a.carte_locale  # with its own eyes: only where it has seen
        nom = f'c{int(time.time()) % 100000}'
        courant = None  # the path the rover is driving (frame of the maps), None when stopped
        envois, regards, muet = 0, 0, None
        fin = time.monotonic() + a.duree_max
        while time.monotonic() < fin:
            t0 = time.monotonic()
            objectif = self.objectif
            if self.arret_demande.is_set() or objectif is None:
                self.journal_mission(texte='mission stopped')
                return 'arret'
            phase, info = rover.suivi()
            if phase == 'liaison':
                muet = muet or t0
                if t0 - muet > a.delai_liaison:
                    self.journal_mission(texte=f'ERROR: no answer for {a.delai_liaison:.0f} s')
                    return 'erreur'
                time.sleep(a.periode_plan)
                continue
            muet = None
            pos = rover.position()
            if phase in ('fini', 'repos'):
                courant = None
            if phase == 'fini':
                statut = info.get('status')
                vu = rover.obstacle_vu(info)
                if vu is not None:
                    with self.verrou:
                        self.carte.signaler_obstacle(
                            *vu, rayon=0.1, t=time.monotonic(), duree=a.duree_obstacle
                        )
                    self.journal_mission(
                        texte=f'OBSTACLE seen by the rover at ({vu[0] - DECALAGE:.2f}, '
                        f'{vu[1] - DECALAGE:.2f}): stopped before it, new route'
                    )
                elif statut != 'done':
                    impact = rover.point_de_choc(statut)
                    if impact is not None:
                        with self.verrou:
                            rover.signaler_choc(impact)
                        self.journal_mission(texte=f'BUMP ({statut}): obstacle marked on the map')
                    else:
                        self.journal_mission(texte=f'path ended on "{statut}"')
            reste = math.dist(pos, objectif)
            if reste <= a.tolerance:
                if courant is None:
                    self.journal_mission(
                        texte=f'ARRIVED {reste:.2f} m from the goal, {envois} paths sent'
                    )
                    return 'arrive'
                time.sleep(a.periode_plan)  # the rover finishes its path by itself
                continue
            if a.carte_locale and (a.source_locale == 'rover' or courant is None):
                pan = math.radians(self.pan(rover.odo.etat))
                vue = self.regard(rover.est, None if rover.cap is None else angle(rover.cap - pan))
                if vue is not None:
                    self.publier(vue)
            with self.verrou:
                self.carte.expirer_chocs(time.monotonic())
                if courant is None:
                    avant, depart = [pos], pos
                else:
                    avant, depart = sc.prolonger(courant, pos, a.vitesse * a.latence + 0.05)
                dec = self.carte.decider(
                    depart,
                    objectif,
                    zone=self.zone_mission(pos),
                    tolerance=a.tolerance,
                    drone_disponible=False,
                )
                self.decision = dec
                chemin = [tuple(p) for p in dec.get('chemin') or []]
                grille = dec['grille']
                if strict:
                    points, _ = ae.troncon_vu(chemin or [depart], grille, self.carte, portee, pos)
                else:
                    points = troncon(chemin or [depart], grille, self.carte, portee, pos)
                libre = sc.corde_libre(
                    self.carte.grille_plan(grille), grille, strict, pos, self.carte.rayon_effectif
                )
                trajet = avant[:-1] + sc.lisser(points, libre)
                barre = courant is not None and (
                    self.carte.premier_point_bloque(sc.reste(courant, pos), horizon=1.0)
                    is not None
                )
            a_rouler = ae.longueur_chemin(points)
            if courant is None:  # stopped: the checks of the segment loop
                bloque = dec['action'] == 'OBJECTIF_DEGRADE' and math.dist(
                    dec.get('point', pos), pos
                )
                if bloque is not False and bloque < 0.05:
                    self.journal_mission(texte='BLOCKED: no way to the goal')
                    return 'bloque'
                if a_rouler < ae.SEGMENT_MIN:
                    if ae.longueur_chemin(chemin) < ae.SEGMENT_MIN and reste <= a.tolerance + 0.1:
                        self.journal_mission(
                            texte=f'ARRIVED {reste:.2f} m from the goal, {envois} paths sent'
                        )
                        return 'arrive'
                    if a.carte_locale and a.source_locale == 'rover' and regards < 1:
                        regards += 1
                        self.journal_mission(texte='nothing seen free ahead: looking around')
                        self._regarder_autour(rover)
                        continue
                    self.journal_mission(
                        texte=f'BLOCKED: nothing to drive ahead ({a_rouler:.2f} m)'
                        + (': only where the map has seen' if strict else '')
                    )
                    return 'bloque'
                envoyer = True
            elif a_rouler < ae.SEGMENT_MIN:
                # nothing better to drive; if the map now bars the path being driven, a short
                # path makes the rover stop a little ahead instead of driving on
                envoyer, trajet = barre, avant
            else:
                envoyer = barre or sc.a_remplacer(courant, trajet, pos, objectif)
            if envoyer and len(trajet) >= 1:
                envois += 1
                if rover.lancer(trajet, f'{nom}-{envois}'):
                    if courant is None:
                        regards = 0
                    courant = trajet
                    self.journal_mission(
                        texte=f'path {envois}: {dec["action"]}, '
                        f'{ae.longueur_chemin(trajet):.2f} m in {len(trajet)} points, '
                        f'{reste:.2f} m to go' + (' (route barred: new route)' if barre else ''),
                        pas=envois,
                    )
            time.sleep(max(0.05, a.periode_plan - (time.monotonic() - t0)))
        self.journal_mission(texte=f'{a.duree_max:.0f} s without arriving')
        return 'max_pas'

    def _boucle_segments(self, rover, ae):
        a = self.args
        sans_vue = 0
        for k in range(a.max_pas):
            objectif = self.objectif
            if self.arret_demande.is_set() or objectif is None:
                self.journal_mission(texte='mission stopped')
                return 'arret'
            pos = rover.position()
            reste = math.dist(pos, objectif)
            if reste <= a.tolerance:
                self.journal_mission(texte=f'ARRIVED {reste:.2f} m from the goal, {k} segments')
                return 'arrive'
            if a.carte_locale:  # a look of the rover's camera at each stop
                pan = math.radians(self.pan(self.dernier_etat))
                cap = None if rover.cap is None else angle(rover.cap - pan)
                vue = self.regard(rover.est, cap)
                if vue is None:
                    sans_vue += 1
                    self.journal_mission(texte=f'perception unavailable ({sans_vue}/3): waiting')
                    if sans_vue >= 3:
                        return 'perception_indisponible'
                    time.sleep(1.0)
                    continue
                sans_vue = 0
                self.publier(vue)
            with self.verrou:
                dec = self.carte.decider(
                    pos,
                    objectif,
                    zone=self.zone_mission(pos),
                    tolerance=a.tolerance,
                    drone_disponible=False,
                )
                self.decision = dec
                chemin = [tuple(p) for p in dec.get('chemin') or []]
                # the part of the path that only crosses cells seen free (by the drone, or by
                # the rover with the local map), in a few points, --segment m at most
                if a.segment_vu:
                    points, raison = ae.troncon_vu(
                        chemin or [pos], dec['grille'], self.carte, a.segment, pos
                    )
                else:
                    points = troncon(chemin or [pos], dec['grille'], self.carte, a.segment, pos)
            bloque = dec['action'] == 'OBJECTIF_DEGRADE' and math.dist(dec.get('point', pos), pos)
            if bloque is not False and bloque < 0.05:
                self.journal_mission(texte='BLOCKED: no way to the goal')
                return 'bloque'
            if len(chemin) < 2:
                if reste <= a.tolerance + 0.1:  # the A* arrives within one 5 cm cell
                    self.journal_mission(
                        texte=f'ARRIVED {reste:.2f} m from the goal, {k} segments'
                    )
                    return 'arrive'
                self.journal_mission(texte='BLOCKED: no path')
                return 'bloque'
            a_rouler = ae.longueur_chemin(points)
            if a_rouler < ae.SEGMENT_MIN:
                if ae.longueur_chemin(chemin) < ae.SEGMENT_MIN and reste <= a.tolerance + 0.1:
                    self.journal_mission(
                        texte=f'ARRIVED {reste:.2f} m from the goal, {k} segments'
                    )
                    return 'arrive'
                self.journal_mission(
                    texte=f'BLOCKED: nothing to drive ahead ({a_rouler:.2f} m)'
                    + (': --segment-vu, only where the map has seen' if a.segment_vu else '')
                )
                return 'bloque'
            duree = ae.duree_prevue(points, rover.cap, a.vitesse)
            self.journal_mission(
                texte=f'segment {k + 1}: {dec["action"]}, {a_rouler:.2f} m in '
                f'{len(points) - 1} points, {reste:.2f} m to go',
                pas=k + 1,
            )
            if self.arret_demande.is_set():
                return 'arret'
            statut, impact = rover.suivre(points[1:], duree)
            if statut in ('liaison', 'redemarrage'):
                self.journal_mission(texte=f'ERROR: segment ended on "{statut}" (link or server)')
                return 'erreur'
            if impact is not None:
                with self.verrou:
                    rover.signaler_choc(impact)
                self.journal_mission(texte=f'BUMP ({statut}): obstacle marked on the map')
        self.journal_mission(texte=f'{a.max_pas} segments without arriving')
        return 'max_pas'

    def _boucle_mission(self, rover):
        a = self.args
        sans_vue = 0
        for k in range(a.max_pas):
            objectif = self.objectif
            if self.arret_demande.is_set() or objectif is None:
                self.journal_mission(texte='mission stopped')
                return 'arret'
            pos = rover.position()
            reste = math.dist(pos, objectif)
            if reste <= a.tolerance:
                self.journal_mission(texte=f'ARRIVED {reste:.2f} m from the goal, {k} steps')
                return 'arrive'
            if a.carte_locale:  # a look of the rover's camera at each stop
                pan = math.radians(self.pan(self.dernier_etat))
                cap = None if rover.cap is None else angle(rover.cap - pan)
                vue = self.regard(rover.est, cap)
                if vue is None:
                    sans_vue += 1
                    self.journal_mission(texte=f'perception unavailable ({sans_vue}/3): waiting')
                    if sans_vue >= 3:
                        return 'perception_indisponible'
                    time.sleep(1.0)
                    continue
                sans_vue = 0
                self.publier(vue)
            with self.verrou:
                dec = self.carte.decider(
                    pos,
                    objectif,
                    zone=self.zone_mission(pos),
                    tolerance=a.tolerance,
                    drone_disponible=False,
                )
                self.decision = dec
            chemin = [tuple(p) for p in dec.get('chemin') or []]
            bloque = dec['action'] == 'OBJECTIF_DEGRADE' and math.dist(dec.get('point', pos), pos)
            if bloque is not False and bloque < 0.05:
                self.journal_mission(texte='BLOCKED: no way to the goal')
                return 'bloque'
            if len(chemin) < 2:
                if reste <= a.tolerance + 0.1:  # the A* arrives within one 5 cm cell
                    self.journal_mission(texte=f'ARRIVED {reste:.2f} m from the goal, {k} steps')
                    return 'arrive'
                self.journal_mission(texte='BLOCKED: no path')
                return 'bloque'
            cible, _ = avancer(pos, list(chemin), a.pas)
            self.journal_mission(
                texte=f'step {k + 1}: {dec["action"]}, {reste:.2f} m to go', pas=k + 1
            )
            if self.arret_demande.is_set():
                return 'arret'
            impact = rover.deplacer(cible)
            if impact is not None:
                with self.verrou:
                    rover.signaler_choc(impact)
                self.journal_mission(texte='BUMP: obstacle marked on the map')
        self.journal_mission(texte=f'{a.max_pas} steps without arriving')
        return 'max_pas'

    # ---- pictures and state --------------------------------------------------------------
    def zone_vue(self, pose):
        xs = [p[0] for p in self.trail] + [pose[0]]
        ys = [p[1] for p in self.trail] + [pose[1]]
        if self.objectif is not None:
            xs.append(self.objectif[0])
            ys.append(self.objectif[1])
        if getattr(self.args, 'drone_cumul', False):  # the view grows with the drone map
            for cle in self.carte.tuiles.get('drone', {}):
                x, y, cote = self.carte.rect_tuile(cle)
                xs += [x, x + cote]
                ys += [y, y + cote]
        m = 2.5
        x0, x1 = min(xs) - m, max(xs) + m
        y0, y1 = min(ys) - m, max(ys) + m
        cote = max(x1 - x0, y1 - y0, 5.0)  # square: the picture keeps its proportions
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return cx - cote / 2, cy - cote / 2, cx + cote / 2, cy + cote / 2

    def dessiner(self, pose, info, etat):
        now0 = time.time()
        with self.verrou:
            z = self.zone_vue(pose)
            g = self.carte.grille(*z)
            objectif = self.objectif
            if objectif is not None and not self.en_mission():
                self.decision = self.carte.decider(
                    pose[:2],
                    objectif,
                    zone=self.zone_mission(pose),
                    tolerance=self.args.tolerance,
                    drone_disponible=False,
                )
            dec = (self.decision or {}) if objectif is not None else {}
            zr = (pose[0] - DEMI, pose[1] - DEMI, pose[0] + DEMI, pose[1] + DEMI)
            gr = self.carte.grille(*zr)
            trail = list(self.trail)
            n_tuiles = len(self.carte.tuiles['rover'])
            n_tuiles_drone = len(self.carte.tuiles.get('drone') or {})
        gx0, gy0 = g.j0 * g.res, g.i0 * g.res
        vue = ff.View(z[0], z[1], z[2], z[3], TAILLE / (z[2] - z[0]))
        depart = trail[0] if trail else None
        chemin = dec.get('chemin')
        fused = ff.render_fused_view(
            g.val, g.src, gx0, gy0, vue, pose, objectif, depart, None, path=chemin, trail=trail
        )
        ff.draw_frame(fused, vue, 1.0)
        # the drone panel: what the overhead camera sees. Placed on the frame of the maps once
        # the rover's marker was seen; before that, in the camera's own frame (not clickable).
        etat_drone = dict(self.drone)
        vd = self.vue_drone
        frais = etat_drone.get('etat') == 'ok' and now0 - etat_drone.get('t', 0) < 10.0
        vue_d = vue
        if vd is not None and vd['place'] is not None:
            md, dx0, dy0 = vd['place']
            drone = vue.raster(ff.drone_classes(md), dx0, dy0, vd['res'])
            ff.draw_frame(drone, vue, 1.0)
            ff.draw_polyline(drone, vue, chemin, ff.PATH_C, 2)
            ff.draw_markers(drone, vue, pose, objectif, depart, None)
            par = 'marker' if etat_drone.get('cale_par') == 'marker' else 'move'
            titres = (
                f'Drone: placed by the rover {par}'
                if etat_drone.get('cale_sur_cette_image')
                else f'Drone: view of {etat_drone.get("age_calage_s")} s ago (not placed now)',
                'Click: goal for the rover',
            )
        elif vd is not None:
            # not placed yet: the path and the goal on an empty map (still clickable), and
            # what the camera sees, in its own frame, as a small picture in the corner
            drone = np.full((TAILLE, TAILLE, 3), ff.PALETTE[ff.K_UNKNOWN], np.uint8)
            ff.draw_frame(drone, vue, 1.0)
            ff.draw_polyline(drone, vue, chemin, ff.PATH_C, 2)
            ff.draw_markers(drone, vue, pose, objectif, depart, None)
            h, w = vd['m'].shape
            cote = max(h, w) * vd['res']
            ox, oy = (w * vd['res'] - cote) / 2, (h * vd['res'] - cote) / 2
            pt = TAILLE // 3
            vc = ff.View(ox, oy, ox + cote, oy + cote, pt / cote)
            petit = vc.raster(ff.drone_classes(vd['m']), 0.0, 0.0, vd['res'])
            if vd['rover'] is not None:
                ff.draw_rover(petit, vc, (vd['rover'][0], vd['rover'][1], vd['rover'][2] or 0.0))
            ph, pw = petit.shape[:2]
            drone[TAILLE - ph - 8 : TAILLE - 8, 8 : 8 + pw] = petit
            cv2.rectangle(drone, (8, TAILLE - ph - 8), (8 + pw, TAILLE - 8), ff.INK_C, 1)
            titres = (
                'Drone: not placed. Drive the rover 15 cm and stop,',
                f'or show its marker (ArUco {self.args.drone_marqueur}). Corner: camera view',
            )
        else:
            drone = np.full((TAILLE, TAILLE, 3), ff.PALETTE[ff.K_UNKNOWN], np.uint8)
            ff.draw_frame(drone, vue, 1.0)
            ff.draw_markers(drone, vue, pose, objectif, depart, None)
            titres = ('Drone: ' + str(etat_drone.get('etat'))[:40], 'Click: goal for the rover')
        if vd is not None and not frais:
            titres = (titres[0], 'OLD PICTURE: ' + str(etat_drone.get('etat'))[:34])
        for k, t in enumerate(titres):
            cv2.putText(
                drone,
                t,
                (12, 24 + 22 * k),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ff.INK_C,
                1,
                cv2.LINE_AA,
            )
        val_rover = np.where(gr.src == 2, gr.val, -1)
        vr = ff.View(zr[0], zr[1], zr[2], zr[3], TAILLE / (2 * DEMI))
        rover = ff.render_fused_view(
            val_rover,
            gr.src,
            gr.j0 * gr.res,
            gr.i0 * gr.res,
            vr,
            pose,
            objectif,
            None,
            None,
            path=None,  # the planned path is drawn on the drone map and the fused map
            trail=trail,
        )
        ff.draw_frame(rover, vr, 0.5)
        pngs = {
            n: ff.encode_png(i)
            for n, i in (('rover.png', rover), ('fused.png', fused), ('drone.png', drone))
        }
        now = time.time()
        chemin = chemin or []
        longueur = sum(math.dist(chemin[k], chemin[k + 1]) for k in range(len(chemin) - 1))
        with self.lock:
            mission = dict(self.mission)
        status = {
            'etat': 'ok',
            'reel': True,
            'genere_a': now,
            'rover': {'x': pose[0] - DECALAGE, 'y': pose[1] - DECALAGE, 'cap': pose[2]},
            'objectif': [c - DECALAGE for c in objectif] if objectif else [],
            'carte_locale': bool(self.args.carte_locale),
            'source_rover': (
                'REAL rover: the obstacle grid of its depth camera (D405, computed on the '
                'rover) and odometry, fused while it drives'
                if self.args.source_locale == 'rover'
                else 'REAL rover: rover camera (DA3 perception on the VM) and odometry, '
                'views fused while standing still'
            )
            if self.args.carte_locale
            else "local map OFF: the rover's camera is not used, odometry only",
            'local': {
                **self.local,
                'garde': ((etat.get('ctrl') or {}).get('local') or {}).get('state'),
                'obstacle_dist': ((etat.get('ctrl') or {}).get('local') or {}).get(
                    'obstacle_dist'
                ),
            },
            'conduite': 'continu'
            if self.args.continu
            else 'segments'
            if self.args.segment > 0
            else 'pas',
            'drone': {
                'source': (
                    f'{self.args.drone_nom or "overhead camera"} (REAL drone): '
                    + (
                        f'placed on the map by the rover {etat_drone.get("cale_par")}, fused'
                        if etat_drone.get('cale')
                        else 'obstacles seen in its own frame, not fused until the rover has '
                        'moved and stopped in view, or its marker (ArUco '
                        f'{self.args.drone_marqueur}) is seen'
                        if etat_drone.get('etat') == 'ok'
                        else str(etat_drone.get('etat'))
                    )
                )
                if (self.args.drone_nom or self.camera is not None)
                else 'no overhead camera connected',
                'reel': bool(self.args.drone_nom or self.camera is not None),
                'calibre': bool(etat_drone.get('cale')),
                'sans_marqueur': bool(self.args.drone_sans_marqueur),
                'tuiles': n_tuiles_drone,
                'vue': etat_drone,
            },
            'tuiles_rover': n_tuiles,
            'scan': self.n_scan,
            'vues_fusionnees': self.vues,
            'derniere_vue_age_s': None
            if self.derniere_vue is None
            else round(now - self.derniere_vue, 1),
            'tourelle_pan_deg': self.pan(etat),
            'batterie_v': etat.get('battery_v'),
            'info': info,
            'mission': mission,
            'decision': {
                'action': dec.get('action') or 'OBSERVATION',
                'pourquoi': info
                if objectif
                else 'no goal: click on the drone map or the fused map',
                'numero': mission.get('pas') or self.vues,
                'age_s': 0,
                'longueur_chemin_m': round(longueur, 1),
                'distance_restante_m': dec.get('distance_restante'),
            },
        }
        with self.lock:
            for n, png in pngs.items():
                self.images[n] = (png, now)
            self.vues_img = {'fused': vue, 'drone': vue_d, 'rover': vr}
            self.status = status

    def poser_objectif(self, corps):
        if corps.get('panneau') is None:
            with self.verrou:
                self.objectif, self.decision = None, None
            return 200, {'ok': True, 'objectif': None}
        with self.lock:
            v = self.vues_img.get(str(corps['panneau']))
        if v is None:
            return 409, {
                'error': 'this map is not placed yet (the drone needs the rover marker): '
                'click on the fused map'
            }
        fx, fy = float(corps['fx']), float(corps['fy'])
        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            return 400, {'error': 'fx and fy must be between 0 and 1'}
        x = v.x0 + fx * (v.x1 - v.x0)
        y = v.y1 - fy * (v.y1 - v.y0)
        with self.verrou:
            self.objectif = (x, y)
        print(
            time.strftime('%H:%M:%S'),
            f'goal ({x - DECALAGE:.2f}, {y - DECALAGE:.2f}) m',
            flush=True,
        )
        return 200, {'ok': True, 'objectif': [x - DECALAGE, y - DECALAGE]}

    def erreur(self, texte):
        with self.lock:
            self.status = dict(self.status, etat='erreur', erreur=texte, erreur_a=time.time())

    def boucle(self):
        while True:
            t0 = time.monotonic()
            try:
                self.tour()
            except Exception as e:  # noqa: BLE001 - slow or lost link: try again
                self.erreur(f'{type(e).__name__}: {e}'[:200])
            time.sleep(max(0.2, self.args.periode - (time.monotonic() - t0)))


def handler(fr):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            nom = self.path.split('?')[0].lstrip('/')
            with fr.lock:
                if nom == 'status.json':
                    s = dict(fr.status)
                    s['mission'] = dict(fr.mission)
                    if 'genere_a' in s:
                        s['age_s'] = round(time.time() - s['genere_a'], 2)
                    image = None
                elif nom in fr.images:
                    image = fr.images[nom]
                else:
                    return self._json(404, {'error': 'unknown'})
            if image is None:
                return self._json(200, s)
            png, t = image
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(png)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Age-Ms', str(int((time.time() - t) * 1000)))
            self.end_headers()
            self.wfile.write(png)

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            brut = self.rfile.read(min(n, 10000)) if n else b''
            try:
                corps = json.loads(brut or b'{}')
                if not isinstance(corps, dict):
                    raise ValueError('a JSON object is expected')
            except ValueError as e:
                return self._json(400, {'error': str(e)[:200]})
            nom = self.path.split('?')[0]
            try:
                if nom == '/reset':
                    if fr.en_mission():
                        return self._json(409, {'error': 'a mission is running: stop it first'})
                    fr.reinitialiser()
                    fr.resets += 1
                    with fr.lock:
                        fr.images.clear()
                        fr.status = {'etat': 'reinitialise', 'genere_a': time.time()}
                    print(time.strftime('%H:%M:%S'), 'maps reset', flush=True)
                    return self._json(200, {'ok': True, 'resets': fr.resets})
                if nom == '/objectif':
                    return self._json(*fr.poser_objectif(corps))
                if nom == '/aller':
                    return self._json(*fr.aller())
                if nom == '/arreter':
                    return self._json(*fr.arreter())
                if nom == '/carte_locale':
                    return self._json(*fr.regler_carte_locale(corps))
                if nom == '/marqueur':
                    return self._json(*fr.regler_marqueur(corps))
                if nom == '/calibrer':
                    return self._json(*fr.calibrer())
            except (KeyError, TypeError, ValueError) as e:
                return self._json(400, {'error': str(e)[:200]})
            return self._json(404, {'error': 'unknown'})

    return H


def options(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--rover', default='http://127.0.0.1:18080')
    p.add_argument('--perception', default='http://127.0.0.1:8095')
    p.add_argument('--port', type=int, default=8096)
    p.add_argument(
        '--carte-locale',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="build the rover's local map from its own camera (the perception service), "
        'outside a mission and at each stop of a mission. Off for now: the maps come from '
        'the drone camera only. Also the "Local map" box of the dashboard',
    )
    p.add_argument('--periode', type=float, default=2.0, help='s between two reads')
    p.add_argument('--delai-http', type=float, default=8.0, help='s per request to the rover')
    p.add_argument(
        '--delai-liaison', type=float, default=15.0, help='s without an answer = link lost'
    )
    p.add_argument(
        '--periode-pas', type=float, default=1.0, help='s between two GET /state while driving'
    )
    p.add_argument(
        '--age-max', type=float, default=3.0, help='an older perception picture is ignored (s)'
    )
    p.add_argument(
        '--pas',
        type=float,
        default=0.15,
        help='m per mission step. The rover drives straight to the end of a step and cuts the '
        "path's corners inside it: short steps go with a small --rayon",
    )
    p.add_argument(
        '--segment',
        type=float,
        default=0.0,
        help='m: drive in segments of this length at most, several points in one command, '
        'without stopping every --pas m. 0: step by step',
    )
    p.add_argument(
        '--calibrage-distance',
        type=float,
        default=0.30,
        help='m driven forward then back by POST /calibrer (the "Calibrate direction" button)',
    )
    p.add_argument(
        '--calibrage-ecart',
        type=float,
        default=25.0,
        help='deg: the calibration is refused when forward and back disagree by more',
    )
    p.add_argument(
        '--segment-vu',
        action='store_true',
        help='with --segment: drive only through cells seen free (by the drone, or by the '
        'rover with the local map); a goal in what nobody has seen is then refused. Default: '
        'unseen cells may be crossed, as step by step',
    )
    p.add_argument(
        '--continu',
        action='store_true',
        help='drive without stopping: the path is planned again every --periode-plan s while '
        'the rover drives and sent again when it changed (the rover takes it over at speed, '
        'POST /path "id"); corners are rounded. Needs the rover server of main; turn the '
        'rover\'s local guard on (POST /config {"local_guard": 1}) so that it stops by '
        'itself before what its depth camera sees. --segment gives the longest path sent '
        '(2 m when 0). With the local map on, only cells seen free are driven',
    )
    p.add_argument(
        '--source-locale',
        choices=('service', 'rover'),
        default='service',
        help="what builds the rover's local map (--carte-locale). service: the perception "
        'service (--perception), one picture at each stop. rover: the obstacle grid the rover '
        'computes from its depth camera (its GET /local_grid), also while it drives',
    )
    p.add_argument(
        '--periode-plan', type=float, default=0.5, help='--continu: s between two plans'
    )
    p.add_argument(
        '--latence',
        type=float,
        default=0.4,
        help='--continu: s between reading the rover pose and the rover receiving the new '
        'path; the plan starts that far ahead of the rover',
    )
    p.add_argument(
        '--duree-obstacle',
        type=float,
        default=30.0,
        help="--continu: s an obstacle reported by the rover's guard stays on the map when "
        'nothing sees it again',
    )
    p.add_argument(
        '--duree-max', type=float, default=600.0, help='--continu: s at most per mission'
    )
    p.add_argument('--vitesse', type=float, default=0.12, help='m/s')
    p.add_argument(
        '--rayon',
        type=float,
        default=0.15,
        help='m kept between the centre of the rover and any obstacle: the circle around the '
        'whole rover seen from above (20 x 23 cm gives 0.152), with no margin on top. Two '
        'obstacles must be twice this apart for the rover to pass',
    )
    p.add_argument('--tolerance', type=float, default=0.25, help='arrived within (m)')
    p.add_argument('--marge-zone', type=float, default=1.5, help='m around the planned route')
    p.add_argument('--max-pas', type=int, default=80, help='steps at most per mission')
    p.add_argument(
        '--drone-nom',
        default=None,
        help='name of the real drone camera (e.g. the iPhone above the area), shown as the '
        'drone source; not fused until calibrated',
    )
    p.add_argument(
        '--drone-source',
        default=None,
        help="the overhead camera's stream or snapshot URL (e.g. the dashboard's "
        'http://127.0.0.1:8082/drone_feed): its obstacles are drawn on the drone map, and '
        "fused once the rover's ArUco marker was seen. Read only. Default: none",
    )
    p.add_argument('--drone-periode', type=float, default=1.0, help='s between two pictures')
    p.add_argument(
        '--drone-hauteur',
        type=float,
        default=1.8,
        help='camera height (m): only a starting point, it need not be known. The scale '
        "is corrected on the rover's first move, and the height found is in the status",
    )
    p.add_argument(
        '--drone-champ',
        type=float,
        default=108.0,
        help='horizontal field of the picture (deg), for the scale with --drone-hauteur '
        '(108: iPhone ultrawide)',
    )
    p.add_argument(
        '--drone-mpp', type=float, default=None, help='metres per pixel, measured (preferred)'
    )
    p.add_argument(
        '--drone-largeur-min',
        type=float,
        default=0.15,
        help='m: what the drone sees thinner than this is a line on the floor, not an obstacle',
    )
    p.add_argument(
        '--drone-detecteur',
        choices=('marqueur', 'ia'),
        default='marqueur',
        help="how the rover is found in the drone's picture. marqueur: its ArUco marker, read "
        '(it must be about 25 px wide). ia: a vision model (OWLv2, sim/rover_du_dessus.py; '
        "needs transformers) finds the rover, the marker's white paper gives the heading",
    )
    p.add_argument(
        '--drone-taille-rover',
        type=float,
        default=0.25,
        help='m: size of the rover seen from above, for the scale with --drone-detecteur ia',
    )
    p.add_argument(
        '--drone-marqueur-devant',
        action='store_true',
        help='the marker paper is at the front of the rover (default: at the back)',
    )
    p.add_argument(
        '--drone-ombres',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cells of the drone's picture that are only a shadow on the floor (darker, but of "
        'the floor colour) are free, not obstacles, with either detector',
    )
    p.add_argument(
        '--drone-cumul',
        action='store_true',
        help='the drone camera moves: every placed picture adds to the drone map, which '
        'grows, instead of replacing it (reset the maps to start again). A picture is placed '
        'only while the rover is in it',
    )
    p.add_argument('--drone-marqueur', type=int, default=10, help="ArUco id on the rover's top")
    p.add_argument(
        '--drone-sans-marqueur',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="no marker on the rover: the heading read in the drone's picture is ignored, and "
        "the drone map is placed by the rover's own move (drive it at least --drone-pas-min "
        'and stop, twice). On for now; --no-drone-sans-marqueur trusts the marker again',
    )
    p.add_argument(
        '--drone-pas-min',
        type=float,
        default=0.12,
        help='m: without a marker, the rover must have moved this far between two stops '
        'for its move to place the drone map',
    )
    p.add_argument(
        '--drone-hauteur-marqueur', type=float, default=0.0, help='marker height above floor (m)'
    )
    args = p.parse_args(argv)
    args.rover = args.rover.rstrip('/')
    return args


def main():
    args = options()
    fr = FusionReelle(args)
    threading.Thread(target=fr.boucle, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), handler(fr))
    print(
        f'fusion reelle : http://127.0.0.1:{args.port}/status.json  rover {args.rover}', flush=True
    )
    srv.serve_forever()


if __name__ == '__main__':
    main()
