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

Outside a mission: read only (GET /state of the rover, GET /obstacles of the perception service).
Frame: the odometry frame at start (or at the last /reset), origin at the start point, x towards
the front of the rover at that time. A view is fused only while the rover stands still.
The drone (--drone-nom: e.g. the iPhone above the area) is declared as the real drone source;
its picture is not fused into the rover's map until it is calibrated on the floor (scale, and
the rover located in its picture).

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
        self.reinitialiser()

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
        if self.en_mission():
            info = 'mission running: the rover looks at each stop'
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
            odo = Odometrie(self, self.requete)
            rover = RaspRoverReel(
                self.carte,
                self.requete,
                position_drone=odo.pose,
                percevoir=self.percevoir_frais,
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
            pan = math.radians(self.pan(self.dernier_etat))
            cap = None if rover.cap is None else angle(rover.cap - pan)
            vue = self.percevoir_frais(rover.est, cap)
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
        m = 2.5
        x0, x1 = min(xs) - m, max(xs) + m
        y0, y1 = min(ys) - m, max(ys) + m
        cote = max(x1 - x0, y1 - y0, 5.0)  # square: the picture keeps its proportions
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return cx - cote / 2, cy - cote / 2, cx + cote / 2, cy + cote / 2

    def dessiner(self, pose, info, etat):
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
        gx0, gy0 = g.j0 * g.res, g.i0 * g.res
        vue = ff.View(z[0], z[1], z[2], z[3], TAILLE / (z[2] - z[0]))
        depart = trail[0] if trail else None
        chemin = dec.get('chemin')
        fused = ff.render_fused_view(
            g.val, g.src, gx0, gy0, vue, pose, objectif, depart, None, path=chemin, trail=trail
        )
        ff.draw_frame(fused, vue, 1.0)
        # the drone panel: same frame, only what the drone saw (nothing until it is calibrated)
        val_drone = np.where(g.src == 1, g.val, -1)
        drone = ff.render_fused_view(
            val_drone, g.src, gx0, gy0, vue, pose, objectif, depart, None, path=chemin
        )
        ff.draw_frame(drone, vue, 1.0)
        titre = (
            f'Drone: {self.args.drone_nom} - not calibrated, not fused yet'
            if self.args.drone_nom
            else 'No overhead camera fused yet'
        )
        for k, t in enumerate((titre, 'Click: goal for the rover')):
            cv2.putText(
                drone,
                t,
                (16, 28 + 26 * k),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                ff.INK_C,
                2,
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
            path=chemin,
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
            'source_rover': 'REAL rover: rover camera (DA3 perception on the VM) and odometry, '
            'views fused while standing still',
            'drone': {
                'source': (
                    f'{self.args.drone_nom} (REAL drone; not calibrated on the floor yet: '
                    'its picture is not fused)'
                )
                if self.args.drone_nom
                else 'no overhead camera connected',
                'reel': bool(self.args.drone_nom),
                'calibre': False,
                'tuiles': 0,
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
            self.vues_img = {'fused': vue, 'drone': vue, 'rover': vr}
            self.status = status

    def poser_objectif(self, corps):
        if corps.get('panneau') is None:
            with self.verrou:
                self.objectif, self.decision = None, None
            return 200, {'ok': True, 'objectif': None}
        with self.lock:
            v = self.vues_img.get(str(corps['panneau']))
        if v is None:
            return 409, {'error': 'no picture of this map yet'}
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
            except (KeyError, TypeError, ValueError) as e:
                return self._json(400, {'error': str(e)[:200]})
            return self._json(404, {'error': 'unknown'})

    return H


def options(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--rover', default='http://127.0.0.1:18080')
    p.add_argument('--perception', default='http://127.0.0.1:8095')
    p.add_argument('--port', type=int, default=8096)
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
    p.add_argument('--pas', type=float, default=0.25, help='m per mission step')
    p.add_argument('--vitesse', type=float, default=0.12, help='m/s')
    p.add_argument('--rayon', type=float, default=0.20, help='rover radius + margin (m)')
    p.add_argument('--tolerance', type=float, default=0.25, help='arrived within (m)')
    p.add_argument('--marge-zone', type=float, default=1.5, help='m around the planned route')
    p.add_argument('--max-pas', type=int, default=80, help='steps at most per mission')
    p.add_argument(
        '--drone-nom',
        default=None,
        help='name of the real drone camera (e.g. the iPhone above the area), shown as the '
        'drone source; not fused until calibrated',
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
