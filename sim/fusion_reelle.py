"""
fusion_reelle.py - Les cartes du VRAI rover pour le tableau de bord (teleop/server.py --fusion),
au meme format que sim/fusion_live.py, et une mission vers un point choisi sur la carte.

    GET  /rover.png /drone.png /fused.png /status.json   les cartes et l'etat
    POST /reset              cartes videes ; nouveau repere a la position actuelle du rover
    POST /objectif           {"panneau": "fused"|"drone", "fx": 0..1, "fy": 0..1} : le point
                             clique sur une carte (fraction de sa largeur et de sa hauteur,
                             depuis le coin en haut a gauche) ; {} efface le point
    POST /aller              le rover part seul vers le point (pas bornes, un regard de sa camera
                             a chaque arret, contournement des obstacles vus)
    POST /arreter            fin de la mission et POST /stop au rover

Hors mission : lecture seule (GET /state du rover, GET /obstacles du service de perception).
Repere : celui de l'odometrie au lancement (ou au dernier /reset), origine au point de depart,
x vers l'avant du rover a ce moment. Une vue n'est fusionnee que rover immobile.

    python sim/fusion_reelle.py --rover http://127.0.0.1:18080 \\
        --perception http://127.0.0.1:8095 --port 8096
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

DECALAGE = 20.0  # m : coordonnees positives pour la grille des tuiles
DEMI = 2.5  # m : fenetre de la carte du rover autour de lui
TAILLE = 480  # px : cote des images


def angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class Odometrie:
    """Tient lieu de camera du dessus pour RaspRoverReel : la pose du rover dans le repere des
    cartes, tiree de GET /state."""

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
        self.verrou = threading.RLock()  # la carte, le repere et la trace
        self.lock = threading.Lock()  # les images et l'etat servis
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
        """Carte vide, trace effacee, nouveau repere : origine = la pose du rover au prochain
        tour (POST /reset)."""
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

    # ---- repere ----------------------------------------------------------------------
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
        """Pan de la tourelle de la camera (deg, + = a droite pour le serveur du rover)."""
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

    # ---- hors mission : observer --------------------------------------------------------
    def tour(self):
        etat = self.requete('GET', '/state')
        if etat is None:
            raise RuntimeError('le rover ne repond pas a GET /state')
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
            info = 'mission en cours : le rover regarde a chaque arret'
        elif not immobile:
            info = 'rover en mouvement : vue non fusionnee'
        else:
            vue = self.percevoir(pose[:2], angle(pose[2] - math.radians(self.pan(etat))))
            if vue is None:
                info = 'perception sans image recente'
            else:
                self.publier(vue)
                info = 'vue fusionnee'
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
                print(time.strftime('%H:%M:%S'), 'mission :', texte, flush=True)
            if pas is not None:
                self.mission['pas'] = pas

    def aller(self):
        if self.objectif is None:
            return 409, {'error': "pas de point d'arrivee : cliquer sur une carte d'abord"}
        if self.en_mission():
            return 409, {'error': 'une mission est deja en cours'}
        self.arret_demande.clear()
        self.journal_mission('en_cours', 'depart', 0)
        self.fil_mission = threading.Thread(target=self._mission, daemon=True)
        self.fil_mission.start()
        return 200, {'ok': True}

    def arreter(self):
        self.arret_demande.set()
        ok = arreter_le_rover(self.requete, journal=lambda t: print(t, flush=True))
        if self.fil_mission is not None:
            self.fil_mission.join(timeout=1.0)
        self.journal_mission('arret', 'arretee depuis le tableau de bord (/stop envoye)')
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
                raise RuntimeError('le rover ne repond pas a GET /state')
            rover.placer(depart[:2])
            rover.recaler()
            issue = self._boucle_mission(rover)
        except ArretOperateur as e:
            issue = 'arret_operateur'
            self.journal_mission(texte=f'ARRET OPERATEUR : {e}')
        except (ServeurRoverError, RuntimeError) as e:
            issue = 'erreur'
            self.journal_mission(texte=f'ERREUR : {e}')
        except Exception as e:  # noqa: BLE001 - la mission s'arrete, le rover aussi
            issue = 'erreur'
            self.journal_mission(texte=f'ERREUR : {type(e).__name__}: {e}')
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
                self.journal_mission(texte='mission arretee')
                return 'arret'
            pos = rover.position()
            reste = math.dist(pos, objectif)
            if reste <= a.tolerance:
                self.journal_mission(texte=f'ARRIVE a {reste:.2f} m du point, {k} pas')
                return 'arrive'
            pan = math.radians(self.pan(self.dernier_etat))
            cap = None if rover.cap is None else angle(rover.cap - pan)
            vue = self.percevoir_frais(rover.est, cap)
            if vue is None:
                sans_vue += 1
                self.journal_mission(texte=f'perception indisponible ({sans_vue}/3) : on attend')
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
                self.journal_mission(texte='BLOQUE : aucun passage vers le point')
                return 'bloque'
            if len(chemin) < 2:
                if reste <= a.tolerance + 0.1:  # le A* arrive a une case de 5 cm pres
                    self.journal_mission(texte=f'ARRIVE a {reste:.2f} m du point, {k} pas')
                    return 'arrive'
                self.journal_mission(texte='BLOQUE : pas de chemin')
                return 'bloque'
            cible, _ = avancer(pos, list(chemin), a.pas)
            self.journal_mission(
                texte=f'pas {k + 1} : {dec["action"]}, reste {reste:.2f} m', pas=k + 1
            )
            if self.arret_demande.is_set():
                return 'arret'
            impact = rover.deplacer(cible)
            if impact is not None:
                with self.verrou:
                    rover.signaler_choc(impact)
                self.journal_mission(texte='CHOC : obstacle note sur la carte')
        self.journal_mission(texte=f'{a.max_pas} pas sans arriver')
        return 'max_pas'

    # ---- images et etat --------------------------------------------------------------
    def zone_vue(self, pose):
        xs = [p[0] for p in self.trail] + [pose[0]]
        ys = [p[1] for p in self.trail] + [pose[1]]
        if self.objectif is not None:
            xs.append(self.objectif[0])
            ys.append(self.objectif[1])
        m = 2.5
        x0, x1 = min(xs) - m, max(xs) + m
        y0, y1 = min(ys) - m, max(ys) + m
        cote = max(x1 - x0, y1 - y0, 5.0)  # carre : l'image garde ses proportions
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
        # le panneau du drone : meme cadre, seulement ce que le drone a vu (rien aujourd'hui)
        val_drone = np.where(g.src == 1, g.val, -1)
        drone = ff.render_fused_view(
            val_drone, g.src, gx0, gy0, vue, pose, objectif, depart, None, path=chemin
        )
        ff.draw_frame(drone, vue, 1.0)
        for k, t in enumerate(('Pas de camera du dessus branchee', "Clic : point d'arrivee")):
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
            'source_rover': 'VRAI rover : camera du rover (perception DA3 sur la VM) et '
            "odometrie, vues fusionnees a l'arret",
            'drone': {'source': 'aucune camera du dessus branchee', 'tuiles': 0},
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
                else "pas de point d'arrivee : clic sur la carte du drone ou fusionnee",
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
            return 409, {'error': "pas encore d'image de cette carte"}
        fx, fy = float(corps['fx']), float(corps['fy'])
        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            return 400, {'error': 'fx et fy entre 0 et 1'}
        x = v.x0 + fx * (v.x1 - v.x0)
        y = v.y1 - fy * (v.y1 - v.y0)
        with self.verrou:
            self.objectif = (x, y)
        print(
            time.strftime('%H:%M:%S'),
            f'objectif ({x - DECALAGE:.2f}, {y - DECALAGE:.2f}) m',
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
            except Exception as e:  # noqa: BLE001 - lien lent ou coupe : on reessaie
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
                    return self._json(404, {'error': 'inconnu'})
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
                    raise ValueError('objet JSON attendu')
            except ValueError as e:
                return self._json(400, {'error': str(e)[:200]})
            nom = self.path.split('?')[0]
            try:
                if nom == '/reset':
                    if fr.en_mission():
                        return self._json(409, {'error': "mission en cours : l'arreter d'abord"})
                    fr.reinitialiser()
                    fr.resets += 1
                    with fr.lock:
                        fr.images.clear()
                        fr.status = {'etat': 'reinitialise', 'genere_a': time.time()}
                    print(time.strftime('%H:%M:%S'), 'cartes reinitialisees', flush=True)
                    return self._json(200, {'ok': True, 'resets': fr.resets})
                if nom == '/objectif':
                    return self._json(*fr.poser_objectif(corps))
                if nom == '/aller':
                    return self._json(*fr.aller())
                if nom == '/arreter':
                    return self._json(*fr.arreter())
            except (KeyError, TypeError, ValueError) as e:
                return self._json(400, {'error': str(e)[:200]})
            return self._json(404, {'error': 'inconnu'})

    return H


def options(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--rover', default='http://127.0.0.1:18080')
    p.add_argument('--perception', default='http://127.0.0.1:8095')
    p.add_argument('--port', type=int, default=8096)
    p.add_argument('--periode', type=float, default=2.0, help='s entre deux lectures')
    p.add_argument('--delai-http', type=float, default=8.0, help='s par requete au rover')
    p.add_argument('--delai-liaison', type=float, default=15.0, help='s sans reponse = coupure')
    p.add_argument(
        '--periode-pas', type=float, default=1.0, help='s entre deux GET /state en roulant'
    )
    p.add_argument(
        '--age-max', type=float, default=3.0, help='image de perception plus vieille : ignoree'
    )
    p.add_argument('--pas', type=float, default=0.25, help='m par pas de la mission')
    p.add_argument('--vitesse', type=float, default=0.12, help='m/s')
    p.add_argument('--rayon', type=float, default=0.20, help='rayon du rover + marge (m)')
    p.add_argument('--tolerance', type=float, default=0.25, help='arrivee a moins de (m)')
    p.add_argument('--marge-zone', type=float, default=1.5, help='m autour du trajet planifie')
    p.add_argument('--max-pas', type=int, default=80)
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
