"""
fusion_reelle.py - Les cartes du VRAI rover pour le tableau de bord (teleop/server.py --fusion),
au meme format que sim/fusion_live.py : GET /rover.png /drone.png /fused.png /status.json.

Lecture seule : GET /state du rover (odometrie, tourelle) et GET /obstacles du service de
perception (camera du rover, Depth Anything 3 sur la VM). Aucune commande au rover.
Repere : celui de l'odometrie au lancement, origine au point de depart (+DECALAGE m), x vers
l'avant du rover au depart. Une vue n'est fusionnee que rover immobile (image recente, pose
stable), pour ne jamais poser une image prise en roulant.

    python sim/fusion_reelle.py --rover http://127.0.0.1:18080 \\
        --perception http://127.0.0.1:8095 \\
        --port 8096 [--objectif X Y]
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE + '/merging_map')
sys.path.insert(0, RACINE + '/sim')

from fusion_carte import CarteFusion, decouper  # noqa: E402
from perception_rover import percevoir_http  # noqa: E402

import fusion_feed as ff  # noqa: E402

DECALAGE = 20.0  # m : coordonnees positives pour la grille des tuiles
DEMI = 2.5  # m : fenetre de la carte du rover autour de lui


def lire_json(url, delai):
    with urllib.request.urlopen(url, timeout=delai) as r:
        return json.loads(r.read())


def angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class FusionReelle:
    def __init__(self, args):
        self.args = args
        self.reinitialiser()
        self.percevoir = percevoir_http(
            args.perception, delai=6.0, age_max=args.age_max, fraiche=False
        )
        self.lock = threading.Lock()
        self.images = {}
        self.status = {'etat': 'demarrage', 'genere_a': time.time()}
        self.resets = 0
        self.verrou_tour = threading.Lock()  # un reset n'arrive jamais au milieu d'un tour

    def reinitialiser(self):
        """Carte vide, trace effacee, nouveau repere : origine = la pose du rover au prochain
        tour (POST /reset)."""
        self.carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
        self.origine = None
        self.pose = None
        self.trail = []
        self.vues = 0
        self.derniere_vue = None
        self.dernier_etat = None
        self.precedente = None
        self.n_scan = {'points': 0, 'free': 0, 'obstacle': 0}

    # ---- lecture du rover -------------------------------------------------------------
    def pose_carte(self, etat):
        o = etat['odom']
        x, y, h = float(o['x']), float(o['y']), math.radians(float(o['heading_deg']))
        if self.origine is None:
            self.origine = (x, y, h)
        x0, y0, h0 = self.origine
        c, s = math.cos(-h0), math.sin(-h0)
        dx, dy = x - x0, y - y0
        return DECALAGE + c * dx - s * dy, DECALAGE + s * dx + c * dy, angle(h - h0)

    def tour(self):
        etat = lire_json(self.args.rover + '/state', self.args.delai_http)
        pose = self.pose_carte(etat)
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
        pan = float(((etat.get('gimbal') or {}).get('pan')) or 0.0)
        info = 'rover en mouvement : vue non fusionnee'
        if immobile:
            vue = self.percevoir(pose[:2], angle(pose[2] - math.radians(pan)))
            if vue is None:
                info = 'perception sans image recente'
            else:
                m, x0, y0 = vue
                for msg in decouper(m, x0, y0, 4, self.carte, 'rover', t=time.monotonic()):
                    self.carte.recevoir(msg)
                self.vues += 1
                self.derniere_vue = time.time()
                self.n_scan = {
                    'points': int(m.size),
                    'free': int((m == 0).sum()),
                    'obstacle': int((m == 1).sum()),
                }
                info = 'vue fusionnee'
        self.dessiner(pose, pan, info, etat)

    # ---- images et etat --------------------------------------------------------------
    def zone(self, pose):
        xs = [p[0] for p in self.trail] + [pose[0]]
        ys = [p[1] for p in self.trail] + [pose[1]]
        if self.args.objectif:
            xs.append(self.args.objectif[0])
            ys.append(self.args.objectif[1])
        m = 2.5
        x0, x1 = min(xs) - m, max(xs) + m
        y0, y1 = min(ys) - m, max(ys) + m
        cote = max(x1 - x0, y1 - y0, 5.0)  # carre : l'image garde ses proportions
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return cx - cote / 2, cy - cote / 2, cx + cote / 2, cy + cote / 2

    def dessiner(self, pose, pan, info, etat):
        z = self.zone(pose)
        g = self.carte.grille(*z)
        gx0, gy0 = g.j0 * g.res, g.i0 * g.res
        ppm = 480.0 / (z[2] - z[0])
        vue = ff.View(z[0], z[1], z[2], z[3], ppm)
        objectif = tuple(self.args.objectif) if self.args.objectif else None
        decision = None
        if objectif is not None:
            decision = self.carte.decider(
                pose[:2], objectif, zone=z, tolerance=0.25, drone_disponible=False
            )
        fused = ff.render_fused_view(
            g.val,
            g.src,
            gx0,
            gy0,
            vue,
            pose,
            objectif,
            self.trail[0] if self.trail else None,
            None,
            path=(decision or {}).get('chemin'),
            trail=self.trail,
        )
        ff.draw_frame(fused, vue, 1.0)
        # carte du rover : sa propre couche, autour de lui
        zr = (pose[0] - DEMI, pose[1] - DEMI, pose[0] + DEMI, pose[1] + DEMI)
        gr = self.carte.grille(*zr)
        val = np.where(gr.src == 2, gr.val, -1)
        vr = ff.View(zr[0], zr[1], zr[2], zr[3], 480.0 / (2 * DEMI))
        rover = ff.render_fused_view(
            val,
            gr.src,
            gr.j0 * gr.res,
            gr.i0 * gr.res,
            vr,
            pose,
            objectif,
            None,
            None,
            trail=self.trail,
        )
        ff.draw_frame(rover, vr, 0.5)
        drone = np.full((480, 480, 3), ff.UNKNOWN_C, np.uint8)
        for k, t in enumerate(
            (
                'Pas de camera du dessus',
                "branchee pour l'instant",
                '(telephone : teleop --drone URL)',
            )
        ):
            cv2.putText(
                drone,
                t,
                (30, 200 + 34 * k),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                ff.INK_C,
                2,
                cv2.LINE_AA,
            )
        pngs = {
            n: ff.encode_png(i)
            for n, i in (('rover.png', rover), ('fused.png', fused), ('drone.png', drone))
        }
        now = time.time()
        dec = decision or {}
        chemin = dec.get('chemin') or []
        longueur = sum(math.dist(chemin[k], chemin[k + 1]) for k in range(len(chemin) - 1))
        status = {
            'etat': 'ok',
            'reel': True,
            'genere_a': now,
            'rover': {'x': pose[0] - DECALAGE, 'y': pose[1] - DECALAGE, 'cap': pose[2]},
            'objectif': [c - DECALAGE for c in objectif] if objectif else [],
            'source_rover': 'VRAI rover : camera du rover (perception DA3 sur la VM) et '
            "odometrie, vues fusionnees a l'arret",
            'drone': {'source': 'aucune camera du dessus branchee', 'tuiles': 0},
            'tuiles_rover': len(self.carte.tuiles['rover']),
            'scan': self.n_scan,
            'vues_fusionnees': self.vues,
            'derniere_vue_age_s': None
            if self.derniere_vue is None
            else round(now - self.derniere_vue, 1),
            'tourelle_pan_deg': pan,
            'batterie_v': etat.get('battery_v'),
            'info': info,
            'decision': {
                'action': dec.get('action') or 'OBSERVATION',
                'pourquoi': info if objectif else "pas d'objectif : cartes en direct seulement",
                'numero': self.vues,
                'age_s': 0,
                'longueur_chemin_m': round(longueur, 1),
                'distance_restante_m': dec.get('distance_restante'),
            },
        }
        with self.lock:
            for n, png in pngs.items():
                self.images[n] = (png, now)
            self.status = status

    def erreur(self, texte):
        with self.lock:
            self.status = dict(self.status, etat='erreur', erreur=texte, erreur_a=time.time())

    def boucle(self):
        while True:
            t0 = time.monotonic()
            try:
                with self.verrou_tour:
                    self.tour()
            except Exception as e:  # noqa: BLE001 - lien lent ou coupe : on reessaie
                self.erreur(f'{type(e).__name__}: {e}'[:200])
            time.sleep(max(0.2, self.args.periode - (time.monotonic() - t0)))


def handler(fr):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            nom = self.path.split('?')[0].lstrip('/')
            with fr.lock:
                if nom == 'status.json':
                    s = dict(fr.status)
                    if 'genere_a' in s:
                        s['age_s'] = round(time.time() - s['genere_a'], 2)
                    body, ctype, age = json.dumps(s).encode(), 'application/json', 0.0
                elif nom in fr.images:
                    png, t = fr.images[nom]
                    body, ctype, age = png, 'image/png', time.time() - t
                else:
                    body, ctype, age = b'{"error": "inconnu"}', 'application/json', None
            self.send_response(200 if age is not None else 404)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            if age is not None:
                self.send_header('X-Age-Ms', str(int(age * 1000)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            if n:
                self.rfile.read(min(n, 10000))
            if self.path.split('?')[0] != '/reset':
                code, body = 404, b'{"error": "inconnu"}'
            else:
                with fr.verrou_tour:
                    fr.reinitialiser()
                    fr.resets += 1
                with fr.lock:
                    fr.images.clear()
                    fr.status = {'etat': 'reinitialise', 'genere_a': time.time()}
                print(time.strftime('%H:%M:%S'), 'cartes reinitialisees', flush=True)
                code, body = 200, json.dumps({'ok': True, 'resets': fr.resets}).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return H


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--rover', default='http://127.0.0.1:18080')
    p.add_argument('--perception', default='http://127.0.0.1:8095')
    p.add_argument('--port', type=int, default=8096)
    p.add_argument('--periode', type=float, default=2.0, help='s entre deux lectures')
    p.add_argument('--delai-http', type=float, default=8.0)
    p.add_argument(
        '--age-max', type=float, default=3.0, help='image de perception plus vieille : ignoree'
    )
    p.add_argument(
        '--objectif',
        type=float,
        nargs=2,
        metavar=('X', 'Y'),
        help='m dans le repere du depart (x devant) : affiche le chemin prevu',
    )
    args = p.parse_args()
    args.rover = args.rover.rstrip('/')
    if args.objectif:
        args.objectif = [args.objectif[0] + DECALAGE, args.objectif[1] + DECALAGE]
    fr = FusionReelle(args)
    threading.Thread(target=fr.boucle, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), handler(fr))
    print(
        f'fusion reelle : http://127.0.0.1:{args.port}/status.json  rover {args.rover}', flush=True
    )
    srv.serve_forever()


if __name__ == '__main__':
    main()
