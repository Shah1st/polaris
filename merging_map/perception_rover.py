"""
perception_rover.py - Ce que voit la camera du rover, pour la fusion (le percevoir de
rover_reel.RaspRoverReel).

Le service rover_mapper/perception_service.py donne, a la demande, une grille -1/0/1 de cases
de 5 cm dans le repere du ROVER (x devant, y a gauche, metres ; case [i][j] en
x = x0 + j*res, y = y0 + i*res). Ici on la tourne et la deplace a la position et au cap donnes,
puis on la recopie sur la grille de la carte (cases de 5 cm, case [i][j] en
X = x0 + j*0.05, Y = y0 + i*0.05, x0 et y0 multiples de 0.05).

Une case qu'on ne connait pas reste -1, jamais 0 : un 0 du rover efface les obstacles du drone
(README). Pour la meme raison, toute erreur, reponse trop vieille ou cap inconnu donne None :
mieux vaut ne rien dire que de dire faux.

    percevoir = percevoir_http('http://127.0.0.1:8095')
    vue = percevoir((x, y), cap)   # -> (matrice, x0, y0) ou None

!! PAS ENCORE ESSAYE AVEC LA CAMERA DU VRAI ROVER (seulement en simulation). !!
"""

import json
import math
import urllib.error
import urllib.request

import numpy as np

RES = 0.05  # case la plus fine de la fusion (niveau z = 4)


def vers_carte(cases, x0, y0, res, position, cap, res_carte=RES, avance_camera=0.0):
    """Grille -1/0/1 du repere rover -> (matrice, X0, Y0) sur la grille de la carte.

    cap : radians, sens trigonometrique depuis l'axe x de la carte (direction de l'axe optique).
    avance_camera : la camera est a tant de metres devant `position`, dans la direction du cap.
    Chaque case de la carte prend la valeur de la case rover qui contient son centre ; puis
    chaque case obstacle du rover marque la case de la carte ou tombe son centre (une case
    obstacle tournee de 45 degres peut ne contenir aucun centre de case de la carte : on ne
    perd pas un obstacle fin)."""
    m = np.asarray(cases, dtype=np.int8)
    if m.ndim != 2 or m.size == 0:
        return None
    n_lignes, n_cols = m.shape
    c, s = math.cos(cap), math.sin(cap)
    px = position[0] + avance_camera * c
    py = position[1] + avance_camera * s

    def carte(x, y):
        return px + c * x - s * y, py + s * x + c * y

    xs = np.array([x0, x0 + n_cols * res, x0, x0 + n_cols * res])
    ys = np.array([y0, y0, y0 + n_lignes * res, y0 + n_lignes * res])
    xm, ym = carte(xs, ys)
    eps = 1e-9
    j0, j1 = math.floor(xm.min() / res_carte + eps), math.ceil(xm.max() / res_carte - eps)
    i0, i1 = math.floor(ym.min() / res_carte + eps), math.ceil(ym.max() / res_carte - eps)
    out = np.full((i1 - i0, j1 - j0), -1, np.int8)

    # centres des cases de la carte -> repere rover
    jc = (np.arange(j0, j1) + 0.5) * res_carte
    ic = (np.arange(i0, i1) + 0.5) * res_carte
    gx, gy = np.meshgrid(jc, ic)
    dx, dy = gx - px, gy - py
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    jj = np.floor((xr - x0) / res + eps).astype(np.int64)
    ii = np.floor((yr - y0) / res + eps).astype(np.int64)
    dedans = (ii >= 0) & (ii < n_lignes) & (jj >= 0) & (jj < n_cols)
    out[dedans] = m[ii[dedans], jj[dedans]]

    oi, oj = np.nonzero(m == 1)
    if len(oi):
        xo, yo = carte(x0 + (oj + 0.5) * res, y0 + (oi + 0.5) * res)
        bj = np.floor(xo / res_carte + eps).astype(np.int64) - j0
        bi = np.floor(yo / res_carte + eps).astype(np.int64) - i0
        ok = (bi >= 0) & (bi < out.shape[0]) & (bj >= 0) & (bj < out.shape[1])
        out[bi[ok], bj[ok]] = 1
    return out, j0 * res_carte, i0 * res_carte


def percevoir_http(
    url, delai=3.0, age_max=2.0, fraiche=True, avance_camera=0.0, ouvrir=urllib.request.urlopen
):
    """percevoir(position, cap) qui interroge le service de perception a `url`.

    delai : attente maximale de la reponse (s). age_max : une image plus vieille ne compte
    pas. fraiche : le service attend une image arrivee apres la demande (le rover vient de
    s'arreter : une image d'avant montrerait un rover qui roule encore). `ouvrir` : pour les
    tests."""
    base = url.rstrip('/')
    chemin = base + '/obstacles' + ('?fresh=1' if fraiche else '')

    def percevoir(position, cap):
        if cap is None or position is None:
            return None
        try:
            with ouvrir(chemin, timeout=delai) as r:
                if r.status != 200:
                    return None
                rep = json.loads(r.read())
            # age mesure par le service lui-meme : juste meme s'il tourne sur un autre
            # ordinateur dont l'horloge differe
            age = float(rep.get('age_s', math.inf))
            if not math.isfinite(age) or age > age_max:
                return None
            res = float(rep['res'])
            return vers_carte(
                rep['cells'],
                float(rep['x0']),
                float(rep['y0']),
                res,
                position,
                float(cap),
                avance_camera=avance_camera,
            )
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
            return None

    return percevoir
