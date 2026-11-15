"""
fusion_carte.py - Fusion des cartes drone + rover en tuiles (style Google Maps)
et decision en cas de blocage.

Convention des cases :  1 = obstacle,  0 = libre,  -1 = inconnu (pas vu)

Tuiles
------
- Reperees par (z, x, y). z = 0 est le niveau le plus grossier, z = ZMAX le plus fin.
- Chaque tuile est une matrice N x N (meme N a tous les niveaux).
- Taille d'une case au niveau z :  res(z) = RES_FINE * 2**(ZMAX - z)
- La tuile (z, x, y) couvre  x*T <= X < (x+1)*T  et  y*T <= Y < (y+1)*T,  avec T = N * res(z)
- data[i][j] = case de coordonnees  X = x*T + j*res(z),  Y = y*T + i*res(z)
  -> la ligne i va vers +Y, la colonne j vers +X. La ligne 0 est au SUD
     (et pas en haut comme dans une image).

Message attendu (drone ou rover)
--------------------------------
    {"source": "drone" | "rover", "t": <secondes>, "z": int, "x": int, "y": int,
     "data": matrice N x N de -1 / 0 / 1}

Plusieurs appareils d'un meme role : "source": "drone:<id>" ou "rover:<id>" (le role est
avant les deux-points). Les tuiles de chaque appareil sont gardees a part, puis fusionnees
par role ("drone" et "rover" tout courts sont des appareils comme les autres).

Regles de fusion
----------------
- Une case INCONNUE (-1) ne modifie jamais rien.
- Meme source, meme tuile : la valeur la plus recente l'emporte.
- Meme role, niveaux differents : le niveau le plus fin l'emporte.
- Meme role, meme niveau, appareils differents : la tuile la plus recente l'emporte.
- Entre roles : une case connue d'un ROVER l'emporte sur les DRONES.
- Passage a un niveau plus grossier : un seul obstacle suffit a bloquer la case (max).
"""

import heapq
import math

import numpy as np

try:  # composantes connexes en C : une victime inatteignable coute des ms, pas des secondes
    import cv2
except ImportError:  # sans OpenCV, meme resultat par un parcours en Python, plus lent
    cv2 = None

OBSTACLE, LIBRE, INCONNU = 1, 0, -1
SOURCES = ('drone', 'rover')  # les roles ; une source est un role ou '<role>:<id>'
SRC_AUCUNE, SRC_DRONE, SRC_ROVER = 0, 1, 2


def role_source(source):
    """'drone', 'rover', 'drone:<id>' ou 'rover:<id>' -> 'drone' ou 'rover' ; ValueError
    sinon."""
    if not isinstance(source, str):
        raise ValueError(f'source inconnue : {source!r}')
    role, deux_points, ident = source.partition(':')
    if role not in SOURCES or (deux_points and not ident):
        raise ValueError(f'source inconnue : {source!r}')
    return role


# ----------------------------------------------------------------------------
# Outils grille
# ----------------------------------------------------------------------------
def dilater(masque, r):
    """Dilatation d'un masque booleen par un disque de rayon r (en cases)."""
    out = masque.copy()
    if r <= 0:
        return out
    nh, nw = masque.shape
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if (di == 0 and dj == 0) or di * di + dj * dj > r * r:
                continue
            out[max(di, 0) : nh + min(di, 0), max(dj, 0) : nw + min(dj, 0)] |= masque[
                max(-di, 0) : nh + min(-di, 0), max(-dj, 0) : nw + min(-dj, 0)
            ]
    return out


def gonfler(masque, rayon, res):
    """Gonfle un masque booleen d'un rayon `rayon` (m) : somme de Minkowski EXACTE des cases
    obstacles (des carres de cote `res`) avec le disque du robot, evaluee au centre des cases.

    Une case libre est bloquee si son CENTRE est a `rayon` ou moins du carre d'une case
    obstacle. Le robot peut alors etre traite comme un POINT : tout chemin fait de centres de
    cases libres laisse au moins `rayon` entre le robot et tout obstacle. Contrairement a
    dilater(masque, ceil(rayon / res)), qui compte en distance entre centres, ca ne laisse pas
    passer le robot trop pres d'un coin en diagonale (case grosse) et ne bloque pas plus qu'il
    ne faut (case grosse aussi)."""
    out = masque.copy()
    if rayon <= 0:
        return out
    nh, nw = masque.shape
    rmax = math.ceil(rayon / res + 0.5)
    for di in range(-rmax, rmax + 1):
        for dj in range(-rmax, rmax + 1):
            if di == 0 and dj == 0:
                continue
            # distance du centre de la case (i, j) au carre de la case obstacle (i - di, j - dj)
            ecart = res * math.hypot(max(abs(di) - 0.5, 0.0), max(abs(dj) - 0.5, 0.0))
            if ecart > rayon + 1e-9:
                continue
            out[max(di, 0) : nh + min(di, 0), max(dj, 0) : nw + min(dj, 0)] |= masque[
                max(-di, 0) : nh + min(-di, 0), max(-dj, 0) : nw + min(-dj, 0)
            ]
    return out


def reduire(grille, k):
    """Reduit une grille -1/0/1 d'un facteur k (k x k -> 1) de facon prudente :
    obstacle si au moins un obstacle, inconnu si tout est inconnu, sinon libre."""
    if k == 1:
        return grille.copy()
    nh, nw = grille.shape
    nhp, nwp = -(-nh // k) * k, -(-nw // k) * k
    pad = np.full((nhp, nwp), INCONNU, np.int8)
    pad[:nh, :nw] = grille
    return pad.reshape(nhp // k, k, nwp // k, k).max(axis=(1, 3))


class Grille:
    """Vue dense de la carte fusionnee au niveau le plus fin, sur une zone rectangulaire."""

    def __init__(self, val, src, zdrone, i0, j0, res):
        self.val = val  # -1 / 0 / 1
        self.src = src  # 0 aucune, 1 drone, 2 rover : qui a donne la valeur
        self.zdrone = zdrone  # niveau de la tuile drone utilisee (-1 si aucune)
        self.i0, self.j0, self.res = i0, j0, res

    def xy(self, i, j):
        """Centre de la case (i, j) en metres."""
        return (j + self.j0 + 0.5) * self.res, (i + self.i0 + 0.5) * self.res

    def etendue(self):
        """Pour imshow(..., origin='lower', extent=...)."""
        nh, nw = self.val.shape
        return (
            self.j0 * self.res,
            (self.j0 + nw) * self.res,
            self.i0 * self.res,
            (self.i0 + nh) * self.res,
        )


class GrillePlan:
    """Grille de planification : niveau `niveau_plan`, obstacles gonfles du rayon du robot."""

    def __init__(self, val, bloque, i0, j0, res):
        self.val, self.bloque = val, bloque
        self.i0, self.j0, self.res = i0, j0, res

    def ij(self, x, y):
        return int(math.floor(y / self.res)) - self.i0, int(math.floor(x / self.res)) - self.j0

    def xy(self, i, j):
        return (j + self.j0 + 0.5) * self.res, (i + self.i0 + 0.5) * self.res

    def dedans(self, i, j):
        return 0 <= i < self.val.shape[0] and 0 <= j < self.val.shape[1]


# ----------------------------------------------------------------------------
# Carte fusionnee
# ----------------------------------------------------------------------------
class CarteFusion:
    def __init__(
        self,
        n=32,
        res_fine=0.05,
        zmax=4,
        niveau_plan=3,
        niveau_zoom=2,
        rayon_robot=0.15,
        cout_inconnu=3.0,
        marge_securite=0.0,
    ):
        """
        n            : cases par cote de tuile
        res_fine     : taille d'une case au niveau zmax (m)
        niveau_plan  : niveau sur lequel tourne le A* (plus grossier = plus rapide)
        niveau_zoom  : niveau demande au drone quand il faut regarder de plus pres
        rayon_robot  : rayon du robot (m), cercle qui englobe toute sa carrosserie : les
                       obstacles sont gonfles de ce rayon, le robot devient un point pour le A*
        cout_inconnu : cout d'une case inconnue (1 = libre). Infini = interdit.
        marge_securite : marge en plus du rayon (m) pour ce que le A* ne sait pas : erreur de
                       suivi du chemin par le rover, bruit du capteur, imprecision de la carte
        """
        self.n, self.res_fine, self.zmax = n, res_fine, zmax
        self.niveau_plan, self.niveau_zoom = niveau_plan, niveau_zoom
        self.rayon_robot, self.cout_inconnu = rayon_robot, cout_inconnu
        self.marge_securite = marge_securite
        # source ('drone', 'rover', 'drone:<id>', 'rover:<id>') -> {(z, x, y): matrice}
        self.tuiles = {s: {} for s in SOURCES}
        self.maj = {s: {} for s in SOURCES}  # source -> {(z, x, y): t}
        self.demandes = set()  # tuiles deja demandees au drone
        self.inspectees = set()  # tuiles deja inspectees par le rover lui-meme (sans drone)
        # Marge en plus autour des obstacles DU DRONE (m). Sans drone, la position du rover
        # n'est plus recalee : on la met a l'incertitude de position, pour rester prudent
        # vis-a-vis de la carte du drone. Les obstacles que le rover voit lui-meme ne sont
        # pas concernes : il les voit par rapport a lui.
        self.marge_drone = 0.0
        # Zones interdites (x, y, rayon) : pas des obstacles physiques mais des endroits ou il
        # ne faut pas aller, par exemple la ou la liaison avec le rover a ete perdue. Le A* les
        # traite comme des obstacles ; elles restent a part pour pouvoir etre retirees (relais
        # radio) et pour que le drone ne vienne pas les « zoomer ».
        self.zones_interdites = []
        # Inspection by the rover itself (no drone): it stops at least inspect_reserve m
        # outside the safety margin of the obstacle, and at most inspect_max_distance m from
        # it. Field values; mission_settings.ARENA sets smaller ones for the 2.4 m arena.
        self.inspect_reserve = 0.35
        self.inspect_max_distance = 1.2
        # Duree de vie (s) des obstacles signales par signaler_obstacle (chocs) ; None = pour
        # toujours (comportement historique). Avec une duree, un faux choc ('stalled',
        # patinage) ne ferme un passage que ce temps-la : expirer_chocs(t) les efface.
        self.duree_choc = None
        self.chocs_temporaires = []  # (x, y, rayon, t_fin, source)

    # ---- geometrie --------------------------------------------------------
    @property
    def rayon_effectif(self):
        """Rayon dont les obstacles sont gonfles : le robot + sa marge de securite (m)."""
        return self.rayon_robot + self.marge_securite

    def res(self, z):
        return self.res_fine * 2 ** (self.zmax - z)

    def taille_tuile(self, z):
        return self.n * self.res(z)

    def cle(self, x, y, z):
        taille = self.taille_tuile(z)
        return (z, math.floor(x / taille), math.floor(y / taille))

    def rect_tuile(self, cle):
        z, tx, ty = cle
        taille = self.taille_tuile(z)
        return tx * taille, ty * taille, taille  # coin sud-ouest + cote

    # ---- reception --------------------------------------------------------
    def sources(self, role):
        """Les sources (appareils) d'un role connues de la carte."""
        return [s for s in self.tuiles if role_source(s) == role]

    def recevoir(self, msg):
        s, z = msg['source'], int(msg['z'])
        role_source(s)
        if not 0 <= z <= self.zmax:
            raise ValueError(f'niveau z={z} hors de [0, {self.zmax}]')
        data = np.asarray(msg['data'], dtype=np.int8)
        if data.shape != (self.n, self.n):
            raise ValueError(f'tuile {data.shape}, attendu ({self.n}, {self.n})')
        if not np.isin(data, (INCONNU, LIBRE, OBSTACLE)).all():
            raise ValueError('valeurs autorisees : -1, 0, 1')
        k = (z, int(msg['x']), int(msg['y']))
        t = float(msg.get('t', 0.0))
        store = self.tuiles.setdefault(s, {})
        self.maj.setdefault(s, {})
        if k in store and self.maj[s].get(k, -math.inf) > t:
            # message plus ancien que ce qu'on a : il ne complete que les trous
            store[k] = np.where(store[k] != INCONNU, store[k], data)
        elif k in store:
            store[k] = np.where(data != INCONNU, data, store[k])
            self.maj[s][k] = t
        else:
            store[k] = data.copy()
            self.maj[s][k] = t

    def instantane_rover(self, source='rover'):
        """Copie des tuiles du rover, pour pouvoir annuler plus tard des observations faites
        avec une position douteuse (pendant une coupure du drone)."""
        tuiles = self.tuiles.get(source, {})
        return {k: v.copy() for k, v in tuiles.items()}, dict(self.maj.get(source, {}))

    def restaurer_rover(self, instantane, source='rover'):
        """Remet les tuiles du rover dans l'etat d'un instantane_rover()."""
        tuiles, maj = instantane
        self.tuiles[source] = {k: v.copy() for k, v in tuiles.items()}
        self.maj[source] = dict(maj)

    def _disque(self, x, y, rayon):
        """Cases fines (niveau zmax) dont le centre est dans le disque : {cle: masque}."""
        r = self.res_fine
        masques = {}
        for i in range(math.floor((y - rayon) / r), math.floor((y + rayon) / r) + 1):
            for j in range(math.floor((x - rayon) / r), math.floor((x + rayon) / r) + 1):
                cx, cy = (j + 0.5) * r, (i + 0.5) * r
                if (cx - x) ** 2 + (cy - y) ** 2 > rayon**2:
                    continue
                k = self.cle(cx, cy, self.zmax)
                m = masques.setdefault(k, np.zeros((self.n, self.n), bool))
                m[i - k[2] * self.n, j - k[1] * self.n] = True
        return masques

    def signaler_obstacle(self, x, y, rayon=0.1, t=0.0, duree=None, source='rover'):
        """Obstacle 'ressenti' par le rover (patinage, blocage, pente trop forte).
        Envoye comme une tuile rover ordinaire, via recevoir().

        `duree` (s ; par defaut self.duree_choc, None = pour toujours) : l'obstacle est
        efface par expirer_chocs(t') des que t' >= t + duree."""
        for (z, tx, ty), masque in self._disque(x, y, rayon).items():
            data = np.where(masque, OBSTACLE, INCONNU).astype(np.int8)
            tk = max(t, self.maj.get(source, {}).get((z, tx, ty), t))  # toujours prioritaire
            self.recevoir({'source': source, 't': tk, 'z': z, 'x': tx, 'y': ty, 'data': data})
        duree = self.duree_choc if duree is None else duree
        if duree is not None:
            self.chocs_temporaires.append((x, y, rayon, t + duree, source))

    def _effacer(self, source, cle, masque):
        """Remet a inconnu les cases `masque` d'une tuile (supprimee si elle devient vide).
        Renvoie le nombre de cases connues effacees."""
        store = self.tuiles.get(source, {})
        if cle not in store:
            return 0
        tuile = store[cle]
        n = int(((tuile != INCONNU) & masque).sum())
        tuile[masque] = INCONNU
        if (tuile == INCONNU).all():
            del store[cle]
            self.maj[source].pop(cle, None)
        return n

    def expirer_chocs(self, t):
        """Efface les chocs dont la duree de vie est passee a l'instant t ; renvoie leur
        nombre. Leurs cases redeviennent inconnues pour le rover : la carte du drone reprend
        la main. Sans duree de vie (defaut), ne fait rien."""
        restants, effaces = [], 0
        for choc in self.chocs_temporaires:
            x, y, rayon, t_fin, source = choc
            if t < t_fin:
                restants.append(choc)
                continue
            for cle, masque in self._disque(x, y, rayon).items():
                self._effacer(source, cle, masque)
            effaces += 1
        self.chocs_temporaires = restants
        return effaces

    def oublier_rover(self, zone=None, source=None):
        """Oublie ce que les rovers ont vu ou ressenti (vues et chocs) dans `zone`
        (x0, y0, x1, y1 en m ; None = partout), pour toutes les sources du role rover ou
        seulement `source`. Pour reparer la carte apres un faux choc ou un passant vu par la
        camera du rover : la carte du drone reprend la main. Renvoie le nombre de cases
        oubliees."""
        sources = [source] if source is not None else self.sources('rover')
        oubliees = 0
        for s in sources:
            for cle in list(self.tuiles.get(s, {})):
                masque = np.ones((self.n, self.n), bool)
                if zone is not None:
                    x0, y0, cote = self.rect_tuile(cle)
                    centres = (np.arange(self.n) + 0.5) * (cote / self.n)
                    dans_x = (x0 + centres >= zone[0]) & (x0 + centres <= zone[2])
                    dans_y = (y0 + centres >= zone[1]) & (y0 + centres <= zone[3])
                    masque = dans_y[:, None] & dans_x[None, :]
                oubliees += self._effacer(s, cle, masque)

        def garde(choc):
            x, y, _, _, s = choc
            if s not in sources:
                return True
            return zone is not None and not (zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3])

        self.chocs_temporaires = [c for c in self.chocs_temporaires if garde(c)]
        return oubliees

    # ---- vue fusionnee ----------------------------------------------------
    def grille(self, x0, y0, x1, y1):
        """Carte fusionnee dense, au niveau le plus fin, sur [x0, x1] x [y0, y1]."""
        r = self.res_fine
        eps = 1e-6  # evite 0.6/0.1 = 6.0000001 -> 7
        i0, j0 = math.floor(y0 / r + eps), math.floor(x0 / r + eps)
        i1, j1 = math.ceil(y1 / r - eps), math.ceil(x1 / r - eps)
        nh, nw = i1 - i0, j1 - j0
        couches = {}
        for role in SOURCES:
            val = np.full((nh, nw), INCONNU, np.int8)
            zc = np.full((nh, nw), -1, np.int8)
            # Du plus grossier au plus fin, puis du plus ancien au plus recent : chaque tuile
            # ecrase ce qui precede la ou elle sait. Pour une seule source, les tuiles d'un
            # meme niveau ne se recouvrent pas : seul l'ordre des niveaux compte.
            tuiles = [
                (z, self.maj[s].get((z, tx, ty), 0.0), tx, ty, data)
                for s in self.sources(role)
                for (z, tx, ty), data in self.tuiles[s].items()
            ]
            tuiles.sort(key=lambda u: (u[0], u[1]))
            for z, _, tx, ty, data in tuiles:
                f = 2 ** (self.zmax - z)  # 1 case de la tuile = f x f cases fines
                cote = self.n * f
                ti0, tj0 = ty * cote, tx * cote
                a0, a1 = max(ti0, i0), min(ti0 + cote, i1)
                b0, b1 = max(tj0, j0), min(tj0 + cote, j1)
                if a0 >= a1 or b0 >= b1:
                    continue
                lignes = (np.arange(a0, a1) - ti0) // f
                cols = (np.arange(b0, b1) - tj0) // f
                patch = data[np.ix_(lignes, cols)]
                connu = patch != INCONNU
                v = val[a0 - i0 : a1 - i0, b0 - j0 : b1 - j0]
                v[connu] = patch[connu]
                zc[a0 - i0 : a1 - i0, b0 - j0 : b1 - j0][connu] = z
            couches[role] = (val, zc)
        vd, zd = couches['drone']
        vr, _ = couches['rover']
        rover_connu = vr != INCONNU
        val = np.where(rover_connu, vr, vd).astype(np.int8)
        src = np.where(rover_connu, SRC_ROVER, np.where(vd != INCONNU, SRC_DRONE, SRC_AUCUNE))
        return Grille(val, src.astype(np.int8), zd, i0, j0, r)

    def _reduire_aligne(self, g, arr):
        """Reduit une grille fine au niveau de planification.

        Les blocs sont alignes sur la grille globale.
        """
        k = 2 ** (self.zmax - self.niveau_plan)
        oi, oj = g.i0 % k, g.j0 % k
        nh, nw = arr.shape
        v = np.full((nh + oi, nw + oj), INCONNU, np.int8)
        v[oi:, oj:] = arr
        return reduire(v, k)

    def grille_plan(self, g):
        k = 2 ** (self.zmax - self.niveau_plan)
        val = self._reduire_aligne(g, g.val)
        res = g.res * k
        r = self.rayon_effectif  # m : apres gonflage, le robot est un point pour le A*
        if self.marge_drone > 0:
            # Marge elargie seulement pour les obstacles du drone situes loin de ce que le
            # rover a vu lui-meme : la ou il vient de regarder de pres, on se fie a ses yeux.
            obstacle = g.val == OBSTACLE
            drone = self._reduire_aligne(g, (obstacle & (g.src == SRC_DRONE)).astype(np.int8))
            rover = self._reduire_aligne(g, (obstacle & (g.src == SRC_ROVER)).astype(np.int8))
            vu = self._reduire_aligne(g, (g.src == SRC_ROVER).astype(np.int8)) == 1
            pres_du_vu = dilater(vu, math.ceil(0.5 / res))
            r_drone = r + self.marge_drone
            bloque = (
                gonfler(rover == 1, r, res)
                | gonfler((drone == 1) & pres_du_vu, r, res)
                | gonfler((drone == 1) & ~pres_du_vu, r_drone, res)
            )
        else:
            bloque = gonfler(val == OBSTACLE, r, res)
        if self.zones_interdites:
            nh, nw = val.shape
            xs = (np.arange(nw) + g.j0 // k + 0.5) * res
            ys = (np.arange(nh) + g.i0 // k + 0.5) * res
            for zx, zy, zr in self.zones_interdites:
                rr = zr + self.rayon_effectif
                bloque |= (xs[None, :] - zx) ** 2 + (ys[:, None] - zy) ** 2 <= rr * rr
        return GrillePlan(val, bloque, g.i0 // k, g.j0 // k, res)

    def interdire_zone(self, x, y, rayon):
        """Ajoute une zone ou le rover ne doit plus aller (cercle de `rayon` m)."""
        self.zones_interdites.append((float(x), float(y), float(rayon)))

    # ---- planification ----------------------------------------------------
    def _liberer_depart(self, plan, i, j):
        """Cases bloquees pour un robot parti de la case (i, j).

        Si le depart est libre, rien ne change : le gonflement garde le robot a son rayon de
        tout obstacle. S'il est dans une zone gonflee (choc, marge du drone qui a grandi,
        obstacle apparu), le robot doit pouvoir en sortir, mais sans s'approcher : pres du
        depart, seules sont liberees les cases au moins aussi loin de l'obstacle brut le plus
        proche que le depart lui-meme (sortie monotone). Les obstacles bruts restent
        bloques."""
        if not plan.dedans(i, j) or not plan.bloque[i, j]:
            return plan.bloque
        r = math.ceil((self.rayon_effectif + self.marge_drone) / plan.res)
        nh, nw = plan.val.shape
        a0, a1, b0, b1 = max(i - r, 0), min(i + r + 1, nh), max(j - r, 0), min(j + r + 1, nw)
        # une case de la fenetre n'est gonflee que par un obstacle a moins de r cases d'elle
        c0, c1 = max(i - 2 * r - 1, 0), min(i + 2 * r + 2, nh)
        d0, d1 = max(j - 2 * r - 1, 0), min(j + 2 * r + 2, nw)
        oi, oj = np.nonzero(plan.val[c0:c1, d0:d1] == OBSTACLE)
        oi, oj = oi + c0, oj + d0

        def distance(ii, jj):
            """Distance (m) du centre des cases au carre de l'obstacle brut le plus proche,
            comme dans gonfler()."""
            if oi.size == 0:
                return np.full(ii.shape, np.inf)
            di = np.maximum(np.abs(ii[..., None] - oi) - 0.5, 0.0)
            dj = np.maximum(np.abs(jj[..., None] - oj) - 0.5, 0.0)
            return plan.res * np.hypot(di, dj).min(axis=-1)

        ii, jj = np.mgrid[a0:a1, b0:b1]
        d_depart = distance(np.array([i]), np.array([j]))[0]
        sortie = (distance(ii, jj) >= d_depart - 1e-9) & (plan.val[a0:a1, b0:b1] != OBSTACLE)
        bloque = plan.bloque.copy()
        bloque[a0:a1, b0:b1] &= ~sortie
        return bloque

    @staticmethod
    def _composante(bloque, si, sj):
        """Cases que le robot peut atteindre depuis (si, sj) sans passer par une case bloquee
        (4-connexite ; l'A* 8-connexe sans diagonale entre deux cases bloquees atteint
        exactement les memes)."""
        if cv2 is not None:
            libre = (~bloque).astype(np.uint8)
            libre[si, sj] = 1
            _, etiquettes = cv2.connectedComponents(libre, connectivity=4)
            return etiquettes == etiquettes[si, sj]
        vu = np.zeros_like(bloque)
        pile = [(si, sj)]
        vu[si, sj] = True
        nh, nw = vu.shape
        while pile:
            i, j = pile.pop()
            for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= a < nh and 0 <= b < nw and not vu[a, b] and not bloque[a, b]:
                    vu[a, b] = True
                    pile.append((a, b))
        return vu

    def astar(self, plan, depart, objectif, tolerance, cout_inconnu=None):
        """A* 8-connexe. Arrivee = n'importe quelle case libre a moins de `tolerance` m.

        Couts : 1 par case droite, 1.4142 en diagonale, multiplies par `cout_inconnu` sur une
        case inconnue. Heuristique : distance a vol d'oiseau (admissible -> chemin optimal).
        Pour la vitesse, l'heuristique, les couts et les cases bloquees sont precalcules en
        numpy puis lus dans des listes Python indexees par p = i * largeur + j."""
        si, sj = plan.ij(*depart)
        if not plan.dedans(si, sj):
            return None
        nh, nw = plan.val.shape
        cout_inc = self.cout_inconnu if cout_inconnu is None else cout_inconnu
        res = plan.res
        xs = (np.arange(nw) + plan.j0 + 0.5) * res
        ys = (np.arange(nh) + plan.i0 + 0.5) * res
        dist = np.hypot(xs[None, :] - objectif[0], ys[:, None] - objectif[1])
        hgrille = np.maximum(0.0, dist / res - tolerance / res)
        bloque = self._liberer_depart(plan, si, sj)
        if cv2 is not None and not (self._composante(bloque, si, sj) & (hgrille == 0.0)).any():
            return None  # aucune arrivee atteignable : inutile d'explorer toute la carte
        h = hgrille.ravel().tolist()
        but = (hgrille == 0.0).ravel().tolist()
        cout = np.where(plan.val == INCONNU, cout_inc, 1.0).ravel().tolist()
        b = bloque.ravel().tolist()

        meilleur = [math.inf] * (nh * nw)
        parent = [-1] * (nh * nw)
        s = si * nw + sj
        meilleur[s] = 0.0
        tas = [(h[s], 0.0, s)]
        diag = 1.4142
        pop, push = heapq.heappop, heapq.heappush
        while tas:
            _, g, p = pop(tas)
            if g > meilleur[p]:
                continue
            if but[p]:
                chemin = [p]
                while parent[chemin[-1]] >= 0:
                    chemin.append(parent[chemin[-1]])
                return [plan.xy(q // nw, q % nw) for q in reversed(chemin)]
            i, j = divmod(p, nw)
            haut, bas, gauche, droite = i > 0, i < nh - 1, j > 0, j < nw - 1
            # meme ordre de voisins qu'avant : haut, bas, gauche, droite, puis diagonales
            for ok, q in ((haut, p - nw), (bas, p + nw), (gauche, p - 1), (droite, p + 1)):
                if ok and not b[q]:
                    ng = g + cout[q]
                    if ng < meilleur[q]:
                        meilleur[q] = ng
                        parent[q] = p
                        push(tas, (ng + h[q], ng, q))
            for ok, q, o1, o2 in (
                (haut and gauche, p - nw - 1, p - nw, p - 1),
                (haut and droite, p - nw + 1, p - nw, p + 1),
                (bas and gauche, p + nw - 1, p + nw, p - 1),
                (bas and droite, p + nw + 1, p + nw, p + 1),
            ):
                # pas de diagonale entre deux obstacles
                if ok and not b[q] and not b[o1] and not b[o2]:
                    ng = g + diag * cout[q]
                    if ng < meilleur[q]:
                        meilleur[q] = ng
                        parent[q] = p
                        push(tas, (ng + h[q], ng, q))
        return None

    def atteignable(self, plan, depart):
        si, sj = plan.ij(*depart)
        if not plan.dedans(si, sj):
            return np.zeros_like(plan.bloque)
        return self._composante(self._liberer_depart(plan, si, sj), si, sj)

    def premier_point_bloque(self, chemin, zone=None, horizon=None):
        """Premier point du chemin (ou de ses `horizon` premiers metres) qui tombe sur un
        obstacle connu, gonfle du rayon du robot. None si tout est libre.

        Seule une fenetre autour de ce morceau de chemin est calculee, au lieu de toute la
        carte : la marge couvre le gonflement des obstacles situes juste hors du chemin."""
        morceau, parcouru = [], 0.0
        for n, (x, y) in enumerate(chemin):
            if n:
                parcouru += math.hypot(x - chemin[n - 1][0], y - chemin[n - 1][1])
            if horizon is not None and parcouru > horizon:
                break
            morceau.append((x, y))
        if len(morceau) < 2:
            return None
        res_plan = self.res(self.niveau_plan)
        marge = self.rayon_effectif + self.marge_drone + 2 * res_plan
        xs, ys = [p[0] for p in morceau], [p[1] for p in morceau]
        fenetre = [min(xs) - marge, min(ys) - marge, max(xs) + marge, max(ys) + marge]
        if zone is not None:  # rien n'est bloque hors de la zone de travail
            fenetre = [
                max(fenetre[0], zone[0]),
                max(fenetre[1], zone[1]),
                min(fenetre[2], zone[2]),
                min(fenetre[3], zone[3]),
            ]
            if fenetre[0] >= fenetre[2] or fenetre[1] >= fenetre[3]:
                return None
        plan = self.grille_plan(self.grille(*fenetre))
        # Le rover peut se retrouver dans une zone gonflee (marge du drone qui grandit, choc) :
        # meme regle de sortie que le A* (_liberer_depart), sinon il replanifierait en boucle
        # sur place ; mais un chemin qui se rapproche d'un obstacle reste refuse.
        bloque = self._liberer_depart(plan, *plan.ij(*morceau[0]))
        for x, y in morceau[1:]:
            i, j = plan.ij(x, y)
            if plan.dedans(i, j) and bloque[i, j]:
                return x, y
        return None

    def chemin_valide(self, chemin, zone=None, horizon=None):
        """Le chemin (ou ses `horizon` premiers metres) evite-t-il tous les obstacles connus ?"""
        return self.premier_point_bloque(chemin, zone, horizon) is None

    # ---- decision ---------------------------------------------------------
    def _tuiles_inconnues_sur(self, plan, chemin):
        out = []
        for x, y in chemin:
            i, j = plan.ij(x, y)
            if plan.dedans(i, j) and plan.val[i, j] == INCONNU:
                k = self.cle(x, y, self.niveau_zoom)
                if k not in out and k not in self.demandes:
                    out.append(k)
        return out

    def _chemin_optimiste(self, g, depart, objectif, tolerance, niveau_max, exclure=()):
        """'Si les obstacles vus seulement de loin par le drone (niveau < niveau_max) etaient
        en fait franchissables, par ou passerais-je ?' On les traite comme inconnus mais chers
        et on replanifie. Renvoie (chemin, grille de plan, cases douteuses) ou (None, ...).
        `exclure` : secteurs (tuiles du niveau le plus fin) deja inspectes, dont les obstacles
        sont consideres comme confirmes."""
        douteux = (g.val == OBSTACLE) & (g.src == SRC_DRONE) & (g.zdrone < niveau_max)
        nh, nw = douteux.shape
        for _, tx, ty in exclure:
            i0, j0 = ty * self.n - g.i0, tx * self.n - g.j0
            a0, a1, b0, b1 = max(0, i0), min(nh, i0 + self.n), max(0, j0), min(nw, j0 + self.n)
            if a0 < a1 and b0 < b1:
                douteux[a0:a1, b0:b1] = False
        if not douteux.any():
            return None, None, None
        g2 = Grille(
            np.where(douteux, INCONNU, g.val).astype(np.int8), g.src, g.zdrone, g.i0, g.j0, g.res
        )
        plan2 = self.grille_plan(g2)
        douteux_plan = self._reduire_aligne(g, douteux.astype(np.int8)) == 1
        chemin = self.astar(
            plan2, depart, objectif, tolerance, cout_inconnu=max(10.0, self.cout_inconnu)
        )
        return chemin, plan2, douteux_plan

    def _candidats_zoom(self, g, depart, objectif, tolerance):
        """Tuiles a faire zoomer par le drone : celles des obstacles douteux que traverse le
        chemin optimiste (dans l'ordre du chemin). Renvoie aussi ce chemin."""
        chemin, plan2, douteux_plan = self._chemin_optimiste(
            g, depart, objectif, tolerance, self.niveau_zoom
        )
        if not chemin:
            return [], None
        out = []
        for x, y in chemin:
            i, j = plan2.ij(x, y)
            if plan2.dedans(i, j) and douteux_plan[i, j]:
                k = self.cle(x, y, self.niveau_zoom)
                if k not in out:
                    out.append(k)
        return out, chemin

    def _point_de_vue(self, plan, atteignable, cible, distance_max, distance_min=0.6):
        """Case atteignable d'ou observer `cible` : la plus proche a au moins distance_min m
        (pour ne pas finir dans la marge de securite de l'obstacle une fois vu de pres), sinon
        la plus proche tout court ; jamais a plus de distance_max m."""
        ii, jj = np.nonzero(atteignable)
        if ii.size == 0:
            return None
        xs = (jj + plan.j0 + 0.5) * plan.res
        ys = (ii + plan.i0 + 0.5) * plan.res
        d = np.hypot(xs - cible[0], ys - cible[1])
        if d.min() > distance_max:
            return None
        loin = np.where(d >= distance_min, d, np.inf)
        k = int(np.argmin(loin)) if np.isfinite(loin.min()) else int(np.argmin(d))
        if d[k] > distance_max:
            k = int(np.argmin(d))
        return float(xs[k]), float(ys[k])

    def _reconnaissance(self, g, plan, depart, objectif, tolerance):
        """Sans drone : au lieu de demander un zoom, le rover va voir lui-meme.

        Meme question que pour le drone (par ou passerais-je si les obstacles vus seulement
        de loin etaient franchissables ?), mais tous les obstacles du drone sont douteux pour
        le rover, qui voit plus fin. Le premier point douteux du chemin optimiste devient la
        cible ; le rover va a la case atteignable la plus proche (a moins de distance_max m)
        pour l'observer, avec une reserve (inspect_reserve, 35 cm par defaut) hors de la marge
        de securite (qui grandit avec l'incertitude). Un secteur (tuile du niveau le plus fin)
        n'est inspecte qu'une fois ; ensuite ses obstacles comptent comme confirmes et la
        recherche passe ailleurs."""
        optimiste, plan2, douteux_plan = self._chemin_optimiste(
            g, depart, objectif, tolerance, self.zmax + 1, exclure=self.inspectees
        )
        distance_min = self.rayon_effectif + self.marge_drone + self.inspect_reserve
        distance_max = max(self.inspect_max_distance, distance_min + self.inspect_max_distance / 2)
        if not optimiste:
            return None
        atteignable = None
        for x, y in optimiste:
            i, j = plan2.ij(x, y)
            if not (plan2.dedans(i, j) and douteux_plan[i, j]):
                continue
            secteur = self.cle(x, y, self.zmax)
            if secteur in self.inspectees:
                continue
            if atteignable is None:
                atteignable = self.atteignable(plan, depart)
            vue = self._point_de_vue(plan, atteignable, (x, y), distance_max, distance_min)
            chemin = self.astar(plan, depart, vue, tolerance=plan.res) if vue else None
            if not chemin:
                self.inspectees.add(secteur)  # impossible de s'en approcher : on passe
                continue
            return {
                'action': 'INSPECTER',
                'secteur': secteur,
                'cible': (x, y),
                'point_de_vue': vue,
                'chemin': chemin,
                'chemin_optimiste': optimiste,
                'grille': g,
            }
        return None

    def zone_survolee(self):
        """Rectangle (x0, y0, x1, y1) couvert par les tuiles des drones, ou None."""
        rects = [self.rect_tuile(k) for s in self.sources('drone') for k in self.tuiles[s]]
        if not rects:
            return None
        return (
            min(x for x, _, _ in rects),
            min(y for _, y, _ in rects),
            max(x + c for x, _, c in rects),
            max(y + c for _, y, c in rects),
        )

    def zone_par_defaut(self, depart, objectif, marge=5.0):
        """Zone de planification quand aucune n'est donnee : la zone survolee par le drone
        (le robot ne part pas explorer hors de la carte), agrandie si besoin pour contenir
        le depart et l'objectif. Sans tuile drone : depart et objectif +/- `marge` m."""
        xs, ys = (depart[0], objectif[0]), (depart[1], objectif[1])
        zone = self.zone_survolee()
        if zone is None:
            return (min(xs) - marge, min(ys) - marge, max(xs) + marge, max(ys) + marge)
        m = 1.0
        return (
            min(zone[0], min(xs) - m),
            min(zone[1], min(ys) - m),
            max(zone[2], max(xs) + m),
            max(zone[3], max(ys) + m),
        )

    def decider(
        self,
        depart,
        objectif,
        zone=None,
        tolerance=0.5,
        marge=5.0,
        max_tuiles=2,
        drone_disponible=True,
    ):
        """
        `zone` : rectangle (x0, y0, x1, y1) ou planifier. Par defaut, la zone survolee par le
        drone (voir zone_par_defaut).
        `drone_disponible` : False quand la liaison avec le drone est coupee. Le rover fait
        alors la reconnaissance lui-meme (INSPECTER) au lieu de demander un zoom.

        Retourne un dict :
          action = "SUIVRE"           -> chemin ; a_verifier = tuiles inconnues sur le chemin
                   "DEMANDER_DRONE"   -> tuiles (z, x, y) a survoler plus bas
                   "INSPECTER"        -> sans drone : cible a observer, point_de_vue, chemin
                   "OBJECTIF_DEGRADE" -> point accessible le plus proche + chemin
        """
        if zone is None:
            zone = self.zone_par_defaut(depart, objectif, marge)
        g = self.grille(*zone)
        plan = self.grille_plan(g)
        tolerance = self._tolerance_arrivee(plan, objectif, tolerance)

        chemin = self.astar(plan, depart, objectif, tolerance)
        if chemin:
            a_verifier = self._tuiles_inconnues_sur(plan, chemin) if drone_disponible else []
            return {
                'action': 'SUIVRE',
                'chemin': chemin,
                'a_verifier': a_verifier[:max_tuiles],
                'grille': g,
            }

        if drone_disponible:
            candidats, optimiste = self._candidats_zoom(g, depart, objectif, tolerance)
            nouvelles = [c for c in candidats if c not in self.demandes][:max_tuiles]
            if nouvelles:
                self.demandes.update(nouvelles)
                return {
                    'action': 'DEMANDER_DRONE',
                    'tuiles': nouvelles,
                    'grille': g,
                    'chemin_optimiste': optimiste,
                    'raison': 'aucun chemin ; passage possible sous un obstacle vu de loin',
                }
        else:
            inspection = self._reconnaissance(g, plan, depart, objectif, tolerance)
            if inspection:
                return inspection

        # plus rien a verifier : on va au plus pres de la victime
        reach = self.atteignable(plan, depart)
        ii, jj = np.nonzero(reach)
        if ii.size == 0:
            return {
                'action': 'OBJECTIF_DEGRADE',
                'point': depart,
                'chemin': [depart],
                'distance_restante': math.hypot(depart[0] - objectif[0], depart[1] - objectif[1]),
                'grille': g,
            }
        xs = (jj + plan.j0 + 0.5) * plan.res
        ys = (ii + plan.i0 + 0.5) * plan.res
        dist = np.hypot(xs - objectif[0], ys - objectif[1])
        best = int(np.argmin(dist))
        point = (float(xs[best]), float(ys[best]))
        chemin = self.astar(plan, depart, point, tolerance=plan.res)
        return {
            'action': 'OBJECTIF_DEGRADE',
            'point': point,
            'chemin': chemin or [depart],
            'distance_restante': float(dist[best]),
            'grille': g,
        }

    def _tolerance_arrivee(self, plan, objectif, tolerance):
        """Rayon d'arrivee autour de la victime. Si aucune case libre n'est a moins de
        `tolerance` (victime vue comme un obstacle par la camera du dessus, ou couchee contre
        un mur, avec une tolerance plus petite que le rayon du robot), on accepte les cases
        juste hors du gonflement : tolerance + rayon du robot + marge du drone + une case."""
        nh, nw = plan.val.shape
        xs = (np.arange(nw) + plan.j0 + 0.5) * plan.res
        ys = (np.arange(nh) + plan.i0 + 0.5) * plan.res
        proche = np.hypot(xs[None, :] - objectif[0], ys[:, None] - objectif[1]) <= tolerance
        if (proche & ~plan.bloque).any():
            return tolerance
        return tolerance + self.rayon_effectif + self.marge_drone + plan.res


# ----------------------------------------------------------------------------
# Aide pour les equipes drone / rover : decouper une grande matrice en tuiles
# ----------------------------------------------------------------------------
def decouper(matrice, x0, y0, z, carte, source, t=0.0):
    """Decoupe une matrice dense (-1/0/1) alignee sur la grille du niveau z.
    matrice[i][j] = case X = x0 + j*res(z), Y = y0 + i*res(z).
    Retourne la liste des messages-tuiles (les tuiles 100 % inconnues sont omises)."""
    m = np.asarray(matrice, dtype=np.int8)
    n, r = carte.n, carte.res(z)
    gi0, gj0 = round(y0 / r), round(x0 / r)
    nh, nw = m.shape
    msgs = []
    for ty in range(math.floor(gi0 / n), math.floor((gi0 + nh - 1) / n) + 1):
        for tx in range(math.floor(gj0 / n), math.floor((gj0 + nw - 1) / n) + 1):
            tuile = np.full((n, n), INCONNU, np.int8)
            a0, a1 = max(ty * n, gi0), min((ty + 1) * n, gi0 + nh)
            b0, b1 = max(tx * n, gj0), min((tx + 1) * n, gj0 + nw)
            tuile[a0 - ty * n : a1 - ty * n, b0 - tx * n : b1 - tx * n] = m[
                a0 - gi0 : a1 - gi0, b0 - gj0 : b1 - gj0
            ]
            if (tuile != INCONNU).any():
                msgs.append({'source': source, 't': t, 'z': z, 'x': tx, 'y': ty, 'data': tuile})
    return msgs
