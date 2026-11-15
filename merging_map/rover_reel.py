"""
rover_reel.py - Implementation de robot.Robot pour le vrai RaspRover, a travers le serveur de
RBDRB (rover/rover_server.py, branche rover-teleop). Ce serveur tourne sur l'ordinateur du
rover (RDK X5) et parle a la carte ESP32 ; c'est par lui que le cerveau (cerveau.mission)
conduit le robot.

!! PAS ENCORE TESTE SUR LE ROBOT : seulement contre un faux serveur (test_rover_reel.py) et
   contre le vrai serveur en simulation (rover_server.py --sim). !!

Partage du travail
------------------
- Ici, sur l'ordinateur : OU aller. Le cerveau decoupe le chemin en pas de quelques dizaines
  de cm ; chaque pas part au serveur par POST /path, dans le repere de son odometrie.
- Le serveur du rover : COMMENT y aller. Il suit le pas sur ses roues et son gyroscope, et
  l'abandonne si les roues ne tournent plus (statut "stalled").

Il faut le serveur de la branche rover-teleop : POST /path y est arrive avec le commit
5a7277d et n'est pas encore dans main (le serveur de main repond 404, et on le dit).

Deux reperes
------------
- La carte (camera du dessus) : x, y en metres.
- L'odometrie du rover (roues + gyroscope, lue dans GET /state) : precise sur quelques
  metres, mais d'origine et d'orientation quelconques, et elle derive doucement.
On garde la transformation carte <- odometrie (une rotation et une translation). Elle n'est
recalee que quand le rover est ARRETE, sur une image prise apres l'arret : une image en
retard sur un rover qui roule decalerait tout. Pendant un pas, la position vient de
l'odometrie, si bien qu'une camera lente ne cree plus de faux chocs.

Cap : si la camera donne le cap du rover, la rotation s'en deduit tout de suite. Sinon elle
se mesure sur un deplacement vu a la fois par la camera et par l'odometrie (au moins
base_cap m). Au premier pas, le rover avance donc d'abord tout droit de pas_cap m.

Choc
----
- Le serveur abandonne le pas quand les roues ne tournent plus : "stalled" -> choc.
- Des roues qui patinent contre un obstacle tournent : le serveur croit etre arrive. A
  l'arret, la camera montre que le rover a bien moins avance que ses roues -> choc aussi.

Liaison
-------
Seule une absence de reponse pendant delai_liaison s (2 s) veut dire « rover injoignable » :
une reponse perdue sur le Wi-Fi ne compte pas. Une reponse HTTP d'erreur (400, 404...) est une
erreur de programme : elle leve ServeurRoverError au lieu de declencher le secours. Quand
l'ordinateur se tait, le serveur finit le pas en cours puis s'arrete : il n'y a PAS de
programme de retour embarque, l'ordinateur attend que la liaison revienne.

Ce qui se branche de l'exterieur
--------------------------------
  - requete(methode, chemin, corps) -> dict, ou None sans reponse : client_http(url) ;
  - position_drone() -> (x, y, cap ou None) du rover vu d'en haut, ou None ;
  - rafraichir_drone() (facultatif) : prend une image tout de suite (camera du dessus :
    camera.refresh(force=True)). Sans lui, on attend `attente_image` s apres l'arret ;
  - percevoir(position, cap) -> (matrice -1/0/1, x0, y0) ou None : la camera du rover, dans
    le repere de la carte (perception a ecrire) ;
  - drone : objet avec disponible() -> bool et zoomer(tuile) -> liste de messages-tuiles.
"""

import json
import math
import time
import urllib.error
import urllib.request

from robot import RobotOdometrie


class ServeurRoverError(RuntimeError):
    """Le serveur du rover a repondu par une erreur, ou n'est pas pret : a corriger, ce n'est
    pas une coupure de liaison."""


def client_http(url, delai=1.0):
    """Fonction requete(methode, chemin, corps=None) -> dict pour le serveur du rover, par
    exemple url = 'http://192.168.1.42:8080'. Renvoie None si le serveur ne repond pas."""
    url = url.rstrip('/')

    def requete(methode, chemin, corps=None):
        donnees = None if corps is None else json.dumps(corps).encode()
        demande = urllib.request.Request(
            url + chemin,
            data=donnees,
            method=methode,
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(demande, timeout=delai) as reponse:
                return json.loads(reponse.read() or b'{}')
        except urllib.error.HTTPError as e:  # le serveur a repondu : ce n'est pas une coupure
            detail = e.read().decode(errors='replace')[:200]
            if e.code == 404 and chemin == '/path':
                raise ServeurRoverError(
                    'Le serveur du rover ne connait pas POST /path : il faut celui de la '
                    'branche rover-teleop (commit 5a7277d), pas encore dans main.'
                ) from e
            raise ServeurRoverError(f'{methode} {chemin} -> HTTP {e.code} {detail}') from e
        except (OSError, ValueError):  # delai depasse, connexion impossible, reponse coupee
            return None

    return requete


def angle_normalise(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def pose_odom(etat):
    """(x, y, cap en rad) de l'odometrie du rover, dans une reponse de GET /state."""
    o = etat['odom']
    return float(o['x']), float(o['y']), math.radians(float(o['heading_deg']))


class Repere:
    """Passage odometrie du rover -> carte : p_carte = R(angle) . p_odom + (tx, ty)."""

    def __init__(self):
        self.angle = None  # inconnu tant que le cap n'a pas ete mesure
        self.tx = self.ty = 0.0

    def vers_carte(self, x, y):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return c * x - s * y + self.tx, s * x + c * y + self.ty

    def vers_odom(self, x, y):
        c, s = math.cos(self.angle), math.sin(self.angle)
        dx, dy = x - self.tx, y - self.ty
        return c * dx + s * dy, -s * dx + c * dy

    def ancrer(self, carte, odom):
        """Translation telle que le point `odom` tombe sur `carte`."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        self.tx = carte[0] - (c * odom[0] - s * odom[1])
        self.ty = carte[1] - (s * odom[0] + c * odom[1])


class RaspRoverReel(RobotOdometrie):
    """Le vrai RaspRover, vu par le cerveau, a travers le serveur de RBDRB."""

    # Liaison perdue : le cerveau attend (attendre_retour_embarque) au lieu de piloter un
    # retour. Le serveur ne ramene pas le rover : il finit le pas en cours puis s'arrete.
    retour_embarque = True

    def __init__(
        self,
        carte,
        requete,
        position_drone,
        percevoir,
        drone,
        *,
        rafraichir_drone=None,
        vitesse=0.2,
        periode=0.2,
        attente_image=0.6,
        base_cap=0.1,
        pas_cap=0.15,
        patinage=0.3,
        delai_pret=15.0,
        delai_liaison=2.0,
        delai_retour=60.0,
        inclinaison_camera=-20,
        portee_camera=2.0,
        horloge=time.monotonic,
        dormir=time.sleep,
    ):
        self.requete = requete
        self.position_drone, self.percevoir, self.drone = position_drone, percevoir, drone
        self.rafraichir_drone = rafraichir_drone
        self.vitesse = vitesse  # m/s, vitesse de croisiere demandee au serveur
        self.periode = periode  # s, lecture de GET /state pendant un pas
        self.attente_image = attente_image  # s apres l'arret, sans rafraichir_drone
        self.base_cap = base_cap  # m, deplacement minimal pour mesurer le cap
        self.pas_cap = pas_cap  # m, ligne droite pour mesurer le cap au depart
        self.patinage = patinage  # deplacement vu / deplacement des roues sous ce rapport
        self.delai_pret, self.delai_retour = delai_pret, delai_retour  # s
        # s sans reponse avant de declarer la liaison perdue : une reponse perdue sur le Wi-Fi
        # n'est pas une coupure (le cerveau marquerait sinon une zone interdite)
        self.delai_liaison = delai_liaison
        self.inclinaison_camera, self.portee_camera = inclinaison_camera, portee_camera
        self.horloge, self.dormir = horloge, dormir
        self.joignable, self.pret = True, False
        super().__init__(carte)  # appelle placer((0, 0)), qui ne parle pas encore au rover

    # ---- liaisons et temps -------------------------------------------------
    def _demander(self, methode, chemin, corps=None):
        reponse = self.requete(methode, chemin, corps)
        self.joignable = reponse is not None
        return reponse

    def _insister(self, methode, chemin, corps=None):
        """Comme _demander, en reessayant pendant delai_liaison s. Renvoyer un pas deja recu
        est sans danger : le serveur remplace le pas en cours par le nouveau."""
        debut = self.temps()
        while True:
            reponse = self._demander(methode, chemin, corps)
            if reponse is not None or self.temps() - debut >= self.delai_liaison:
                return reponse
            self.dormir(self.periode)

    def _etat(self):
        return self._insister('GET', '/state')

    def arreter(self):
        self._demander('POST', '/stop')

    def temps(self):
        return self.horloge()

    def attendre(self, duree):
        self.arreter()
        self.dormir(duree)

    def drone_disponible(self):
        return bool(self.drone.disponible()) and self.position_drone() is not None

    def rover_joignable(self):
        return self.joignable

    # ---- position ----------------------------------------------------------
    @property
    def cap_fiable(self):
        return self.repere.angle is not None

    def placer(self, pos):
        super().placer(pos)
        self.repere = Repere()
        self.cap = None  # cap du rover dans le repere de la carte (rad)
        self._appui = None  # (point carte, point odometrie) qui sert a mesurer le cap
        self._vue = None  # (point carte, point odometrie) du dernier recalage a l'arret
        self._a_recaler = True  # au premier pas : image + odometrie, rover arrete

    def recaler(self):
        mesure = self._mesure_fraiche()
        etat = self._etat()
        if mesure is None or etat is None:
            return 0.0
        erreur = math.dist(self.est, mesure[:2])
        self._recaler_a_l_arret(mesure, etat)
        return erreur

    def _suivre(self, etat):
        """Pendant un pas : la position vient de l'odometrie, l'incertitude grandit."""
        if self.repere.angle is None:
            return
        xo, yo, ho = pose_odom(etat)
        self.cap = angle_normalise(ho + self.repere.angle)
        self._avancer_estimation(self.repere.vers_carte(xo, yo))

    def _mesure_fraiche(self):
        """Position vue par la camera sur une image prise maintenant (rover arrete), ou None.
        L'image est prise avant disponible(), pour ne pas la traiter deux fois."""
        if self.rafraichir_drone is not None:
            self.rafraichir_drone()
        else:
            self.dormir(self.attente_image)  # laisser passer une image prise apres l'arret
        if not self.drone.disponible():
            return None
        return self.position_drone()

    def _recaler_a_l_arret(self, mesure, etat, nouvel_appui=False):
        """Rover arrete, image prise apres l'arret : la camera donne la position, et le cap
        (directement, ou d'apres le deplacement depuis le point d'appui)."""
        x, y, cap = mesure
        xo, yo, ho = pose_odom(etat)
        if cap is not None:
            self.repere.angle = angle_normalise(cap - ho)
            nouvel_appui = True
        elif self._appui is not None:
            (xc0, yc0), (xo0, yo0) = self._appui
            vu, roues = math.hypot(x - xc0, y - yc0), math.hypot(xo - xo0, yo - yo0)
            if min(vu, roues) >= self.base_cap:
                angle = math.atan2(y - yc0, x - xc0) - math.atan2(yo - yo0, xo - xo0)
                if self.repere.angle is None:
                    self.repere.angle = angle_normalise(angle)
                else:  # moyenne glissante : le bruit de la camera pese moins
                    ecart = angle_normalise(angle - self.repere.angle)
                    self.repere.angle = angle_normalise(self.repere.angle + 0.5 * ecart)
                nouvel_appui = True
        if self._appui is None or nouvel_appui:
            self._appui = ((x, y), (xo, yo))
        if self.repere.angle is not None:
            self.repere.ancrer((x, y), (xo, yo))
            self.cap = angle_normalise(ho + self.repere.angle)
        self._recaler((x, y))
        self._vue = ((x, y), (xo, yo))

    # ---- action ------------------------------------------------------------
    def _attendre_le_rover(self):
        """Une fois, avant le premier pas : l'ESP32 est connecte et le gyroscope calibre (le
        README du rover demande de ne pas bouger avant). False si le rover ne repond pas."""
        debut = self.temps()
        while True:
            etat = self._etat()
            if etat is None:
                return False
            if etat.get('link', {}).get('connected') and etat['odom'].get('gyro_ready'):
                self.pret = True
                return True
            if self.temps() - debut > self.delai_pret:
                raise ServeurRoverError(
                    f'Rover pas pret apres {self.delai_pret:.0f} s : ESP32 non connecte ou '
                    'gyroscope pas calibre (voir GET /state, link et odom.gyro_ready).'
                )
            self.dormir(0.5)

    def _faire_un_pas(self, chemin, corps, longueur):
        """Envoie un pas au serveur et le suit jusqu'a l'arret. Renvoie (statut, etat) : le
        statut du serveur ('done', 'stalled', 'timeout', 'stopped'...), 'delai' s'il ne finit
        pas, ou (None, None) si la liaison est perdue."""
        avant = self._etat()
        if avant is None:
            return None, None
        dernier = (avant['ctrl'].get('history') or [None])[0]
        if self._insister('POST', chemin, corps) is None:
            return None, None
        # le serveur abandonne un pas apres max(20 s, 3 x longueur / vitesse + 10 s) : on lui
        # laisse 5 s de plus pour le faire lui-meme
        delai = max(20.0, 3.0 * longueur / self.vitesse + 10.0) + 5.0
        debut, vu_en_cours = self.temps(), False
        derniere_reponse = debut
        etat = avant
        while self.temps() - debut < delai:
            self.dormir(self.periode)
            etat = self._demander('GET', '/state')
            if etat is None:
                if self.temps() - derniere_reponse < self.delai_liaison:
                    continue  # une reponse perdue : le serveur continue le pas sans nous
                return None, None  # liaison perdue : le serveur finit le pas puis s'arrete
            derniere_reponse = self.temps()
            self._suivre(etat)
            ctrl = etat['ctrl']
            # en attente, en cours, ou a l'arret le temps de mesurer (settle, 0,4 s)
            en_cours = ctrl.get('mode') in ('move', 'turn', 'path', 'settle')
            if en_cours or ctrl.get('queued') or ctrl.get('current') is not None:
                vu_en_cours = True
                continue
            historique = ctrl.get('history') or []
            if historique and (vu_en_cours or historique[0] != dernier):
                return historique[0].get('status'), etat
        self.arreter()
        return 'delai', self._etat() or etat

    def _arret(self, statut, etat):
        """Pas fini, rover arrete : recalage sur une image prise maintenant. Renvoie True si
        les roues ont tourne sans que le rover avance (patinage contre un obstacle)."""
        avant = self._vue
        mesure = self._mesure_fraiche()
        if mesure is None:
            return False
        patine = False
        if avant is not None and statut == 'done':
            (xc0, yc0), (xo0, yo0) = avant
            xo, yo, _ = pose_odom(etat)
            roues = math.hypot(xo - xo0, yo - yo0)
            vu = math.hypot(mesure[0] - xc0, mesure[1] - yc0)
            patine = roues > 0.05 and vu < self.patinage * roues
        bloque = patine or statut == 'stalled'
        self._recaler_a_l_arret(mesure, etat, nouvel_appui=bloque)
        return patine

    def _devant(self, point, direction):
        return point[0] + 0.15 * math.cos(direction), point[1] + 0.15 * math.sin(direction)

    def _mesurer_le_cap(self):
        """Cap inconnu (rover sans marqueur) : le rover avance tout droit de pas_cap m, et la
        camera et l'odometrie voient le meme deplacement. S'il est bloque devant, il essaie en
        arriere. Renvoie le point d'impact s'il etait bloque en avancant, sinon None."""
        bloque = None
        for distance in (self.pas_cap, -self.pas_cap):
            statut, etat = self._faire_un_pas('/move', {'distance': distance}, abs(distance))
            if statut is None:
                return None
            patine = self._arret(statut, etat)
            if distance > 0 and (patine or statut == 'stalled'):
                bloque = (self.est, pose_odom(etat)[2])
            if self.repere.angle is not None:
                break
        else:
            raise ServeurRoverError(
                "Cap du rover impossible a mesurer : il n'a bouge ni en avant ni en arriere, "
                'ou la camera ne le voit pas.'
            )
        if bloque is None:
            return None
        point, cap_odom = bloque
        return self._devant(point, cap_odom + self.repere.angle)

    def deplacer(self, cible):
        """Envoie le pas jusqu'a `cible` au serveur (POST /path, repere de l'odometrie) et
        attend qu'il soit fini. Renvoie le point d'impact si le rover a touche un obstacle
        (roues bloquees, ou qui patinent d'apres la camera), sinon None."""
        if not self.pret and not self._attendre_le_rover():
            return None
        if self._a_recaler:
            etat, mesure = self._etat(), self._mesure_fraiche()
            if etat is None:
                return None
            if mesure is not None:
                self._recaler_a_l_arret(mesure, etat)
            self._a_recaler = False
        if self.repere.angle is None:
            if not self.drone_disponible():
                raise ServeurRoverError(
                    'Cap du rover inconnu et camera indisponible : impossible de savoir dans '
                    'quelle direction partir.'
                )
            impact = self._mesurer_le_cap()
            if impact is not None or not self.joignable:
                return impact
        but = self.repere.vers_odom(*cible)
        corps = {
            'points': [[round(but[0], 4), round(but[1], 4)]],
            'frame': 'odom',
            'speed': self.vitesse,
        }
        statut, etat = self._faire_un_pas('/path', corps, math.dist(self.est, cible))
        if statut is None:
            return None  # liaison perdue : le cerveau le voit avec rover_joignable()
        patine = self._arret(statut, etat)
        if not (patine or statut == 'stalled'):
            return None  # 'done', ou arrete sans choc ('timeout', 'stopped', 'delai'...)
        dx, dy = cible[0] - self.est[0], cible[1] - self.est[1]
        if math.hypot(dx, dy) > 0.02:
            return self._devant(self.est, math.atan2(dy, dx))
        return self._devant(self.est, self.cap) if self.cap is not None else self.est

    def attendre_retour_embarque(self, chemin):
        """Liaison perdue. Le serveur du rover finit le pas en cours puis s'arrete : il n'a
        pas (encore) de programme de retour. L'ordinateur attend que la liaison revienne
        (coupure Wi-Fi passagere), arrete le rover et recale sa position."""
        debut = self.temps()
        while self.temps() - debut < self.delai_retour:
            self.dormir(0.5)
            if self._demander('GET', '/state') is None:
                continue
            self.arreter()
            etat = self._etat()
            if etat is None:
                continue
            mesure = self._mesure_fraiche()
            if mesure is not None:
                self._recaler_a_l_arret(mesure, etat)
            else:
                self._suivre(etat)
            return True
        return False

    # ---- perception et drone -----------------------------------------------
    def observer(self, tour_complet=False):
        """Ce que voit la camera ; tour_complet : la tourelle fait le tour (8 vues). Pour le
        serveur, pan + = a droite : le cap de la camera est donc cap - pan."""
        angles = [-180, -135, -90, -45, 0, 45, 90, 135] if tour_complet else [0]
        for pan in angles:
            if tour_complet:
                self._demander('POST', '/gimbal', {'pan': pan, 'tilt': self.inclinaison_camera})
                self.dormir(0.4)
            cap = None if self.cap is None else angle_normalise(self.cap - math.radians(pan))
            vue = self.percevoir(self.est, cap)
            if vue is not None:
                self.publier_vue(*vue)
        if tour_complet:
            self._demander('POST', '/gimbal', {'pan': 0, 'tilt': self.inclinaison_camera})

    def demander_zoom(self, tuile):
        for msg in self.drone.zoomer(tuile) or []:
            self.carte.recevoir(msg)
