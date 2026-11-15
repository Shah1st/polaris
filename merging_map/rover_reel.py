"""
rover_reel.py - Implementation de robot.Robot pour le vrai RaspRover (Waveshare), pilote
depuis l'ordinateur : c'est ce qui permet au cerveau (cerveau.mission) de conduire le robot.

!! PAS ENCORE TESTE SUR LE ROBOT : seulement contre un faux rover (test_rover_reel.py). !!

Ce qui est fait ici :
  - les commandes de l'ESP32 (JSON, envoyees via l'application du Pi, route /send_command) :
    T:13 (vitesse X m/s, rotation Z rad/s), T:1 (arret), T:133 (tourelle), T:130 (etat) ;
  - le suivi d'un segment, avec des commandes a 5 Hz (le heartbeat de l'ESP32 arrete les
    moteurs apres 3 s sans commande) ;
  - l'odometrie (odl / odr, voie de 12,5 cm du RaspRover) quand le drone ne voit pas le rover ;
  - la detection de choc : roues commandees, mais le rover n'avance plus ;
  - le tour d'horizon avec la tourelle, et l'attente du retour autonome (liaison perdue).

Ce qui se branche de l'exterieur (a ecrire / brancher par l'equipe) :
  - position_drone() -> (x, y, cap) du rover vu par le drone, ou None. Le cap peut etre
    None (rover trouve par sa taille, sans marqueur) : on garde alors celui de l'odometrie et
    on le mesure d'apres le mouvement du rover entre deux images ;
  - lire_odometrie() -> (odl, odr) en METRES (le retour T:1001 donne des cm), ou None.
    L'application du Pi ne les expose pas en HTTP : petit script a ajouter sur le Pi. Sans
    odometrie, le rover estime son deplacement d'apres les commandes envoyees ;
  - percevoir(position, cap) -> (matrice -1/0/1, x0, y0) : ce que voit la camera, a la
    resolution la plus fine de la carte, dans le repere de la carte (perception a ecrire) ;
  - drone : objet avec disponible() -> bool et zoomer(tuile) -> liste de messages-tuiles ;
  - le programme de secours du Pi, qui ramene le rover seul quand l'ordinateur se tait.
"""

import json
import math
import time
import urllib.parse
import urllib.request

from robot import RobotOdometrie

VOIE_RASPROVER = 0.125  # distance entre les roues gauche et droite (m), firmware ugv_base_ros


def commande_texte(commande):
    """Ligne de commande de l'application du Pi pour un JSON de l'ESP32. Le JSON ne doit
    contenir aucun espace : l'application decoupe la commande aux espaces."""
    return 'base -c ' + json.dumps(commande, separators=(',', ':'))


def envoyeur_http(url='http://192.168.50.5:5000', delai=0.5):
    """Fonction envoyer(commande) -> bool qui passe par l'application web du Pi.
    Renvoie False si le rover n'a pas repondu (liaison perdue)."""

    def envoyer(commande):
        donnees = urllib.parse.urlencode({'command': commande_texte(commande)}).encode()
        try:
            with urllib.request.urlopen(url + '/send_command', data=donnees, timeout=delai) as r:
                return 200 <= r.status < 300
        except OSError:
            return False

    return envoyer


def angle_normalise(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Odometrie:
    """Transforme les distances cumulees des roues (odl, odr, en m) en increments
    (distance parcourue, rotation) pour un robot differentiel."""

    def __init__(self, voie=VOIE_RASPROVER):
        self.voie = voie
        self.precedent = None

    def increment(self, odl, odr):
        if self.precedent is None:
            self.precedent = (odl, odr)
            return 0.0, 0.0
        dl, dr = odl - self.precedent[0], odr - self.precedent[1]
        self.precedent = (odl, odr)
        return (dl + dr) / 2, (dr - dl) / self.voie


class RaspRoverReel(RobotOdometrie):
    """Le vrai RaspRover, vu par le cerveau."""

    retour_embarque = True  # liaison perdue : le programme de secours du Pi le ramene seul

    def __init__(
        self,
        carte,
        envoyer,
        lire_odometrie,
        position_drone,
        percevoir,
        drone,
        *,
        vitesse=0.2,
        vitesse_rotation=0.8,
        periode=0.2,
        tolerance=0.03,
        delai_segment=8.0,
        duree_patinage=0.6,
        delai_retour=60.0,
        base_cap=0.1,
        inclinaison_camera=-20,
        portee_camera=2.0,
        voie=VOIE_RASPROVER,
        horloge=time.monotonic,
        dormir=time.sleep,
    ):
        self.envoyer_brut, self.lire_odometrie = envoyer, lire_odometrie
        self.position_drone, self.percevoir, self.drone = position_drone, percevoir, drone
        self.vitesse, self.vitesse_rotation = vitesse, vitesse_rotation  # m/s, rad/s
        self.periode, self.tolerance = periode, tolerance  # s, m
        self.delai_segment, self.duree_patinage = delai_segment, duree_patinage  # s
        self.delai_retour = delai_retour  # s, attente du retour autonome
        # cap mesure d'apres le mouvement : distance en ligne droite entre deux mesures (m)
        self.base_cap = base_cap
        self.inclinaison_camera, self.portee_camera = inclinaison_camera, portee_camera
        self.horloge, self.dormir = horloge, dormir
        self.odometrie = Odometrie(voie)
        self.cap = math.pi / 2
        self.cap_fiable = False  # cap donne par le drone, ou mesure d'apres le mouvement
        self.joignable = True
        self._x_commande = self._z_commande = 0.0  # derniere commande T:13 (m/s, rad/s)
        self._ancre = None  # debut de la ligne droite en cours, pour mesurer le cap
        super().__init__(carte)

    # ---- liaisons et temps -------------------------------------------------
    def commander(self, commande):
        self.joignable = bool(self.envoyer_brut(commande))
        if commande['T'] == 13:
            self._x_commande, self._z_commande = commande['X'], commande['Z']
        elif commande['T'] == 1:
            self._x_commande = self._z_commande = 0.0
        return self.joignable

    def arreter(self):
        self.commander({'T': 1, 'L': 0, 'R': 0})

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
    def placer(self, pos):
        super().placer(pos)
        lecture = self.lire_odometrie()
        if lecture:
            self.odometrie.increment(*lecture)  # point de depart de l'odometrie
        mesure = self.position_drone() if self.drone.disponible() else None
        if mesure is not None and mesure[2] is not None:
            self.cap, self.cap_fiable = mesure[2], True

    def recaler(self):
        mesure = self.position_drone()
        if mesure is None:
            return 0.0
        x, y, cap = mesure
        erreur = math.dist(self.est, (x, y))
        self._recaler((x, y))
        if cap is not None:
            self.cap, self.cap_fiable = cap, True
        return erreur

    def _increment(self):
        """(distance, rotation) depuis la mesure precedente : odometrie des roues, ou a
        defaut la derniere commande envoyee (estime)."""
        lecture = self.lire_odometrie()
        if lecture:
            return self.odometrie.increment(*lecture)
        return self._x_commande * self.periode, self._z_commande * self.periode

    def _cap_par_le_mouvement(self, pos):
        """Sans cap venant du drone : en ligne droite, la direction entre deux positions vues
        par le drone, a au moins base_cap m d'ecart, donne le cap."""
        en_ligne_droite = self._x_commande > 0.02 and abs(self._z_commande) < 0.15
        if not en_ligne_droite:
            self._ancre = None
        elif self._ancre is None:
            self._ancre = pos
        elif math.dist(self._ancre, pos) >= self.base_cap:
            self.cap = math.atan2(pos[1] - self._ancre[1], pos[0] - self._ancre[0])
            self.cap_fiable = True
            self._ancre = pos

    def _mesurer(self):
        """Nouvelle position estimee : celle du drone s'il voit le rover, sinon l'odometrie
        (et l'incertitude grandit). Le cap vient du drone s'il le connait, sinon de
        l'odometrie, corrige d'apres le mouvement."""
        distance, rotation = self._increment()
        mesure = self.position_drone() if self.drone.disponible() else None
        if mesure is not None:
            x, y, cap = mesure
            if cap is not None:
                self.cap, self.cap_fiable = cap, True
            else:
                self.cap += rotation
                self._cap_par_le_mouvement((x, y))
            self._recaler((x, y))
        else:
            cap_milieu = self.cap + rotation / 2
            self.cap += rotation
            self._avancer_estimation(
                (
                    self.est[0] + distance * math.cos(cap_milieu),
                    self.est[1] + distance * math.sin(cap_milieu),
                )
            )

    # ---- action ------------------------------------------------------------
    def _mesurer_le_cap(self):
        """Cap inconnu (rover sans marqueur) : une courte ligne droite, vue par le drone,
        suffit a le mesurer. Renvoie le point d'impact si le rover est bloque."""
        debut, depart = self.temps(), self.est
        self._ancre = None
        while not self.cap_fiable and self.temps() - debut < 3.0:
            if not self.commander({'T': 13, 'X': self.vitesse, 'Z': 0}):
                return None
            self.dormir(self.periode)
            self._mesurer()
            if self.temps() - debut > 1.0 and math.dist(depart, self.est) < 0.02:
                self.arreter()
                return self.est[0], self.est[1]  # il ne bouge pas : quelque chose devant
        self.arreter()
        return None

    def deplacer(self, cible):
        """Suit le segment jusqu'a `cible` : tourne sur place si l'ecart de cap est grand,
        sinon avance en corrigeant le cap. Renvoie le point d'impact si le rover est bloque
        (roues commandees mais position qui n'avance plus), sinon None."""
        if not self.cap_fiable and self.drone_disponible():
            impact = self._mesurer_le_cap()
            if impact is not None:
                return impact
        debut, immobile = self.temps(), 0.0
        impact = None
        while self.temps() - debut < self.delai_segment:
            dx, dy = cible[0] - self.est[0], cible[1] - self.est[1]
            distance = math.hypot(dx, dy)
            if distance <= self.tolerance:
                break
            ecart = angle_normalise(math.atan2(dy, dx) - self.cap)
            z = max(-self.vitesse_rotation, min(self.vitesse_rotation, 2.0 * ecart))
            x = self.vitesse * math.cos(ecart) if abs(ecart) < 0.6 else 0.0
            x = min(x, distance / self.periode)  # ne pas depasser la cible
            if not self.commander({'T': 13, 'X': round(x, 3), 'Z': round(z, 3)}):
                break  # liaison perdue : le cerveau le voit avec rover_joignable()
            avant = self.est
            self.dormir(self.periode)
            self._mesurer()
            attendu = x * self.periode
            if attendu > 0.005 and math.dist(avant, self.est) < 0.25 * attendu:
                immobile += self.periode
            else:
                immobile = 0.0
            if immobile >= self.duree_patinage:
                impact = (
                    self.est[0] + 0.15 * math.cos(self.cap),
                    self.est[1] + 0.15 * math.sin(self.cap),
                )
                break
        if self.joignable:
            self.arreter()
        return impact

    def attendre_retour_embarque(self, chemin):
        """Liaison perdue : le programme de secours du Pi ramene le rover sur sa trace.
        L'ordinateur ne peut que l'attendre, en suivant sa position grace au drone."""
        debut = self.temps()
        while self.temps() - debut < self.delai_retour:
            self.dormir(0.5)
            self._mesurer()
            if self.commander({'T': 130}):  # demande d'etat : sert de « ping »
                return True
        return False

    # ---- perception et drone -----------------------------------------------
    def observer(self, tour_complet=False):
        """Ce que voit la camera ; tour_complet : la tourelle fait le tour (8 vues)."""
        angles = [-180, -135, -90, -45, 0, 45, 90, 135] if tour_complet else [0]
        for pan in angles:
            if tour_complet:
                self.commander(
                    {'T': 133, 'X': pan, 'Y': self.inclinaison_camera, 'SPD': 0, 'ACC': 0}
                )
                self.dormir(0.4)
            vue = self.percevoir(self.est, self.cap + math.radians(pan))
            if vue is not None:
                self.publier_vue(*vue)
        if tour_complet:
            self.commander({'T': 133, 'X': 0, 'Y': self.inclinaison_camera, 'SPD': 0, 'ACC': 0})

    def demander_zoom(self, tuile):
        for msg in self.drone.zoomer(tuile) or []:
            self.carte.recevoir(msg)
