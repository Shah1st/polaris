"""
robot.py - Ce dont le cerveau (cerveau.mission) a besoin pour piloter un rover.

Deux implementations :
  - demo_fusion.Simu        : le rover et le drone simules (tests, tableau de bord) ;
  - rover_reel.RaspRoverReel : le vrai RaspRover, pilote depuis l'ordinateur.

Toutes les positions sont en metres dans le repere de la carte (celui du drone / ArUco).
"""

import abc
import math

from fusion_carte import decouper

INCERTITUDE_PAR_METRE = 0.025  # sans drone, l'erreur de position possible grandit de 2,5 cm/m


class Robot(abc.ABC):
    """Interface entre le cerveau et un rover (+ son drone)."""

    #: portee utile de la camera du rover (m)
    portee_camera = 2.0
    #: True si, quand la liaison avec l'ordinateur est coupee, le rover revient TOUT SEUL
    #: (programme de secours embarque). Le cerveau attend alors son retour au lieu de le piloter.
    retour_embarque = False

    # ---- temps et liaisons -------------------------------------------------
    @abc.abstractmethod
    def temps(self):
        """Temps courant (s)."""

    @abc.abstractmethod
    def attendre(self, duree):
        """Rester immobile `duree` s."""

    @abc.abstractmethod
    def drone_disponible(self):
        """La liaison ordinateur <-> drone fonctionne-t-elle ?"""

    @abc.abstractmethod
    def rover_joignable(self):
        """La liaison ordinateur <-> rover fonctionne-t-elle ?"""

    # ---- position ----------------------------------------------------------
    @abc.abstractmethod
    def position(self):
        """Position estimee (x, y) : celle du drone s'il voit le rover, sinon l'odometrie."""

    @abc.abstractmethod
    def incertitude(self):
        """Erreur de position possible (m) : 0 quand le drone voit le rover."""

    def position_reelle(self):
        """Position reelle, connue seulement en simulation (None sur le vrai robot)."""
        return None

    @abc.abstractmethod
    def placer(self, pos):
        """Debut de mission : le rover est en `pos`, position connue."""

    @abc.abstractmethod
    def recaler(self):
        """Le drone revoit le rover : la position estimee devient celle du drone. Renvoie
        l'erreur de position qui s'etait accumulee (m)."""

    @abc.abstractmethod
    def trace_depuis_recalage(self):
        """(dernier point vu par le drone, positions estimees parcourues depuis)."""

    # ---- action et perception ----------------------------------------------
    @abc.abstractmethod
    def deplacer(self, cible):
        """Aller en ligne droite jusqu'a `cible` (quelques dizaines de cm). Renvoie le point
        d'impact (repere de la carte) si le rover a touche un obstacle, sinon None."""

    @abc.abstractmethod
    def observer(self, tour_complet=False):
        """Envoyer a la carte ce que voit la camera (tour_complet : tourelle a 360 degres)."""

    @abc.abstractmethod
    def demander_zoom(self, tuile):
        """Demander au drone de survoler la tuile (z, x, y) plus bas."""

    @abc.abstractmethod
    def signaler_choc(self, impact):
        """Marquer sur la carte l'obstacle touche en `impact`."""

    @abc.abstractmethod
    def garder_a_bord(self):
        """Liaison rover coupee : ce que voit le rover est garde a bord."""

    @abc.abstractmethod
    def vider_tampon(self):
        """Liaison rover retrouvee : envoyer ce qui a ete garde a bord."""

    def attendre_retour_embarque(self, chemin):
        """Si retour_embarque : attendre que le rover soit revenu seul et joignable."""
        raise NotImplementedError


class RobotOdometrie(Robot):
    """Base commune : position estimee, incertitude, trace depuis le dernier recalage par le
    drone, et couche provisoire (observations faites sans drone, reecrites au recalage)."""

    def __init__(self, carte):
        self.carte = carte
        self.tampon = None  # liaison rover coupee : les tuiles vues sont gardees a bord
        self.placer((0.0, 0.0))

    # ---- position ----------------------------------------------------------
    def position(self):
        return self.est

    def incertitude(self):
        return self.sigma

    def placer(self, pos):
        self.est = pos
        self.sigma = 0.0
        self.dernier_fix, self.trace_fix = pos, [pos]
        self.dist_fix = 0.0  # distance parcourue depuis le dernier recalage par le drone
        self.instantane, self.provisoires = None, []

    def trace_depuis_recalage(self):
        return self.dernier_fix, list(self.trace_fix)

    def _avancer_estimation(self, est):
        """Nouvelle position estimee, sans drone : l'incertitude grandit avec la distance."""
        longueur = math.dist(self.est, est)
        self.est = est
        self.sigma += INCERTITUDE_PAR_METRE * longueur
        self.dist_fix += longueur
        self.trace_fix.append(est)

    def _recaler(self, mesure):
        """Position mesuree par le drone : on reprojette la couche provisoire, puis on repart
        de cette position avec une incertitude nulle."""
        self._reprojeter(mesure)
        self.est = mesure
        self.sigma, self.dist_fix = 0.0, 0.0
        self.dernier_fix, self.trace_fix = mesure, [mesure]

    def _reprojeter(self, mesure):
        """L'erreur finale E a grandi a peu pres avec la distance parcourue : une observation
        faite apres une distance d est decalee de E * d / D. On annule la couche provisoire
        puis on la reecrit corrigee, ce qui supprime les obstacles fantomes."""
        if self.instantane is None:
            return
        ex, ey = mesure[0] - self.est[0], mesure[1] - self.est[1]
        total = max(self.dist_fix, 1e-6)
        self.carte.restaurer_rover(self.instantane)
        t = self.temps()
        for genre, contenu, d in self.provisoires:
            f = min(1.0, d / total)
            if genre == 'vue':
                loc, x0, y0 = contenu
                for msg in decouper(
                    loc, x0 + ex * f, y0 + ey * f, self.carte.zmax, self.carte, 'rover', t
                ):
                    self.carte.recevoir(msg)
            else:
                self.carte.signaler_obstacle(contenu[0] + ex * f, contenu[1] + ey * f, 0.1, t)
        self.instantane, self.provisoires = None, []

    def provisoire(self, genre, contenu):
        """Note une observation faite sans drone (genre 'vue' ou 'choc')."""
        self.provisoires.append((genre, contenu, self.dist_fix))

    # ---- perception --------------------------------------------------------
    def publier_vue(self, loc, x0, y0):
        """Envoie a la carte une vue de la camera : matrice -1/0/1 a la resolution la plus
        fine, dont la case [0][0] a son coin en (x0, y0) dans le repere de la carte. Sans
        drone, la vue va aussi dans la couche provisoire ; sans liaison rover, elle est
        gardee a bord."""
        sans_drone = not self.drone_disponible()
        if sans_drone and self.instantane is None:
            self.instantane = self.carte.instantane_rover()
        for msg in decouper(loc, x0, y0, self.carte.zmax, self.carte, 'rover', self.temps()):
            if self.tampon is not None:
                self.tampon.append(msg)
            else:
                self.carte.recevoir(msg)
        if sans_drone:
            self.provisoire('vue', (loc, x0, y0))
        return loc, x0, y0

    def signaler_choc(self, impact):
        self.carte.signaler_obstacle(*impact, rayon=0.1, t=self.temps())
        if not self.drone_disponible():
            self.provisoire('choc', impact)

    def garder_a_bord(self):
        self.tampon = []

    def vider_tampon(self):
        for msg in self.tampon or []:
            self.carte.recevoir(msg)
        self.tampon = None
