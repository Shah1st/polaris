"""
rover_reel.py - Implementation de robot.Robot pour le vrai RaspRover, a travers le serveur de
RBDRB (rover/rover_server.py, dans main). Ce serveur tourne sur l'ordinateur du
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

Il faut un serveur qui connait POST /path (rover/rover_server.py de main, depuis le commit
5a7277d). Un serveur plus ancien repond 404, et on le dit. La version qui tourne sur le robot
n'a pas ete comparee au depot.

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
base_cap m). Au premier pas, le rover avance donc d'abord tout droit, de pas_cap m au plus.
Comme on ne sait pas encore vers ou il part, ce pas est limite a ce qui est libre tout autour
de lui sur la carte.

Choc
----
- Le serveur abandonne le pas quand les roues ne tournent plus : "stalled" -> choc.
- Des roues qui patinent contre un obstacle tournent : le serveur croit etre arrive. A
  l'arret, la camera montre que le rover a bien moins avance que ses roues -> choc aussi.
- Un pas qui n'a jamais fini ('timeout') ou que le garde-fou d'inclinaison du serveur a
  arrete ('tilt: ...', puis recul de 15 cm) : choc aussi, sinon le pas suivant y retourne.

Liaison
-------
Seule une absence de reponse pendant delai_liaison s (2 s) veut dire « rover injoignable » :
une reponse perdue sur le Wi-Fi ne compte pas. Une reponse HTTP d'erreur (400, 404...) est une
erreur de programme : elle leve ServeurRoverError au lieu de declencher le secours. Quand
l'ordinateur se tait, le serveur finit le pas en cours puis s'arrete : il n'y a PAS de
programme de retour embarque, l'ordinateur attend que la liaison revienne.
Liaison lente (tunnel SSH vers le robot, 1,5 a 4 s par requete, 15 nov.) : ce qui compte est
le nombre de requetes. Un pas en coute GET /state (controles), POST, puis les lectures de
l'etat jusqu'a l'arret ; avec attente_pas, la premiere lecture attend la duree attendue du
pas. lancer_overhead.py --liaison-lente regle delai_liaison, periode et le delai HTTP.

Securite
--------
- Arret operateur : un pas qui finit 'stopped', 'cancelled by manual control' ou
  'replaced by a new command' sans que l'adaptateur l'ait cause (STOP de la page du rover ou
  du tableau de bord, conduite au clavier, commande d'un autre client) leve ArretOperateur.
  C'est definitif : le cerveau ne renvoie pas de pas, lancer_overhead envoie /stop et finit.
- Redemarrage du serveur (ou /reset_odom) : detecte a chaque lecture de GET /state (boot_id
  qui change, ou odom.odometer ou link.rx_count qui diminue). L'odometrie est
  alors repartie de zero : avant le pas suivant on attend que l'ESP32 et le gyroscope soient
  prets, puis on recale le repere carte <- odometrie sur une image neuve (le cap aussi est
  oublie si la camera ne le donne pas : il est remesure).
- Panne du serveur : un pas qui finit 'no ESP32 feedback', 'ESP32 link lost' ou
  'controller error: ...' (le serveur a arrete le rover de lui-meme, voir ctrl.fault) n'est pas
  un pas fini : ServeurRoverError, et le pas suivant se recale d'abord sur une image neuve.
- STOP entre deux pas : rover_core ne laisse alors aucune trace dans ctrl.history. Un serveur
  recent compte les /stop recus (stop_count dans GET /state) : avant chaque pas, s'il en a recu
  plus que l'adaptateur n'en a envoye (avec une reponse), ArretOperateur. Sans ce champ
  (serveur plus ancien), ce STOP n'est pas vu.
- Borne : un POST /path dont la cible est plus loin de la pose odometrique actuelle que le pas
  demande + marge_pas n'est jamais envoye (ServeurRoverError).
- Le pas de mesure du cap est un /path vers un point absolu de l'odometrie : le renvoyer apres
  une reponse perdue ne le fait pas deux fois (un /move relatif le refaisait). En arriere
  (avant bloque), c'est un /move : un /path vers un point derriere le rover a moins de
  path_lookahead (0,2 m) finit 'done' sans bouger. Il n'est renvoye apres une reponse perdue
  que si GET /state n'en montre aucune trace.
- Un redemarrage vu pendant un pas (/reset_odom : meme processus, le controleur suit encore
  le pas vers une cible de l'ancien repere) est suivi d'un /stop.
- Conduite au clavier entre deux pas : vue a ctrl.mode 'manual', ou, une fois finie, a
  odom.odometer qui a grandi depuis la fin du dernier pas.

Ce qui se branche de l'exterieur
--------------------------------
  - requete(methode, chemin, corps) -> dict, ou None sans reponse : client_http(url) ;
  - position_drone() -> (x, y, cap ou None) du rover vu d'en haut, ou None ;
  - rafraichir_drone() (facultatif) : prend une image tout de suite (camera du dessus :
    camera.refresh(force=True)). Sans lui, on attend `attente_image` s apres l'arret ;
  - percevoir(position, cap) -> (matrice -1/0/1, x0, y0) ou None : la camera du rover, dans
    le repere de la carte (perception a ecrire) ;
  - drone : objet avec disponible() -> bool et zoomer(tuile) -> liste de messages-tuiles ;
  - signaler(texte) (facultatif) : journal des evenements de securite (redemarrage...).
"""

import http.client
import json
import math
import time
import urllib.error
import urllib.request

from fusion_carte import OBSTACLE
from robot import RobotOdometrie


class ServeurRoverError(RuntimeError):
    """Le serveur du rover a repondu par une erreur, ou n'est pas pret : a corriger, ce n'est
    pas une coupure de liaison."""


class ArretOperateur(Exception):  # noqa: N818 - nom fixe par l'interface entre les parties
    """Un operateur a arrete le rover ou l'a repris en main : la mission s'arrete la."""


# Statuts de ctrl.history qui, si l'adaptateur ne les a pas causes, viennent d'un operateur
ARRETS_OPERATEUR = ('stopped', 'cancelled by manual control', 'replaced by a new command')

# Statuts de ctrl.history d'un pas que le serveur a abandonne de lui-meme (Controller.fail de
# rover/rover_core.py) : plus de retour de l'ESP32, liaison serie perdue, erreur dans sa boucle
PANNES_SERVEUR = ('no ESP32 feedback', 'ESP32 link lost', 'controller error')


def panne_serveur(statut):
    """True si `statut` (de ctrl.history ou ctrl.fault) dit que le serveur a arrete le rover a
    cause d'une panne : le pas n'est pas fini."""
    return isinstance(statut, str) and statut.startswith(PANNES_SERVEUR)


# m : le serveur tient la fin d'un /path pour atteinte a cette distance pres (path_goal_tol de
# rover/rover_core.py) ; le rover s'arrete donc jusqu'a autant avant le point vise.
TOLERANCE_BUT = 0.02


def serveur_redemarre(avant, apres):
    """True si deux reponses de GET /state viennent de deux vies differentes du serveur (ou si
    l'odometrie a ete remise a zero entre les deux)."""
    if avant.get('boot_id') != apres.get('boot_id'):
        return True
    # Meme boot_id (ou aucun, serveur plus ancien) : l'odometrie a pu repartir de zero sans
    # nouveau processus (Rover recree, /reset_odom). Ces compteurs ne font que croitre sinon.
    for bloc, cle in (('odom', 'odometer'), ('link', 'rx_count')):
        v0, v1 = (avant.get(bloc) or {}).get(cle), (apres.get(bloc) or {}).get(cle)
        if v0 is not None and v1 is not None and float(v1) < float(v0) - 1e-6:
            return True
    return False


def garde_recule(ctrl):
    """True si le garde-fou d'inclinaison du serveur fait reculer le rover ou le tient arrete
    (ctrl.guard.state 'backoff' ou 'hold', champ facultatif) : ctrl.mode est alors 'idle' mais
    le rover bouge encore."""
    return ((ctrl.get('guard') or {}).get('state')) in ('backoff', 'hold')


# m : apres avoir arrete le pas ('tilt: ...'), le garde-fou d'inclinaison recule le rover de
# tilt_backoff_dist (rover/rover_core.py) : la bosse est d'autant plus loin qu'un choc
RECUL_GARDE = 0.15


def garde_declenche(statut):
    """True si `statut` (de ctrl.history) dit que le garde-fou d'inclinaison a arrete le pas
    ('tilt: ...') : le sol devant fait basculer le rover."""
    return isinstance(statut, str) and statut.startswith('tilt')


def statut_de_choc(statut):
    """True si le pas a fini contre quelque chose : roues bloquees ('stalled'), pas jamais
    fini ('timeout' : roues qui patinent contre un obstacle mou), ou garde-fou d'inclinaison.
    Sans choc marque, le pas suivant repartirait au meme endroit."""
    return statut in ('stalled', 'timeout') or garde_declenche(statut)


def odometre(etat):
    """odom.odometer (m parcourus par les roues), ou None si le champ manque."""
    v = (etat.get('odom') or {}).get('odometer')
    return None if v is None else float(v)


def nouvelles_entrees(historique, dernier):
    """Entrees de ctrl.history (la plus recente d'abord) arrivees apres `dernier`."""
    if dernier is None:
        return list(historique)
    for k, entree in enumerate(historique):
        if entree == dernier:
            return list(historique[:k])
    return list(historique)


def client_http(url, delai=1.0):
    """Fonction requete(methode, chemin, corps=None) -> dict pour le serveur du rover, par
    exemple url = 'http://192.168.1.42:8080'. Renvoie None si le serveur ne repond pas dans
    `delai` s (ou si la reponse arrive coupee)."""
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
                    'Le serveur du rover ne connait pas POST /path : rover_server.py trop ancien '
                    'sur le robot, deployer celui de main (commit 5a7277d ou plus recent).'
                ) from e
            raise ServeurRoverError(f'{methode} {chemin} -> HTTP {e.code} {detail}') from e
        # delai depasse, connexion impossible, reponse coupee (IncompleteRead n'est pas un
        # OSError) : une reponse perdue, pas une erreur de programme
        except (OSError, ValueError, http.client.HTTPException):
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
        attente_pas=False,
        attente_image=0.6,
        base_cap=0.1,
        pas_cap=0.15,
        patinage=0.3,
        delai_pret=15.0,
        delai_liaison=2.0,
        delai_retour=60.0,
        inclinaison_camera=-20,
        portee_camera=2.0,
        latence_video=0.7,
        marge_pas=0.2,
        signaler=None,
        horloge=time.monotonic,
        dormir=time.sleep,
    ):
        self.requete = requete
        self.position_drone, self.percevoir, self.drone = position_drone, percevoir, drone
        self.rafraichir_drone = rafraichir_drone
        self.vitesse = vitesse  # m/s, vitesse de croisiere demandee au serveur
        self.periode = periode  # s, lecture de GET /state pendant un pas
        # liaison lente : apres le POST d'un pas, pas de GET /state avant sa duree attendue
        # (longueur / vitesse, moins une periode) ; chaque requete y coute 1,5 a 4 s
        self.attente_pas = attente_pas
        self.attente_image = attente_image  # s apres l'arret, sans rafraichir_drone
        self.base_cap = base_cap  # m, deplacement minimal pour mesurer le cap
        self.pas_cap = pas_cap  # m, ligne droite pour mesurer le cap au depart
        self.patinage = patinage  # deplacement vu / deplacement des roues sous ce rapport
        self.delai_pret, self.delai_retour = delai_pret, delai_retour  # s
        # s sans reponse avant de declarer la liaison perdue : une reponse perdue sur le Wi-Fi
        # n'est pas une coupure (le cerveau marquerait sinon une zone interdite)
        self.delai_liaison = delai_liaison
        self.inclinaison_camera, self.portee_camera = inclinaison_camera, portee_camera
        # s : retard du flux video du rover jusqu'a la perception (une image « fraiche » est
        # arrivee apres la demande, mais a ete prise jusqu'a autant plus tot)
        self.latence_video = latence_video
        self.horloge, self.dormir = horloge, dormir
        # m : un /path ne part jamais vers un point plus loin que le pas demande + marge_pas
        # de la pose odometrique actuelle (repere faux apres un redemarrage non vu...)
        self.marge_pas = marge_pas
        self.signaler = signaler or (lambda texte: None)
        self.joignable, self.pret = True, False
        self._dernier_etat = None  # derniere reponse de GET /state
        self._redemarrage = False  # le serveur a redemarre depuis le dernier recalage
        self.redemarrages = 0
        # /stop : stop_count du serveur au dernier controle (None : champ absent ou pas encore
        # lu) et /stop envoyes par l'adaptateur depuis, comptes seulement s'ils ont eu une
        # reponse (un /stop recu dont la reponse se perd donne au pire un faux arret operateur)
        self._stops_vus, self._stops_envoyes = None, 0
        # odom.odometer a la fin du dernier pas (rover arrete) : s'il a grandi au pas suivant,
        # quelqu'un a conduit le rover entre les deux (None : rien a comparer)
        self._odo_fin = None
        super().__init__(carte)  # appelle placer((0, 0)), qui ne parle pas encore au rover

    # ---- liaisons et temps -------------------------------------------------
    def _demander(self, methode, chemin, corps=None):
        reponse = self.requete(methode, chemin, corps)
        self.joignable = reponse is not None
        if reponse is not None and (methode, chemin) == ('GET', '/state'):
            if self._dernier_etat is not None and serveur_redemarre(self._dernier_etat, reponse):
                self._redemarrage = True
            if self._dernier_etat is not None and self._stops_vus is not None:
                if self._dernier_etat.get('boot_id') != reponse.get('boot_id'):
                    # nouveau processus : son stop_count part de zero, et tout /stop qu'il a
                    # recu sans qu'on le sache compte comme un STOP d'operateur (un /reset_odom
                    # garde le boot_id et le compteur : rien a remettre a zero)
                    self._stops_vus, self._stops_envoyes = 0, 0
            self._dernier_etat = reponse
        elif reponse is not None and (methode, chemin) == ('POST', '/stop'):
            self._stops_envoyes += 1
        return reponse

    def _insister(self, methode, chemin, corps=None):
        """Comme _demander, en reessayant pendant delai_liaison s. A reserver aux requetes
        qu'on peut renvoyer sans danger (GET, /path en repere odometrie)."""
        return self._envoyer(methode, chemin, corps)[0]

    def _envoyer(self, methode, chemin, corps=None):
        """(reponse, nombre d'envois) : on reessaie pendant delai_liaison s. Un /path deja recu
        et renvoye remplace le pas en cours par le meme ('replaced by a new command' : le
        nombre d'envois dit combien de ces statuts viennent de nous)."""
        debut, envois = self.temps(), 0
        while True:
            reponse = self._demander(methode, chemin, corps)
            envois += 1
            if reponse is not None or self.temps() - debut >= self.delai_liaison:
                return reponse, envois
            self.dormir(self.periode)

    def _envoyer_une_fois(self, chemin, corps, dernier):
        """POST d'un pas RELATIF (/move) : apres une reponse perdue, renvoye seulement si GET
        /state n'en montre aucune trace (ni en cours, ni en file, ni dans ctrl.history apres
        `dernier`) : recu deux fois, il se ferait deux fois. Renvoie la reponse, {} si le pas
        a ete recu sans reponse, ou None si la liaison est perdue (delai_liaison s)."""
        debut = self.temps()
        while True:
            reponse = self._demander('POST', chemin, corps)
            if reponse is not None:
                return reponse
            etat = self._etat()
            if etat is None or self.temps() - debut >= self.delai_liaison:
                return None
            ctrl = etat.get('ctrl') or {}
            trace = ctrl.get('mode', 'idle') != 'idle' or ctrl.get('queued')
            trace = trace or ctrl.get('current') is not None
            if self._redemarrage or trace or nouvelles_entrees(ctrl.get('history') or [], dernier):
                return {}  # recu (ou redemarrage : le suivi du pas le traite)
            self.dormir(self.periode)

    def _etat(self):
        return self._insister('GET', '/state')

    @staticmethod
    def _est_pret(etat):
        """ESP32 connecte, gyroscope calibre, et rien ne fait rouler le rover (pas en cours ou
        en file, recul du garde-fou d'inclinaison) : sinon un recalage sur image se ferait sur
        un rover qui roule. Champs ctrl facultatifs (serveur plus ancien : pas controles)."""
        ctrl = etat.get('ctrl') or {}
        return (
            bool((etat.get('link') or {}).get('connected'))
            and bool(etat['odom'].get('gyro_ready'))
            and ctrl.get('mode', 'idle') == 'idle'
            and not ctrl.get('queued')
            and ctrl.get('current') is None
            and not garde_recule(ctrl)
        )

    def _apres_redemarrage(self):
        """Le serveur a redemarre (ou l'odometrie a ete remise a zero) : le repere carte <-
        odometrie ne vaut plus rien. On attendra l'ESP32 et le gyroscope, puis on recalera sur
        une image neuve ; le cap sera remesure si la camera ne le donne pas."""
        self._redemarrage = False
        self.redemarrages += 1
        self.pret = False
        self._a_recaler = True
        self.repere = Repere()
        self._appui = self._vue = None
        self._odo_fin = None
        self.signaler(
            'serveur du rover redemarre (odometrie remise a zero) -> attente du rover, '
            'nouveau recalage sur la camera avant tout mouvement'
        )

    def arreter(self):
        self._demander('POST', '/stop')

    def _controler_stops(self, etat):
        """Avant un pas : un STOP presse pendant que le rover attendait (aucune trace dans
        ctrl.history) se voit au compteur stop_count, s'il a grandi de plus que les /stop de
        l'adaptateur. Sans ce champ (serveur plus ancien) : rien a controler."""
        n = etat.get('stop_count')
        if n is None:
            return
        vus, envoyes = self._stops_vus, self._stops_envoyes
        if vus is not None and n < vus:
            # compteur reparti de zero sous le meme boot_id (Rover recree) : comme apres un
            # redemarrage, tout /stop qu'il a compte vient d'ailleurs
            vus, envoyes = 0, 0
        if vus is not None and n - vus > envoyes:
            raise ArretOperateur(
                f'STOP recu par le serveur du rover entre deux pas (stop_count {vus} -> {n}, '
                f'{envoyes} /stop de la mission) : fin de mission'
            )
        self._stops_vus, self._stops_envoyes = n, 0  # premier controle, ou rien d'anormal

    def _panne(self, statut, ctrl):
        """Le serveur a arrete le rover de lui-meme pendant le pas : ce n'est pas un pas fini.
        Le pas suivant se recalera sur une image neuve (le cerveau arrete et reessaie)."""
        self._a_recaler = True
        raise ServeurRoverError(
            f"Pas abandonne par le serveur du rover (statut '{statut}', ctrl.fault "
            f"{ctrl.get('fault')!r}) : rover arrete, voir GET /state (link, ctrl.fault)."
        )

    def temps(self):
        return self.horloge()

    def _duree_pas(self, longueur):
        """s que le rover met a faire un pas de `longueur` m (attente_pas)."""
        return longueur / self.vitesse

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
        self._odo_fin = None

    def recaler(self):
        mesure = self._mesure_fraiche()
        etat = self._etat()
        if self._redemarrage:
            self._apres_redemarrage()  # sinon on recalerait avec le cap de l'ancienne odometrie
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
            if (etat.get('ctrl') or {}).get('mode') == 'manual':
                raise ArretOperateur('rover conduit a la main (commande manuelle /cmd)')
            if self._est_pret(etat):
                self.pret = True
                return True
            if self.temps() - debut > self.delai_pret:
                raise ServeurRoverError(
                    f'Rover pas pret apres {self.delai_pret:.0f} s : ESP32 non connecte, '
                    'gyroscope pas calibre ou rover qui roule encore (voir GET /state, link, '
                    'odom.gyro_ready, ctrl.mode et ctrl.guard).'
                )
            self.dormir(0.5)

    def _borner(self, corps, avant, longueur):
        """Refuse un /path qui irait plus loin que le pas demande : la cible est comparee a la
        pose odometrique lue juste avant, pas a ce que l'adaptateur croit."""
        but = corps['points'][-1]
        ici = pose_odom(avant)[:2]
        ecart = math.dist(but, ici)
        if ecart > longueur + self.marge_pas:
            raise ServeurRoverError(
                f'Pas refuse : la cible est a {ecart:.2f} m de la pose odometrique du rover '
                f'pour un pas de {longueur:.2f} m (marge {self.marge_pas:.2f} m). Repere '
                'carte <- odometrie faux (serveur redemarre, /reset_odom ?) : rien envoye.'
            )

    def _faire_un_pas(self, chemin, corps, longueur, avant=None):
        """Envoie un pas au serveur et le suit jusqu'a l'arret. Renvoie (statut, etat) : le
        statut du serveur ('done', 'stalled', 'timeout'...), 'delai' s'il ne finit pas,
        'redemarrage' si le serveur a redemarre, ou (None, None) si la liaison est perdue.
        Leve ArretOperateur si un operateur a arrete ou remplace le pas. `avant` : la reponse
        de GET /state que l'appelant vient de lire (rien entre elle et le POST), sinon elle
        est lue ici : une requete de moins par pas."""
        self._odo_fin = None
        if avant is None:
            avant = self._etat()
        if avant is None:
            return None, None
        if self._redemarrage:
            self.arreter()  # /reset_odom : le controleur peut encore suivre un ancien pas
            return 'redemarrage', avant  # le repere a perime : rien n'est envoye
        self._controler_stops(avant)
        if chemin == '/path':
            self._borner(corps, avant, longueur)
        dernier = (avant['ctrl'].get('history') or [None])[0]
        if chemin == '/path':
            reponse, envois = self._envoyer('POST', chemin, corps)
        else:  # un /move est relatif : renvoye apres une reponse perdue, il se referait
            reponse, envois = self._envoyer_une_fois(chemin, corps, dernier), 1
        if reponse is None:
            return None, None
        # le serveur abandonne un pas apres max(20 s, 3 x longueur / vitesse + 10 s) : on lui
        # laisse 5 s de plus pour le faire lui-meme
        delai = max(20.0, 3.0 * longueur / self.vitesse + 10.0) + 5.0
        debut, vu_en_cours = self.temps(), False
        etat = avant
        # attente_pas : premiere lecture a la fin attendue du pas, pas avant ; ce silence voulu
        # ne compte pas dans delai_liaison
        attente = max(0.0, self._duree_pas(longueur) - self.periode) if self.attente_pas else 0.0
        derniere_reponse = debut + attente
        while self.temps() - debut < delai:
            self.dormir(self.periode + attente)
            attente = 0.0
            etat = self._demander('GET', '/state')
            if etat is None:
                if self.temps() - derniere_reponse < self.delai_liaison:
                    continue  # une reponse perdue : le serveur continue le pas sans nous
                return None, None  # liaison perdue : le serveur finit le pas puis s'arrete
            derniere_reponse = self.temps()
            if self._redemarrage:
                # l'historique et l'odometrie sont repartis a zero ; apres un /reset_odom le
                # controleur suit encore le pas, vers une cible de l'ancien repere : on l'arrete
                self.arreter()
                return 'redemarrage', etat
            self._suivre(etat)
            ctrl = etat['ctrl']
            if ctrl.get('mode') == 'manual':
                raise ArretOperateur(
                    'rover repris en main (commande manuelle /cmd) pendant un pas'
                )
            # en attente, en cours, ou a l'arret le temps de mesurer (settle, 0,4 s)
            en_cours = ctrl.get('mode') in ('move', 'turn', 'path', 'settle')
            # le garde-fou d'inclinaison finit le pas ('tilt: ...') puis fait reculer le rover
            # (jusqu'a 15 cm / 3 s) en mode 'idle' : il roule encore
            en_cours = en_cours or garde_recule(ctrl)
            if en_cours or ctrl.get('queued') or ctrl.get('current') is not None:
                vu_en_cours = True
                continue
            nouvelles = nouvelles_entrees(ctrl.get('history') or [], dernier)
            if nouvelles:
                self._controler_operateur(nouvelles, envois)
                if panne_serveur(nouvelles[0].get('status')):
                    self._panne(nouvelles[0].get('status'), ctrl)
                if nouvelles[0].get('status') == 'done' and ctrl.get('fault'):
                    # panne pendant le settle (0,4 s) : l'historique dit 'done', ctrl.fault non
                    self._panne(nouvelles[0].get('status'), ctrl)
                self._odo_fin = odometre(etat)
                return nouvelles[0].get('status'), etat
            if panne_serveur(ctrl.get('fault')):
                # Pas sorti de la file sans trace, panne apres notre POST (qui remet ctrl.fault
                # a None) : un STOP compte par le serveur passe avant la panne.
                self._controler_stops(etat)
                self._panne(ctrl.get('fault'), ctrl)
            if vu_en_cours:  # parti de la file sans laisser de trace : un /stop l'a videe
                raise ArretOperateur("pas annule avant de partir (/stop d'un autre client)")
        self.arreter()
        return 'delai', self._etat() or etat

    @staticmethod
    def _controler_operateur(nouvelles, envois):
        """Statuts arrives pendant notre pas. Seuls nos propres renvois (envois - 1) peuvent
        expliquer un 'replaced by a new command' ; tout le reste vient d'un operateur."""
        statuts = [h.get('status') for h in nouvelles]
        remplaces = statuts.count('replaced by a new command')
        autres = [s for s in statuts if s in ARRETS_OPERATEUR[:2]]
        if autres or remplaces > envois - 1:
            cause = autres[0] if autres else 'replaced by a new command'
            raise ArretOperateur(
                f"pas interrompu par un operateur (statut serveur '{cause}') : fin de mission"
            )

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
        bloque = patine or statut_de_choc(statut)
        self._recaler_a_l_arret(mesure, etat, nouvel_appui=bloque)
        return patine

    def _devant(self, point, direction, recul=0.0):
        d = 0.15 + recul  # recul : ce que le garde-fou d'inclinaison a fait reculer le rover
        return point[0] + d * math.cos(direction), point[1] + d * math.sin(direction)

    def _degagement(self, portee):
        """Distance (m) dont le rover peut avancer dans N'IMPORTE QUELLE direction sans que sa
        carrosserie (rayon_effectif de la carte) atteigne un obstacle connu de la carte ;
        `portee` s'il n'y en a aucun a cette distance. Les cases pas vues ne comptent pas."""
        x, y = self.est
        r = portee + self.carte.rayon_effectif
        g = self.carte.grille(x - r, y - r, x + r, y + r)
        ii, jj = (g.val == OBSTACLE).nonzero()
        if not len(ii):
            return portee
        # distance du rover au carre de chaque case obstacle
        x0, y0 = (jj + g.j0) * g.res, (ii + g.i0) * g.res
        dx = (x0 - x).clip(min=0) + (x - x0 - g.res).clip(min=0)
        dy = (y0 - y).clip(min=0) + (y - y0 - g.res).clip(min=0)
        libre = float(((dx * dx + dy * dy) ** 0.5).min()) - self.carte.rayon_effectif
        return max(0.0, min(portee, libre))

    def _pas_pour_le_cap(self):
        """Distance visee par la ligne droite qui sert a mesurer le cap. C'est un /path : le
        rover parcourt entre la visee moins TOLERANCE_BUT et la visee, d'ou une visee de
        TOLERANCE_BUT plus loin que la longueur voulue (pas_cap). Le cap n'etant pas connu, on
        ne sait pas vers ou le rover part : la visee est donc limitee a ce qui est libre tout
        autour de lui sur la carte. Trop pres d'un obstacle pour ca, il vise le plus petit pas
        qui se mesure encore a l'arrivee (base_cap + 2 cm, plus TOLERANCE_BUT) et seule la
        detection de choc le protege."""
        minimum = self.base_cap + 0.02 + TOLERANCE_BUT
        visee = self.pas_cap + TOLERANCE_BUT
        return max(minimum, min(visee, self._degagement(visee)))

    def _mesurer_le_cap(self, frais=None):
        """Cap inconnu (rover sans marqueur) : le rover avance tout droit (au plus pas_cap m,
        voir _pas_pour_le_cap), et la camera et l'odometrie voient le meme deplacement. S'il
        est bloque devant, il essaie en arriere. Renvoie le point d'impact s'il etait bloque en
        avancant, sinon None. `frais` : GET /state que deplacer vient de lire."""
        bloque = None
        pas = round(self._pas_pour_le_cap(), 3)
        for distance in (pas, -pas):
            # En avant : un /path vers un point ABSOLU de l'odometrie, calcule sur la pose lue
            # juste avant : renvoye apres une reponse perdue, il ne fait pas le pas deux fois
            # (un /move relatif repartait de la ou le rover en etait). En arriere : un /move,
            # envoye une seule fois. Un /path vers un point derriere le rover a moins de
            # path_lookahead (0,2 m) finit 'done' sans bouger sur le vrai serveur.
            etat = frais or self._etat()
            frais = None  # en arriere : relu apres l'arret et l'image
            if etat is None or self._redemarrage:
                return None
            if distance < 0:
                corps = {'distance': distance}
                statut, etat = self._faire_un_pas('/move', corps, -distance, avant=etat)
            else:
                xo, yo, ho = pose_odom(etat)
                but = (xo + distance * math.cos(ho), yo + distance * math.sin(ho))
                corps = {
                    'points': [[round(but[0], 4), round(but[1], 4)]],
                    'frame': 'odom',
                    'speed': self.vitesse,
                }
                statut, etat = self._faire_un_pas('/path', corps, distance, avant=etat)
            if statut is None or statut == 'redemarrage':
                return None
            patine = self._arret(statut, etat)
            if distance > 0 and (patine or statut_de_choc(statut)):
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
        (roues bloquees, ou qui patinent d'apres la camera), sinon None. Leve ArretOperateur
        si un operateur a arrete le rover pendant le pas."""
        etat = self._etat()
        if etat is None:
            return None
        if (etat.get('ctrl') or {}).get('mode') == 'manual':
            raise ArretOperateur('rover conduit a la main (commande manuelle /cmd) entre deux pas')
        # Conduite au clavier finie depuis plus de manual_timeout (0,5 s) : ctrl.mode est revenu
        # a 'idle' sans trace, mais les roues ont tourne depuis la fin du dernier pas
        odo, fin = odometre(etat), self._odo_fin
        if odo is not None and fin is not None and not self._redemarrage and odo - fin > 0.05:
            raise ArretOperateur(
                f'rover deplace entre deux pas ({odo - fin:.2f} m aux roues sans pas de la '
                'mission : conduite au clavier ?) : fin de mission'
            )
        for _ in range(3):  # redemarrages a la suite : on finit par abandonner
            if self._redemarrage:
                self._apres_redemarrage()
            elif not self._est_pret(etat):
                self.pret = False  # ESP32 deconnecte ou gyroscope a recalibrer
                self._a_recaler = True
            else:
                self.pret = True  # vu pret sur le GET /state qu'on vient de lire
            if self.pret:
                break
            if not self._attendre_le_rover():
                return None
            etat = self._dernier_etat
            if not self._redemarrage:
                break
        else:
            raise ServeurRoverError('Le serveur du rover redemarre sans arret : mission arretee.')
        if self._a_recaler:
            # l'image d'abord : l'etat lu ensuite est le plus recent, il sert aussi au pas
            mesure = self._mesure_fraiche()
            etat = self._etat()
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
            impact = self._mesurer_le_cap(etat if etat is self._dernier_etat else None)
            if impact is not None or not self.joignable or self.repere.angle is None:
                return impact  # choc, liaison perdue, ou redemarrage : le pas suivant reprend
        but = self.repere.vers_odom(*cible)
        corps = {
            'points': [[round(but[0], 4), round(but[1], 4)]],
            'frame': 'odom',
            'speed': self.vitesse,
        }
        # etat : la derniere reponse de GET /state (debut de deplacer ou recalage, rien n'est
        # parti depuis) sert au pas ; apres la mesure du cap, il est relu
        avant = etat if etat is self._dernier_etat else None
        statut, etat = self._faire_un_pas('/path', corps, math.dist(self.est, cible), avant)
        if statut is None:
            return None  # liaison perdue : le cerveau le voit avec rover_joignable()
        if statut == 'redemarrage':
            return None  # rien n'a bouge ; le pas suivant attend le rover et recale tout
        patine = self._arret(statut, etat)
        if not (patine or statut_de_choc(statut)):
            return None  # 'done', ou arrete sans choc ('stopped', 'delai'...)
        # garde-fou d'inclinaison : un choc aussi, sinon le pas suivant remonte sur la bosse
        recul = RECUL_GARDE if garde_declenche(statut) else 0.0
        dx, dy = cible[0] - self.est[0], cible[1] - self.est[1]
        if math.hypot(dx, dy) > 0.02:
            return self._devant(self.est, math.atan2(dy, dx), recul)
        return self._devant(self.est, self.cap, recul) if self.cap is not None else self.est

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
            if self._redemarrage:
                return True  # repere perime : le prochain pas attend le rover et recale tout
            mesure = self._mesure_fraiche()
            if mesure is not None:
                self._recaler_a_l_arret(mesure, etat)
            else:
                self._suivre(etat)
            return True
        return False

    # ---- perception et drone -----------------------------------------------
    @staticmethod
    def _pan(etat):
        """gimbal.pan (degres) dans une reponse de GET /state, ou None si le champ manque."""
        pan = ((etat or {}).get('gimbal') or {}).get('pan')
        return None if pan is None else float(pan)

    def _percevoir_a(self, pan):
        cap = None if self.cap is None else angle_normalise(self.cap - math.radians(pan))
        vue = self.percevoir(self.est, cap)
        if vue is not None:
            self.publier_vue(*vue)

    def _tourelle_arretee(self):
        """Apres POST /gimbal : le pan rapporte par GET /state une fois la tourelle arretee
        (gimbal.moving faux, champ facultatif), ou None si le serveur ne donne pas gimbal."""
        if self._pan(self._dernier_etat) is None:
            return None  # serveur sans gimbal dans /state : on garde l'angle commande
        debut = self.temps()
        while True:
            etat = self._demander('GET', '/state')
            if etat is None:
                return None
            if not (etat.get('gimbal') or {}).get('moving') or self.temps() - debut > 5.0:
                return self._pan(etat)
            self.dormir(0.2)

    def observer(self, tour_complet=False):
        """Ce que voit la camera ; tour_complet : la tourelle fait le tour (8 vues). Pour le
        serveur, pan + = a droite : le cap de la camera est donc cap - pan. Hors du tour, le
        pan est celui de la derniere reponse de GET /state (la tourelle se regle au clavier sur
        la page du rover), 0 si le champ manque."""
        if not tour_complet:
            if self._dernier_etat is None:
                self._demander('GET', '/state')  # avant le premier pas : pan pas encore lu
            pan = self._pan(self._dernier_etat)
            self._percevoir_a(0.0 if pan is None else pan)
            return
        for pan in [-180, -135, -90, -45, 0, 45, 90, 135]:
            corps = {'pan': pan, 'tilt': self.inclinaison_camera}
            if self._demander('POST', '/gimbal', corps) is None:
                continue  # tourelle peut-etre pas tournee : la vue serait posee de travers
            self.dormir(0.4)  # pivot de la tourelle
            reel = self._tourelle_arretee()
            # puis une image prise apres l'arret de la tourelle doit atteindre la perception
            self.dormir(self.latence_video)
            self._percevoir_a(pan if reel is None else reel)
        corps = {'pan': 0, 'tilt': self.inclinaison_camera}
        if self._demander('POST', '/gimbal', corps) is not None and self._pan(self._dernier_etat):
            # la prochaine vue hors tour lit le pan dans _dernier_etat : il n'est plus a jour
            gimbal = dict(self._dernier_etat['gimbal'], pan=0.0)
            self._dernier_etat = dict(self._dernier_etat, gimbal=gimbal)

    def demander_zoom(self, tuile):
        for msg in self.drone.zoomer(tuile) or []:
            self.carte.recevoir(msg)
