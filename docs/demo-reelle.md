# Démo réelle : marche à suivre (15 nov. 2026)

Pour l'équipe sur place cet après-midi. **Rien de ce qui suit n'a encore tourné sur le vrai
robot ni sur une vraie caméra** : les commandes ont été vérifiées sur la VM (`--help`, essais à
blanc sur une photo, faux rover), pas sur le matériel. Ce qui n'a pas été vérifié est marqué
*non vérifié*. Gardez toujours quelqu'un la main sur le Ctrl-C (voir « Arrêter »).

## 1. Qui tourne où

| Programme | Où | Pourquoi |
|---|---|---|
| Caméra du dessus (téléphone avec une appli caméra IP, ou caméra USB) | dans la salle, fixée au-dessus de l'arène, objectif vers le bas | c'est le « drone » |
| `merging_map/lancer_overhead.py` (la mission) | **un portable sur le Wi-Fi de la salle**, le même réseau que le téléphone et le rover | il doit lire le flux du téléphone et parler au rover ; la VM Google Cloud ne peut pas ouvrir le téléphone sur le Wi-Fi de la salle |
| `rover/rover_server.py` | sur le rover (RDK X5), service `polaris-rover-api` | seul programme qui parle à l'ESP32 |
| `teleop/server.py --suivi` (page du jury, `/mission`) | le **même portable** que la mission | il lit le fichier `--suivi` écrit par la mission, en local |
| `rover_mapper/perception_service.py` (option) | la VM (il faut le GPU) | voir l'étape 9 ; pas pour la première démo |

Faire tourner la mission sur la VM demanderait un tunnel pour la caméra, par exemple depuis le
portable `ssh -R 8170:<IP du téléphone>:8080 <vous>@<VM>` (le flux devient
`http://127.0.0.1:8170/video` sur la VM), et le rover joint par son propre tunnel
(`http://127.0.0.1:18080` sur la VM, hors ligne ce matin). Avec 160-700 ms d'aller-retour vers
le robot et 0,5 s de retard vidéo, c'est plus lent et moins sûr : *non vérifié*, à éviter.

## 2. Installer sur le portable (pas de ROS)

Python 3.10 ou plus, `git`, puis à la racine du dépôt (branche `Romain` ou celle qu'on vous
donne) :

```bash
pip install numpy opencv-contrib-python requests
```

`opencv-contrib-python` (pas `opencv-python`) : il faut le module ArUco. `requests` ne sert qu'à
la page du jury. Dans chaque terminal, à la racine du dépôt, avant toutes les commandes
`python` ci-dessous :

```bash
export PYTHONPATH=merging_map:src/guidance_calib:src/guidance_global_map:src/guidance_rover_pose
```

Sous Windows (PowerShell) : `$env:PYTHONPATH="merging_map;src/guidance_calib;src/guidance_global_map;src/guidance_rover_pose"`
(séparateur `;`, *non vérifié*).

## 3. Pas à pas

### 3.1 Monter la caméra et vérifier le flux

Caméra fixe, bien à la verticale, assez haute pour voir toute l'arène (objectif principal du
téléphone, pas l'ultra grand-angle ; exposition et mise au point bloquées si l'appli le permet).
Arène : sol mat d'une couleur, obstacles de couleurs vives, **rien d'autre au sol** (pas de
ruban, de câble, de sac, de marqueur ArUco au sol : sur la photo d'essai, les marqueurs posés au
sol sortent comme des obstacles), lumière égale sans ombres dures.

```bash
python -m guidance_global_map.measure_scale http://<IP du téléphone>:8080/video --distance 1 --save frame.jpg
```

Source : `http://<IP>:8080/video` (flux MJPEG, appli du type « IP Webcam »), ou une URL d'image
unique comme `http://<IP>:8080/shot.jpg`, ou `0` pour une caméra USB du portable. Sans
`--points`, la commande s'arrête en affichant `image is LxH; give --points ...` : c'est normal,
elle a lu une image (taille affichée) et l'a enregistrée dans `frame.jpg`. Vérifiez que
`frame.jpg` est récente et montre toute l'arène. (Lecture de flux testée seulement contre des
serveurs locaux, jamais contre un téléphone.)

### 3.2 Mesurer les mètres par pixel (au mètre ruban)

Posez un mètre ruban à plat au milieu de la vue, prenez deux marques à 1 m ou plus, relevez
leurs pixels dans `frame.jpg` (n'importe quel visualiseur d'images), puis :

```bash
python -m guidance_global_map.measure_scale frame.jpg --points U1 V1 U2 V2 --distance 1.0
# ou, sur un portable avec écran : ... frame.jpg --distance 1.0 --click
```

Exemple vérifié : `image 780x630, 333.0 px for 1.0 m` puis
`metres_per_pixel: 0.003003  (view 2.34 x 1.89 m)`. La vue affichée doit correspondre à
l'arène. **Remesurez si la résolution du flux change ou si la caméra bouge.** Retirez le mètre
avant la mission.

### 3.3 Le marqueur ArUco sur le rover

- Id **10**, dictionnaire `DICT_4X4_50` :
  `python -m guidance_calib.generate_markers --ids 10 --out markers`
  écrit `markers/aruco_10.png` (vérifié).
- Imprimé d'au moins 10 cm de côté avec sa marge blanche, collé **à plat** sur le dessus du
  rover, **bord du haut du marqueur vers l'avant du rover** (aucune option de
  `lancer_overhead.py` ne corrige un autre montage).
- Toujours lancer la mission avec `--marqueur 10`. Sans marqueur, le rover est cherché comme un
  objet de sa taille : sur la photo d'essai, une boîte de 20 x 26 cm a été prise pour le rover
  (0,84 ; 0,75) alors que le vrai rover était en (0,62 ; 1,33), et le cap est inconnu.
- Hauteur du marqueur : mesurez-la au mètre (sol -> surface du marqueur) et donnez-la avec
  `--hauteur-marqueur h`, **avec `--hauteur H`** (sol -> objectif de la caméra, 1,9 m par
  défaut) : la correction de parallaxe utilise les deux. Sans `--hauteur-marqueur`, aucune
  correction : la position vue est décalée vers l'extérieur d'environ distance au centre de
  l'image x h / (H - h), soit 8 à 9 cm à 1 m du centre pour h = 0,15 m et H = 1,9 m. Vérifié
  seulement sur la photo d'essai (le rover passe de (0,62 ; 1,33) à (0,66 ; 1,30) avec
  `--hauteur-marqueur 0.15`, vers le centre de l'image) ; au mètre sur le vrai rover : *non
  vérifié*.

Contrôle sur une image (écrit `salle_overlay.jpg` : obstacles en rouge, rover en bleu, rien
d'autre en rouge) :

```bash
python -m guidance_global_map.segment_image frame.jpg salle --metres-per-pixel 0.0030 --rover-marker 10
```

### 3.4 Le serveur du rover : GET /state

Le rover est à `http://<IP du rover>:8080` (IP sur le Wi-Fi de la salle : sur le robot,
`hostname -I`). Rover posé, immobile :

```bash
curl -s http://<IP du rover>:8080/state
```

| Champ | Ce qu'il dit |
|---|---|
| `link.connected` | l'ESP32 répond par l'USB ; `false` : rien ne bougera |
| `odom.gyro_ready` | biais du gyroscope mesuré ; `false` pendant ~10 s après un démarrage (ESP32 ~7 s + 2 s) : **ne pas toucher le rover** ; tout pas est refusé (HTTP 409) avant |
| `boot_id`, `uptime_s` | identité du processus serveur et depuis quand il tourne ; un `boot_id` qui change = serveur redémarré. **Absent** = le robot a encore l'ancien serveur (voir section 6) |
| `ctrl.mode`, `ctrl.fault` | `idle` au repos ; `fault` dit pourquoi le serveur a coupé un pas de lui-même (`no ESP32 feedback`, `ESP32 link lost`, `controller error: ...`) ; la page du rover l'affiche dans un bandeau rouge |
| `stop_count` | nombre de STOP reçus (`POST /stop`, `/cmd {"T":0}`) depuis le démarrage du serveur ; la mission s'en sert pour voir un STOP entre deux pas. **Absent** = ancien serveur : ce STOP n'est pas vu (section 4) |
| `loop_errors`, `link.handler_errors` | erreurs de la boucle de commande ; doivent rester à 0 |

(Champs vérifiés sur `rover_server.py --sim` du dépôt ; la version du robot n'a jamais été
comparée.)

### 3.5 Essai à blanc (`--dry-run`) : aucun POST ne part

```bash
python merging_map/lancer_overhead.py --camera http://<IP du téléphone>:8080/video \
    --metres-par-pixel 0.0030 --marqueur 10 --hauteur <H> --hauteur-marqueur <h> \
    --rover http://<IP du rover>:8080 --victime 1.8 1.5 --suivi $HOME/suivi.json --dry-run
```

`--victime X Y` et `--depart X Y` sont dans le repère de la caméra : origine au **coin en bas à
gauche de l'image**, x vers la droite, y vers le haut, en mètres. Depuis un pixel (u, v) de
`frame.jpg` de hauteur H pixels : x = u x mpp, y = (H - v) x mpp. La victime doit être dans la
zone, sinon `ValueError ... hors de la zone`.

Lire la sortie (exemple assemblé à partir de deux essais à blanc vérifiés sur la photo d'essai :
la ligne `Rover joint` contre `rover_server.py --sim`, les autres avec `--marqueur 10`) :

```
Rover joint : liaison ESP32 True, gyroscope pret True, boot_id 122b409f3c60dd72
Zone vue par la camera : 2.34 x 1.89 m (...) ; echelle 3.000 mm/pixel (--metres-par-pixel)
Rover vu par la camera : (0.62, 1.33) m, cap -34 deg
[dry-run] decision : SUIVRE, chemin de 3.88 m
[dry-run] pas 1 : (0.62, 1.33) -> (0.49, 1.19) m, 0.19 m
```

À vérifier : la zone mesure bien l'arène ; la position du rover tombe à quelques cm près de la
vraie (au mètre) ; le cap est donné (sinon le marqueur n'est pas vu) ; la décision et les
premiers pas ont du sens ; **le rover n'a pas bougé**. `--camera` sans `--metres-par-pixel` n'est
accepté qu'avec `--dry-run` (échelle déduite de `--hauteur`, peu fiable). `--image photo.jpg`
(photo fixe) n'est accepté qu'avec `--dry-run` (sinon refus, code 2).

### 3.6 Premier vrai pas, la main sur l'arrêt

Rover posé dans une zone dégagée, victime proche (moins de 1 m), une personne à Ctrl-C, une
autre près du rover. Même commande **sans `--dry-run`**, puis Ctrl-C dès le premier pas fini.
Contrôlez : le rover a fait un pas d'environ 20 cm dans la bonne direction, il s'est arrêté au
Ctrl-C, la sortie dit `Interrompu (Ctrl-C) : /stop envoye au rover.`

Sans cap (marqueur non vu), le premier mouvement est une ligne droite d'environ 15 cm **dans une
direction inconnue** (ou un pivot puis 15 cm en arrière si l'avant est bloqué) : laissez de la
place tout autour.

### 3.7 La mission complète

Même commande sans `--dry-run`. Options utiles : `--vitesse 0.15` (m/s, défaut 0,2),
`--delai-http 2` si le Wi-Fi est lent, `--latence 0.8` si le flux du téléphone a plus de 0,5 s
de retard (la caméra attend une image arrivée au moins `--latence` s après l'arrêt du rover),
`--tolerance 0.15` (arrivée à moins de, en m). En fin de mission, deux lignes :
`Fin de mission (arrive) : x m parcourus.` (ou `(abandon)`, `(erreur)`, `(arret_operateur)`),
puis `Issue : <issue> ; raison : <raison>` qui dit pourquoi elle s'est arrêtée (`arrivee`,
`objectif_degrade`, `erreur_robot`, `rover_injoignable`, `sans_progres`, `arret_operateur`...).

### 3.8 La page du jury

Sur le même portable, dans un autre terminal (il faut `requests`) :

```bash
python teleop/server.py --robot http://<IP du rover>:8080 --suivi $HOME/suivi.json --port 8082
```

puis ouvrir `http://127.0.0.1:8082/mission` en plein écran (même chemin absolu que le `--suivi`
de la mission). Pour un écran sur une autre machine, `--host 0.0.0.0` ouvre le serveur au réseau
de la salle, **y compris la conduite du rover (`/api/cmd`)** : préférez brancher le projecteur
sur le portable.

Répéter sans robot :

```bash
python teleop/demo_mission.py --port 8150     # puis http://127.0.0.1:8150/mission
```

(Vérifié : `/mission` et `/api/mission` répondent 200, et un `--suivi` écrit par un essai à
blanc de `lancer_overhead.py` s'affiche dans `/api/mission`. Aucun navigateur n'a encore
affiché la page : regardez-la avant la démo.)

### 3.9 Option : la perception du rover (pas pour la première démo)

`rover_mapper/perception_service.py` a besoin de Depth Anything 3 et du GPU : il tourne sur la
VM, lit la caméra du rover par le tableau de bord de la VM (donc par le tunnel du robot), et
n'écoute que sur 127.0.0.1. Chaîne *non vérifiée* :

```bash
# sur la VM (tunnel du robot en marche, tableau de bord teleop sur 8082)
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/perception_service.py \
    --source http://127.0.0.1:8082/video_rect --cam-height 0.22 --fov 100 --port 8095
# sur le portable
ssh -N -L 8095:127.0.0.1:8095 <vous>@<VM>
# puis ajouter à la mission : --perception http://127.0.0.1:8095
# et à la page du jury : --perception-debug http://127.0.0.1:8095/debug.jpg
```

Mesurez d'abord la hauteur réelle de la caméra du rover (`--cam-height`) et regardez
`/debug.jpg` avec une boîte à 0,5 m puis 1 m (rouge = obstacle, vert = sol). Une réponse de plus
de 2 s est ignorée : le rover roule alors sur la seule carte du dessus, sans erreur.

## 4. Arrêter

- **Ctrl-C sur le portable : l'arrêt sûr.** Pendant la mission, il l'arrête et envoie
  `POST /stop` au rover (3 essais). (Avant la première image, rien n'a encore été envoyé.)
- STOP de la page du rover **pendant un pas** : la mission affiche `ARRET OPERATEUR` et
  s'arrête pour de bon (testé sur le banc simulé). Tenir une flèche de la page (reprise en main)
  fait de même.
- STOP de la page **entre deux pas** (rover immobile) : vu **si le serveur du rover a
  `stop_count` dans `/state`** (le serveur de ce dépôt, pas encore sur le robot : section 6).
  Avant d'envoyer chaque pas, la mission compare ce compteur aux `/stop` qu'elle a envoyés
  elle-même ; s'il y en a plus, elle affiche `ARRET OPERATEUR : STOP recu par le serveur du
  rover entre deux pas ...` et n'envoie plus rien (testé contre un faux serveur et le vrai
  serveur avec un ESP32 simulé, jamais sur le robot). Limites : un STOP pressé entre le
  lancement et le **premier** pas n'est pas vu ; si la réponse d'un `/stop` de la mission se perd
  sur le Wi-Fi, la mission peut s'arrêter comme sur un STOP (fausse alerte, dans le sens sûr).
  **Sans `stop_count`** (ancien serveur sur le robot) : non vu, la mission enverra le pas
  suivant et le rover repartira ; c'est alors Ctrl-C.
- Côté robot : l'ESP32 coupe les moteurs après 3 s sans ordre, une conduite manuelle s'arrête
  0,5 s après la dernière touche, et le serveur de ce dépôt (pas encore sur le robot) arrête
  aussi le rover après 0,5 s sans retour de l'ESP32.

## 5. Quand ça ne va pas

| Ce qu'on lit | Cause probable | Que faire |
|---|---|---|
| `Le rover ne repond pas a GET /state` (rien envoyé) | mauvaise IP/port, serveur arrêté, pas le même Wi-Fi | `curl http://<IP>:8080/state` ; voir section 6 |
| `Aucune image de la camera ... en 15 s` | URL du flux, téléphone en veille, autre réseau | ouvrir l'URL dans un navigateur du portable |
| `overhead camera: frame rejected: ...` puis `La camera ne donne pas d'image` | image noire, surexposée, trop sombre, presque uniforme (`nearly uniform`), main devant | lumière, exposition ; un sol vide très uni peut être rejeté : laisser un obstacle en vue |
| `Rover vu par la camera : NON` | marqueur non vu (taille, reflet, flou, pas à plat), ou rover pris pour un obstacle | vérifier avec `segment_image --rover-marker 10`. `--depart x y` ne règle rien : le premier pas échouera (`Cap du rover inconnu et camera indisponible`) |
| Zone trop grande ou trop petite, rover décalé | `--metres-par-pixel` faux, résolution du flux changée | refaire 3.1 et 3.2 |
| `ERREUR du rover -> arret, nouvel essai : Pas refuse : la cible est a ...` | repère carte <- odométrie faux (redémarrage non vu, `/reset_odom`) ou bruit de la caméra au-delà de 0,2 m | la mission réessaie (`/stop`, 1 s) 3 fois de suite au plus, puis s'arrête ; si ça revient, Ctrl-C et relancer. La marge (0,2 m) n'a pas d'option en ligne de commande |
| `Cap du rover impossible a mesurer` | le rover n'a bougé ni en avant ni en arrière, ou la caméra ne le voit pas | chaque essai refait la ligne droite de 15 cm ; vérifier `gyro_ready`, les roues, le marqueur ; utiliser `--marqueur 10` |
| `serveur du rover redemarre ...` | `rover_server.py` relancé (service, autre portable sur le port série) | la mission attend l'ESP32 et le gyroscope, se recale sur une image neuve et continue ; `... redemarre sans arret : mission arretee` = trouver qui relance le serveur |
| `Rover pas pret apres 15 s` | ESP32 non connecté ou gyroscope pas calibré | `GET /state` : `link.connected`, `odom.gyro_ready` ; ne pas bouger le rover au démarrage |
| `LIAISON ROVER PERDUE` puis `abandon: rover injoignable` | Wi-Fi coupé plus de 2 s ; abandon après ~60 s sans retour | le rover finit son pas puis s'arrête seul ; rapprocher le routeur, `--delai-http 2` |
| `abandon: le rover n'avance plus` | 8 pas sans bouger de 5 cm, ou 4 chocs au même endroit | dégager, vérifier les chenilles |
| `ERREUR du rover -> arret, nouvel essai : Pas abandonne par le serveur du rover (statut 'no ESP32 feedback' ...` (ou `ESP32 link lost`, `controller error`) | le serveur a arrêté le rover de lui-même ; pour `no ESP32 feedback` sans raison visible : retour de l'ESP32 avec des trous > 0,5 s sur le RDK chargé | la mission réessaie après une image neuve, puis s'arrête en `erreur_robot` après 4 erreurs de suite ; regarder `ctrl.fault` et `link.rx_age` dans `/state` (bandeau rouge de la page du rover) ; prévenir Raphaël |
| `ARRET OPERATEUR : STOP recu par le serveur du rover entre deux pas` | quelqu'un a appuyé sur STOP (page du rover, tableau de bord), ou réponse d'un `/stop` de la mission perdue | voulu : c'est fini ; sinon relancer la mission |

Les nombres de cette table (3 essais, 0,2 m, 15 cm, 60 s, 8 pas) viennent de `merging_map/`
(`rover_reel.py`, `cerveau.py`, `mission_settings.py`) au 15 nov. et peuvent changer : Paco y
travaille.

## 6. Déployer le serveur du rover (avec Raphaël)

Le serveur du robot (`/home/sunrise/rover`, lancé par le service `polaris-rover-api` avec
`--fps 40` à travers un script de garde) **diffère du dépôt** et personne n'a comparé. Avec
Raphaël, et jamais pendant une mission :

```bash
scp -r sunrise@<IP du rover>:~/rover /tmp/rover-robot    # garder la version du robot
diff -ru /tmp/rover-robot rover | less                   # reprendre ce que le robot a en plus
```

- Garder `rover_config.json` du robot (réglages calibrés, pas dans git) : ne pas l'écraser.
- Garder les options propres au robot (`--fps 40`, port série...) : elles sont dans le service,
  pas dans le code.
- **Jamais deux programmes sur le port série** (ouvrir le port redémarre l'ESP32 ;
  un portable d'équipier a déjà relancé le serveur deux fois). Le service de garde relance un
  serveur dès que le port est libre 5 s : pour lancer `rover_server.py` à la main, d'abord
  `sudo systemctl stop polaris-rover-api`, et `sudo systemctl start polaris-rover-api` après.
- Après copie : `sudo systemctl restart polaris-rover-api`, attendre `gyro_ready`, vérifier que
  `/state` contient `boot_id` et `stop_count` (puis : STOP sur la page, rover immobile ->
  `stop_count` augmente de 1). Au démarrage, un message `Settings file ... using the default`
  signifie qu'un réglage du robot sort des nouvelles bornes : vérifier les réglages dans la page.
- Puis les 6 essais roues en l'air de `rover/README.md` (section « Deploying to the robot »).
- À vérifier : que le `SIGTERM` de `systemctl` atteint bien `rover_server.py` à travers le script
  de garde (sinon le rover n'est pas arrêté à l'arrêt du service).

## 7. Écrit, testé sur la VM, mais jamais sur le vrai robot ni une vraie caméra

- Toute la chaîne `lancer_overhead.py` -> `rover_reel.py` -> `rover_server.py` en vrai : seulement
  contre un faux serveur et le vrai serveur avec un ESP32 simulé.
- La lecture du flux d'un téléphone (`camera_reader`) et le réglage `--latence` ; l'échelle au
  mètre ruban ; la détection du marqueur ArUco sur le vrai rover ; le rejet des mauvaises images
  sur l'éclairage de la salle.
- L'arrêt opérateur (STOP de la page pendant un pas ou entre deux pas avec `stop_count`,
  reprise en main), la détection d'un redémarrage du serveur, le pas borné (`Pas refuse`), la
  mesure du cap par une ligne droite, la reprise après une panne du serveur (`no ESP32
  feedback`...), la correction de parallaxe `--hauteur-marqueur`.
- Le serveur du rover de ce dépôt : surveillance de l'ESP32 (0,5 s), refus avant `gyro_ready`,
  validation des nombres et de `/config`, `boot_id`, `stop_count`, arrêt sur `SIGTERM`,
  `/snapshot.jpg` avec `X-Age-Ms`, le bandeau rouge des pannes de sa page (servi, jamais
  affiché dans un navigateur). Pas encore déployé sur le robot.
- La page `/mission` dans un navigateur, et sur une vraie mission.
- Le service de perception sur la caméra du vrai rover (évalué seulement en simulation) et son
  branchement `--perception`.
- La sortie du rover près d'un obstacle, la détection de non-progrès face au bruit réel, les
  seuils de rayon du rover (0,155 m) et de tolérance d'arrivée.
- La couche `devices/` (flotte d'appareils) : jamais essayée sur le robot ; le chemin testé
  pour la démo est `lancer_overhead.py`.

## 8. Comment ce guide a été vérifié (VM, 15 nov.)

`--help` de `lancer_overhead.py`, `measure_scale`, `segment_image`, `generate_markers`,
`rover_server.py`, `teleop/server.py`, `demo_mission.py`, `perception_service.py` ;
`lancer_overhead.py --dry-run` sur `data/overhead/synthetic_scene.png` (avec et sans
`--marqueur 10`, sans rover et avec `rover_server.py --sim`) ; refus de `--image` sans
`--dry-run` ; une image noire (rejet affiché) ; `measure_scale --points` et `--save` sur la même
photo ; `GET /state` de `rover_server.py --sim` ; `demo_mission.py` et `teleop/server.py --suivi`
(`/mission`, `/api/mission`).

Relu après la fusion de `fin/integration` (sécurité, banc de bout en bout, restes) : options et
messages cités vérifiés dans le code fusionné, `--hauteur-marqueur` essayé en `--dry-run` sur la
photo d'essai.
