# RoverMap — carte 3D en direct + vue 3ᵉ personne du RaspRover

Le Mac lit le flux caméra du rover, reconstruit l'environnement en 3D avec
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) et affiche
le robot qui s'y déplace, avec une caméra de suivi comme sur les voitures récentes.
Rien à installer sur le robot.

## Lancer

```bash
.venv/bin/python run.py --robot IP_DU_ROVER --cam-height 0.10
```

Puis ouvrir http://localhost:8080.

- `--cam-height` : hauteur de l'objectif au-dessus du sol, en mètres. **À mesurer** : elle fixe l'échelle de toute la carte.
- `--source URL` : si le flux n'est pas sur `http://IP:5000/video_feed` (adresse supposée d'après la doc Waveshare, non vérifiée sur le robot).
- `--model depth-anything/DA3-SMALL` : plus rapide, moins précis. `--res 504` : un peu plus précis, plus lent.
- `--source 0` (webcam du Mac), `--source 1` (iPhone via Continuity Camera, après autorisation caméra dans macOS) ou `--source video.mp4 --loop` pour tester sans le robot.

## Mesures (MacBook Pro M5, GPU, DA3-BASE, résolution 336)

Séquence TUM `freiburg1_desk` (caméra à la main, 9,4 m de trajet), une image sur 4 :

| Scénario | Erreur de trajectoire | Images clés | Images perdues |
|---|---|---|---|
| Passage simple | 3,8 cm | 80 | 0 |
| Aller-retour (retour dans la zone connue) | 3,6 cm au retour | +4 au retour | 0 |
| Saut brutal au point de départ, 6 tours | 3,7 cm à chaque tour | +2 au 2ᵉ tour, puis 0 | 1 par tour (puis retrouvé) |
| 40 images d'un autre lieu, puis retour au bureau | 3,7 cm au retour | 0 pendant l'autre lieu, +2 au retour | 40/40 dans l'autre lieu, 0 au retour |

- **Vitesse** : 0,155 s par image, soit environ 6 Hz. L'affichage est animé à 30 Hz entre deux poses.
- **Cohérence** : la profondeur de la carte et la trajectoire concordent à environ 5 % près.
- **Échelle absolue** : la carte calibrée par le sol est trop grande d'environ 14 % (11 à 22 % selon les réglages) sur cette séquence. Le modèle de profondeur métrique de DA3 se trompe dans l'autre sens (environ 8 % trop petit). Aucune des deux n'est une référence fiable ; sur le rover, préférer la distance parcourue par les roues ou une caméra stéréo.
- **Capteur de profondeur (optionnel, non branché pour l'instant)** : `Mapper.step(rgb, sensor_depth)` accepte une profondeur en mètres alignée sur l'image (ex. RealSense D405), et l'échelle vient alors du capteur au lieu du sol. Testé avec la Kinect de TUM coupée à 0,8 m pour imiter un capteur à courte portée : profondeur juste à 5 % près au lieu d'environ 14 % d'erreur, trajectoire environ 10 % trop courte, précision inchangée (3,7 cm). Le reste vient d'une dérive d'échelle d'environ 5 % le long du trajet. Ce n'est pas un essai avec une vraie D405.
- **Affinage en tâche de fond** (`Mapper.refine_keyframe`, chaque image clé recalculée avec 4 voisines en 504) : positions des images clés 3,8 → 3,0 cm, mais profondeur inchangée (7 % d'erreur). Gain jugé trop faible pour l'activer ; la fonction reste disponible.

Les mesures sont faites avec `scripts/eval_tum.py` (`mode=once|fwdrev|loop|gap`) et `scripts/eval_refine.py`.

## Essais en direct (iPhone, Continuity Camera)

Lanceur : `scripts/live_iphone.command`, à ouvrir avec l'app Terminal de macOS, qui demandera l'accès caméra. Chaque session est enregistrée dans `out/rec_<date>.avi`, avec un journal par image dans `out/live_log.csv`. Pour rejouer une session à l'identique : `python scripts/replay.py out/rec_….avi [début_s] [fin_s]`.

- **Personnes et objets mobiles** : un petit modèle de segmentation (DeepLabV3-MobileNetV3, 11 ms par image sur le GPU du Mac) repère personnes, animaux et véhicules. Ces pixels sont exclus de la carte et du suivi. Sur un passage avec des gens qui passent : pertes de suivi 29 % → 9 %. Option `--no-mask` pour le désactiver ; il l'est aussi d'office tant que les poids du modèle ne sont pas téléchargés.
- **Rendu** : la carte s'affiche en « Splats (gaussiennes) », un disque par point orienté selon la surface, ou en « Points ». Le menu « Rendu » de la vue permet de passer de l'un à l'autre, et `--render` choisit le rendu au démarrage.
- **Bouton « Reconnecter la caméra »** : rouvre la caméra et démarre un nouvel enregistrement. La reconnexion se fait aussi d'elle-même si un flux en direct s'arrête.
- **Images ignorées** : les images noires (objectif masqué, Continuity Camera en pause) et sans détail (floues, surexposées) ne sont pas utilisées et ne deviennent jamais des images clés.
- **Références** : seules comptent celles que l'image actuelle recouvre vraiment. Une mauvaise image clé ne fait donc plus échouer tout le suivi.
- **Mesures** : sur un passage au ras du sol dans une salle (point de vue du rover), le suivi est perdu environ 15 à 25 % du temps. Ce sont surtout des pertes brèves (0,1 à 0,5 s), causées par le flou de bougé, des personnes qui passent ou des vues très rapprochées. Filmer un bureau avec ses mains qui bougent dans le champ donne nettement plus de pertes.

## Limites connues

- Pas de fermeture de boucle globale : la carte peut dériver sur un très long trajet. Le suivi réutilise toutefois les images clés les plus proches, ce qui limite la dérive quand le robot repasse au même endroit.
- Quand le suivi est perdu, la position est gelée (indicateur rouge) et le programme cherche un endroit connu ressemblant à l'image actuelle. Si le robot a continué dans une zone jamais cartographiée, il reste figé : il faut revenir dans une zone connue ou réinitialiser la carte. Une reprise automatique existe (`lost_patience`) mais est désactivée, car en test elle a inséré une image clé venant d'un lieu inconnu.
- En direct, avec la vidéo de test en boucle, le nombre d'images clés croît encore lentement (moins d'une par tour, contre 0 avec des images fixes).
- Échelle absolue juste à environ ±15 % seulement sur la séquence testée (voir ci-dessus).

## Fichiers

- `rovermap/source.py` : lecture du flux (MJPEG, webcam ou vidéo), en ne gardant que l'image la plus récente.
- `rovermap/mapper.py` : suivi, images clés, détection du sol et nuage de points.
- `rovermap/viewer.py` : scène 3D (viser), modèle du rover et caméra de suivi.
- `scripts/fake_rover.py` : simule le flux du rover à partir d'une vidéo.
- `scripts/eval_tum.py` : mesure de précision sur un jeu de données TUM (`0` comme pas d'images = rythme réel).

## Vraie D405 du rover (`--d405`)

Le robot ne fait que capturer et envoyer (`d405_server.py` sur l'ordinateur du rover, 127.0.0.1:8091, joint depuis la VM par un tunnel SSH) : une image couleur JPEG et la profondeur brute 16 bits dans le repère du capteur de profondeur, non alignée, avec intrinsèques et extrinsèques d'usine, seulement quand on la demande. Tout le calcul se fait sur la VM (`rovermap/source.py`, `D405Source`) : profondeur en mètres (brut × `depth_scale` lu sur l'appareil, l'unité de la D405 n'est pas le mm), alignement profondeur → couleur pour « D405 seule », reprojection profondeur → caméra du robot pour « Les deux ».

```bash
# D405 seule (échelle donnée par le capteur)
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/run.py --d405 http://127.0.0.1:8091 \
    --sensors d405 --d405-height 0.06
# Les deux : image du robot (redressée par le tableau de bord) + profondeur D405
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/run.py --source http://127.0.0.1:8082/video_rect \
    --rgb-fov 100 --d405 http://127.0.0.1:8091 --sensors both --d405-mount TX TY TZ TANGAGE \
    --cam-height 0.21 [--rover http://127.0.0.1:18080]
```

- `--d405-height` : hauteur de l'objectif de la D405 au-dessus du sol (à mesurer).
- `--d405-mount TX TY TZ [TANGAGE [LACET [ROULIS]]]` : caméra couleur de la D405 vue de la caméra du robot, axes OpenCV (x à droite, y vers le bas, z devant, mètres) ; tangage > 0 = la D405 regarde plus bas, lacet > 0 = plus à gauche, roulis > 0 = son côté droit plus bas. **À mesurer.**
- `--rgb-fov` : champ horizontal de `--source`, une image sténopé centrée (le `--rect-fov` du tableau de bord, 100 par défaut). Pas `--robot` : son flux brut est en fisheye.
- `--d405-max-age 1.0` : une image D405 plus vieille n'est pas utilisée (âge mesuré par l'horloge monotone du robot plus la nôtre ; l'horloge murale du robot a des heures de décalage).
- **« Les deux » : les deux images ne sont pas prises ensemble.** La caméra du robot arrive par le tableau de bord avec ~0,5 s de retard, la D405 est la plus récente : en tournant ou en roulant vite, la profondeur tombe sur les mauvais pixels. Rouler lentement.
- **Tourelle** : la caméra du robot est sur une tourelle ; `--d405-mount` ne vaut qu'à pan 0 et au tilt de la mesure. Avec `--rover URL` (serveur du rover, `GET /state`), un pan non nul est signalé.

Sans caméra branchée, `d405_server.py` la recherche après 2 s, puis 4, 8 et 15 s au plus (chaque essai énumère l'USB : un essai toutes les 2 s prenait 23 % d'un cœur du robot) ; dès que des images arrivent, l'attente revient à 2 s. L'attente en cours est dans `/health` (`retry_s`).

Sans le robot : `scripts/fake_d405.py` sert le même contrat HTTP à partir du simulateur de l'appartement (`--sim http://127.0.0.1:8090`, 2 images/s au plus ; jamais celui de la forêt sur 5050). Tests : `test_d405_source.py` (faux serveur D405 avec extrinsèques non triviales, sans appareil ni modèle).

## Perception du rover pour la mission (une seule image, sans suivi)

`perception_service.py` donne à `merging_map` (interface 4 : `merging_map/perception_rover.py`, `percevoir_http(url)`) les cases de 5 cm devant le rover : 1 obstacle, 0 sol vu, -1 inconnu. À chaque demande : dernière image de la caméra → DA3 sur cette seule image → plan du sol (RANSAC, moitié basse) → échelle donnée par `--cam-height` → repère rover (x devant = axe optique projeté sur le sol, y à gauche, z depuis le sol) → obstacle = points entre `--band-low` 4 cm et `--band-high` 40 cm au-dessus du sol ; ce qui est derrière un obstacle ou hors du champ reste -1 (un 0 effacerait les obstacles du drone). Pas de suivi : celui du mappeur n'a pas tenu sur le vrai rover.

```bash
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/perception_service.py \
    --source http://127.0.0.1:8082/video_rect --cam-height 0.22 --fov 100 --port 8095
# GET /obstacles (JSON, 503 sans image de moins de 2 s ; ?fresh=1 attend une image arrivée
# après la demande), GET /debug.jpg (image avec cases rouges/vertes), GET /health
```

`--fov 100` : `/video_rect` est par construction une image sténopé centrée de 100° (option `--rect-fov` du dashboard) ; sans `--fov`/`--k`, les intrinsèques estimées par DA3 sont utilisées. Liaison 127.0.0.1 seulement.

**Vrai rover sur une liaison lente (commande recommandée).** Par le tunnel SSH, chaque requête au robot coûte 1,3 à 4 s quelle que soit sa taille : le service lit directement l'image brute du robot, une requête par regard, sans passer par le tableau de bord (qui tire le flux en continu) :

```bash
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/perception_service.py \
    --source http://127.0.0.1:18080/snapshot.jpg --calib teleop/calib/plumbline.json \
    --rect-fov 100 --fov 100 --cam-height 0.22 --range 1.5 --port 8095
```

- `--calib FICHIER` : modèle de l'objectif fisheye (`teleop/calib/plumbline.json`). La source donne alors des images BRUTES (le `/snapshot.jpg` du robot, fisheye 640x480) et le service redresse chacune lui-même, avec la même K et les mêmes tables `cv2.fisheye.initUndistortRectifyMap` que `CameraHub` de `teleop/server.py` (le test compare octet par octet avec `/video_rect`). Une image d'une autre taille que celle du modèle est refusée.
- `--rect-fov DEG` (100 par défaut) : champ horizontal de l'image redressée ; avec `--calib` c'est aussi le champ connu (`--fov`, s'il est donné, doit être égal ; `--k` est refusé).
- `--fetch-timeout 10` : délai d'un GET de l'image (la liaison prend 1,3 à 4 s ; l'ancien délai de 3 s prenait une réponse lente pour une panne).
- Une URL qui finit par `.jpg`, `.jpeg`, `.png` ou `.npz` est une image unique : pas de requête de sondage au démarrage (sur liaison lente, ce sondage faisait répondre 503 à la première demande).
- `GET /obstacles?fresh=1` : avec une telle source, UN nouveau GET de l'image par demande (frais par construction) ; sa durée est dans `fetch_s` (le coût de la liaison, visible par l'appelant). Sans `fresh=1`, une image de moins de `--max-age` s est réutilisée sans requête, et `GET /debug.jpg` juste après montre la même image et le même résultat, sans nouvelle requête.
- `POST /obstacles` (corps : un JPEG, brut avec `--calib`, déjà redressé sinon) : pour un pilote qui a déjà téléchargé l'image ; même JSON que `GET /obstacles`, avec `age_s` 0, `t` = heure de réception et `fetch_s` null. Le service n'ouvre alors aucune connexion vers le robot. Image illisible ou de mauvaise taille : 400.
- Côté appelant : un délai d'au moins `--fetch-timeout` + l'inférence (~0,6 s) pour `?fresh=1` ; le panoramique de la tourelle n'est pas pris en compte ici (l'appelant tourne la vue, `cap - pan`).

**Mesures en simulation** (`scripts/eval_perception.py`, 30 images de la forêt enregistrée + 30 de l'appartement, même classification appliquée à la profondeur exacte, portée 2 m, cases à 1 case près ; inférence 0,08-0,1 s sur la L4 partagée, 0,57 s par requête dans le service en direct) :

| | appartement, K connue | appartement, K de DA3 | forêt, K connue | forêt, K de DA3 |
|---|---|---|---|---|
| échelle (profondeur / vérité, 0-1 m) | 0,98 | 1,19 | 0,91 | 1,17 |
| obstacles 0,5-1 m : précision / rappel | 0,71 / 0,77 | 0,67 / 0,61 | 0,35 / 0,59 | 0,47 / 0,80 |
| obstacles 1-1,5 m : précision / rappel | 0,49 / 0,60 | 0,45 / 0,36 | 0,22 / 0,36 | 0,28 / 0,37 |
| cases « libres » qui sont des obstacles, 0,5-1 m / 1-1,5 m | 0,2 % / 1,9 % | 1,4 % / 3,8 % | 0,5 % / 1,8 % | 0,5 % / 1,7 % |

À la case exacte, la précision tombe à 0,46 (appartement, 0,5-1 m, K connue) : les positions sont justes à ~5 cm près, pas mieux. Au-delà de 1-1,5 m le rappel s'effondre : portée par défaut 1,5 m. En intérieur, la K connue est meilleure (échelle et obstacles) ; en forêt, DA3 avec sa propre K l'est. **Jamais essayé avec la caméra du vrai rover.** À vérifier d'abord : la hauteur réelle de l'objectif (elle fixe l'échelle), la tourelle de niveau (le plan du sol absorbe une inclinaison, mais l'axe x suit l'axe optique : un panoramique non nul tourne toute la vue), `/debug.jpg` devant une boîte posée à 0,5 m et 1 m, et si l'avant du rover est dans l'image (`--min-range`).
