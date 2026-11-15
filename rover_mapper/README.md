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
