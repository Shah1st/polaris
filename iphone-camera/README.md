# CamStream

Une app iPhone qui rend la caméra disponible sur votre réseau Wi-Fi local.
Ouvrez l'app sur l'iPhone, puis tapez l'adresse affichée (ex. `http://192.168.1.23:8080`) dans un navigateur du Mac : vous voyez la caméra en direct.

Le flux est en **MJPEG sur HTTP**, un format universel : Safari, Chrome, VLC, OBS, ffmpeg et OpenCV le lisent sans rien installer.

## Installer sur l'iPhone

1. Ouvrez `CamStream.xcodeproj` dans Xcode.
2. Cible **CamStream** › onglet **Signing & Capabilities** › choisissez votre **Team** (un compte Apple gratuit suffit).
   Si Xcode signale que l'identifiant est déjà pris, remplacez `com.example.CamStream` par un identifiant à vous (ex. `com.votrenom.CamStream`).
3. Branchez l'iPhone (ou sélectionnez-le en Wi-Fi), choisissez-le comme destination, puis lancez avec **⌘R**.
4. La première fois, sur l'iPhone :
   - activez le **Mode développeur** (Réglages › Confidentialité et sécurité) si iOS le demande ;
   - avec un compte gratuit, faites confiance au développeur dans Réglages › Général › VPN et gestion de l'appareil ;
   - acceptez l'accès à la **caméra** et au **réseau local**. Sans le réseau local, le Mac ne pourra pas se connecter.

> Avec un compte Apple gratuit, l'app installée expire au bout de 7 jours : il suffit de la relancer depuis Xcode.

## Utilisation

L'app démarre la diffusion automatiquement et affiche l'adresse à ouvrir. L'iPhone et le Mac doivent être sur le **même réseau Wi-Fi** (ou le Mac connecté au partage de connexion de l'iPhone).

| Adresse | Contenu |
|---|---|
| `http://IP:8080/` | Page web : flux, nombre de spectateurs, capture, plein écran |
| `http://IP:8080/stream` | Flux MJPEG brut (VLC, OBS, ffmpeg, OpenCV…) |
| `http://IP:8080/view` | Page avec uniquement l'image (idéale pour OBS) |
| `http://IP:8080/snapshot.jpg` | Dernière image en JPEG |
| `http://IP:8080/status` | État en JSON (résolution, i/s, caméra, objectif, LiDAR, stabilisation, spectateurs) |
| `http://IP:8080/depth` | Profondeur LiDAR brute (voir « LiDAR » plus bas) |
| `http://IP:8080/depth.jpg` | Aperçu de la profondeur : proche = blanc, 5 m et plus = noir |

Le service est aussi annoncé en Bonjour sous le nom « CamStream » (`dns-sd -B _http._tcp` sur le Mac).

Dans l'app :
- **🔄** change de caméra (arrière / avant) ;
- **1× / 0,5×** passe la caméra arrière en **grand angle** (objectif ultra grand-angle, champ environ deux fois plus large) et retour. Le bouton n'apparaît que sur un iPhone qui a cet objectif (iPhone 11 et suivants, pas le SE) ; la caméra avant reste en 1× ;
- **Réglages** : caméra et objectif, **profondeur LiDAR**, **stabilisation** (Aucune / Standard / Cinéma / Maximum), résolution (480p / 720p / 1080p), images par seconde (15 / 24 / 30), qualité JPEG, liste des adresses à copier ;
- **🌙** éteint l'affichage (écran noir) pendant que la diffusion continue : moins de batterie et moins de chauffe. Touchez l'écran pour le rallumer.
- L'interface reste en portrait. **Le flux suit l'orientation réelle de l'iPhone** : tenez-le à l'horizontale pour une image 16:9.

## Utiliser le flux sur le Mac

**Navigateur** : ouvrez `http://IP:8080`.

**VLC** : Fichier › Ouvrir un réseau… › `http://IP:8080/stream`.

**OBS** : ajoutez une source **Navigateur** avec l'URL `http://IP:8080/view` et la taille du flux (ex. 1280×720).

**Comme webcam dans Zoom, Meet, Teams ou FaceTime** : dans OBS, cliquez sur **Démarrer la caméra virtuelle**, puis choisissez « OBS Virtual Camera » comme caméra dans l'application de visio.

**ffmpeg** (enregistrer 30 s en MP4) :
```bash
ffmpeg -i http://IP:8080/stream -t 30 -c:v libx264 -pix_fmt yuv420p sortie.mp4
```

**Python / OpenCV** :
```python
import cv2
cap = cv2.VideoCapture("http://IP:8080/stream")
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("iPhone", frame)
    if cv2.waitKey(1) == 27: break
```

## Bon à savoir

- **L'app doit rester au premier plan.** iOS coupe la caméra dès qu'une app passe en arrière-plan. L'écran ne se verrouille pas tout seul pendant la diffusion. Au retour dans l'app, le flux reprend et la page web se reconnecte seule.
- **Latence** : si le réseau ralentit, les images en retard sont sautées plutôt que mises en file d'attente. Le décalage ne s'accumule donc pas : on voit toujours l'image la plus récente.
- **Sans spectateur**, l'app n'encode qu'une image par seconde, pour garder `/snapshot.jpg` à jour sans user la batterie.
- **Sécurité** : tout appareil du réseau local peut voir le flux, sans mot de passe. À éviter sur un Wi-Fi public.
- **Alternative native** : pour utiliser l'iPhone uniquement comme webcam du Mac, la fonction *Caméra de continuité* d'Apple le fait sans app. CamStream est utile pour tout le reste : autre ordinateur, navigateur, OBS, scripts, plusieurs spectateurs…

## Grand angle et caméra du dessus (Polaris)

Le grand angle voit environ deux fois plus large à la même hauteur : utile pour couvrir toute l'arène depuis un point bas. Le choix est mémorisé, et l'objectif utilisé est donné par `/status` (`"lens": "wide"`, `"ultrawide"`, ou `"none"` pour la mire).

- **Un seul objectif, toujours le même.** L'app choisit la caméra physique (1× ou 0,5×), jamais la caméra « virtuelle » d'iOS qui change d'objectif toute seule selon la distance et la lumière : l'échelle de l'image ne change donc pas en pleine mission.
- **Distorsion** : iOS corrige par défaut celle du grand angle (`isGeometricDistortionCorrectionEnabled` vaut `true` quand l'appareil le permet, d'après la documentation d'Apple). La chaîne de la caméra du dessus (`docs/overhead-camera.md`, `docs/demo-reelle.md`) ne corrige aucune distorsion et recommandait jusqu'ici l'objectif principal seul : vérifier sur une image du flux que les bords droits de l'arène restent droits jusque dans les coins avant de s'y fier.
- **Échelle** : remesurer les mètres par pixel (`python -m guidance_global_map.measure_scale`) après chaque changement d'objectif, de résolution ou de hauteur ; l'ancienne valeur est fausse d'un facteur 2 environ.
- Le changement d'objectif se fait sur l'iPhone seulement, pas par le réseau : le serveur n'a pas de mot de passe, et n'importe qui sur le Wi-Fi pourrait sinon changer l'échelle pendant une mission.

## LiDAR

Sur un iPhone Pro (depuis l'iPhone 12 Pro), **Réglages › LiDAR › Profondeur LiDAR** capture la distance de chaque point en plus de l'image. L'app utilise alors la caméra LiDAR d'iOS : l'objectif principal 1× et le scanner LiDAR ensemble ; le grand angle n'est pas disponible dans ce mode. Portée d'environ 5 m. Le réglage est mémorisé ; `/status` dit `"depth": "lidar"` quand la profondeur est capturée (`"off"` : coupée, `"unavailable"` : pas de LiDAR).

`GET /depth` renvoie la dernière carte de profondeur (au plus 1/5 s d'âge), brute :

- le corps : des `float32` little-endian, en **mètres** le long de l'axe optique, ligne après ligne depuis le coin haut gauche ; 0 ou NaN = pas de mesure ;
- les en-têtes : `X-Depth-Width`, `X-Depth-Height`, `X-Depth-Intrinsics` (`fx,fy,cx,cy` en pixels de la carte de profondeur, quand iOS les donne), `X-Video-Rotation` (rotation appliquée à la vidéo, en degrés), `X-Age-Ms`.

La carte de profondeur est dans l'orientation propre du capteur (paysage) et n'est **pas** tournée comme la vidéo : pour les superposer, la tourner de `X-Video-Rotation`. Elle est plus petite que l'image vidéo et filtrée par iOS (trous bouchés, bords lissés).

```python
import numpy as np, requests
r = requests.get("http://IP:8080/depth", timeout=3)
w, h = int(r.headers["X-Depth-Width"]), int(r.headers["X-Depth-Height"])
profondeur = np.frombuffer(r.content, "<f4").reshape(h, w)   # mètres ; 0 / NaN = pas de mesure
fx, fy, cx, cy = map(float, r.headers["X-Depth-Intrinsics"].split(","))
```

Pour la caméra du dessus (Polaris), iPhone à plat au-dessus de l'arène : le sol est à la distance la plus fréquente, ce qui donne la hauteur de la caméra et donc l'échelle (mètres par pixel) sans mètre ruban ; un obstacle est ce qui dépasse du sol de quelques centimètres, quelle que soit sa couleur (ni les ombres ni les traits au sol ne comptent). Ce traitement n'est pas encore écrit côté Python.

Pas encore essayé sur un vrai iPhone : le code compile (Mac de GitHub, voir `.github/workflows/camstream-ipa.yml`), mais la capture LiDAR n'a jamais tourné.

## Stabilisation

**Réglages › Stabilisation** lisse les tremblements d'un iPhone tenu à la main. Par défaut : **Aucune** (comme avant, l'image n'est pas stabilisée). Le choix est mémorisé, et `/status` donne le mode réellement utilisé (`"stabilization"`).

| Réglage | Mode iOS demandé |
|---|---|
| Aucune | aucun (`off`) |
| Standard | `standard` |
| Cinéma | `cinematic`, sinon `standard` |
| Maximum | le plus fort que l'iPhone accepte : `cinematic-extended-enhanced` (iOS 18), sinon `cinematic-extended`, `cinematic`, `standard` |

Chaque caméra et chaque résolution n'acceptent pas tous les modes : l'app prend le premier de la liste que le format actif accepte, et `off` si aucun. iOS stabilise en recadrant et en décalant l'image :

- **le champ de vue rétrécit** : pour la caméra du dessus (Polaris), l'échelle (mètres par pixel) change, à remesurer ;
- **l'image se déforme d'une image à l'autre** : gênant pour une carte, où la géométrie doit rester fixe ;
- **Cinéma et Maximum retardent le flux** d'une fraction de seconde (iOS garde des images d'avance pour lisser).

iPhone fixe au-dessus de la zone : laisser sur **Aucune**. iPhone tenu à la main : **Standard** est un bon compromis. Avec le **LiDAR**, la stabilisation est coupée : la profondeur doit rester alignée sur l'image.

Pas encore essayé sur un vrai iPhone : le code compile sur le Mac de GitHub (voir `.github/workflows/camstream-ipa.yml`).

## Tester sans iPhone

Le simulateur iOS n'a pas de caméra : l'app y diffuse une **mire de test animée** sur le même serveur. Lancez l'app dans le simulateur, puis ouvrez `http://localhost:8080` sur le Mac.

## Structure du code

| Fichier | Rôle |
|---|---|
| `CameraManager.swift` | Capture AVFoundation, orientation, encodage JPEG (Core Image), cadence |
| `MJPEGServer.swift` | Serveur HTTP (Network.framework), flux MJPEG, Bonjour, comptage des spectateurs |
| `StreamModel.swift` | État de l'app, réglages mémorisés, liaison caméra ↔ serveur |
| `ContentView.swift` / `SettingsView.swift` | Interface SwiftUI |
| `WebPage.swift` | Pages HTML servies au navigateur |
| `NetworkInfo.swift` | Adresses IP locales de l'iPhone |
| `TestPattern.swift` | Mire de test pour le simulateur |

Prérequis : Xcode 16 ou plus récent, iOS 17 ou plus récent. Swift 6, aucune dépendance externe.
