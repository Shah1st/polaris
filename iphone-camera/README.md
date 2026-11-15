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
| `http://IP:8080/status` | État en JSON (résolution, i/s, caméra, spectateurs) |

Le service est aussi annoncé en Bonjour sous le nom « CamStream » (`dns-sd -B _http._tcp` sur le Mac).

Dans l'app :
- **🔄** change de caméra (arrière / avant) ;
- **Réglages** : résolution (480p / 720p / 1080p), images par seconde (15 / 24 / 30), qualité JPEG, liste des adresses à copier ;
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
