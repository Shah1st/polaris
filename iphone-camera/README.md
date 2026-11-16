# CamStream

An iPhone app that makes the camera available on your local Wi-Fi network.
Open the app on the iPhone, then type the address shown (e.g. `http://192.168.1.23:8080`) into a browser on the Mac: you see the camera live.

The stream is **MJPEG over HTTP**, a universal format: Safari, Chrome, VLC, OBS, ffmpeg and OpenCV read it with nothing to install.

## Install on the iPhone

1. Open `CamStream.xcodeproj` in Xcode.
2. **CamStream** target › **Signing & Capabilities** tab › choose your **Team** (a free Apple account is enough).
   If Xcode reports that the identifier is already taken, replace `com.example.CamStream` with an identifier of your own (e.g. `com.votrenom.CamStream`).
3. Plug in the iPhone (or select it over Wi-Fi), choose it as the destination, then run with **⌘R**.
4. The first time, on the iPhone:
   - turn on **Developer Mode** (Settings › Privacy & Security) if iOS asks for it;
   - with a free account, trust the developer in Settings › General › VPN & Device Management;
   - allow access to the **camera** and to the **local network**. Without the local network, the Mac will not be able to connect.

> With a free Apple account, the installed app expires after 7 days: just run it again from Xcode.

## Usage

The app starts streaming automatically and shows the address to open. The iPhone and the Mac must be on the **same Wi-Fi network** (or the Mac connected to the iPhone's Personal Hotspot).

| Address | Content |
|---|---|
| `http://IP:8080/` | Web page: stream, number of viewers, snapshot, full screen |
| `http://IP:8080/stream` | Raw MJPEG stream (VLC, OBS, ffmpeg, OpenCV…) |
| `http://IP:8080/view` | Page with the image only (ideal for OBS) |
| `http://IP:8080/snapshot.jpg` | Latest image as JPEG |
| `http://IP:8080/status` | State as JSON (resolution, fps, camera, lens, LiDAR, stabilisation, viewers) |
| `http://IP:8080/depth` | Raw LiDAR depth (see "LiDAR" below) |
| `http://IP:8080/depth.jpg` | Depth preview: near = white, 5 m and beyond = black |

The service is also advertised over Bonjour under the name "CamStream" (`dns-sd -B _http._tcp` on the Mac).

In the app:
- **🔄** switches camera (back / front);
- **1× / 0,5×** switches the back camera to **wide angle** (ultra wide lens, field of view about twice as wide) and back. The button only appears on an iPhone that has this lens (iPhone 11 and later, not the SE); the front camera stays at 1×;
- **Réglages** (Settings): camera and lens, **LiDAR depth**, **stabilisation** (Aucune (None) / Standard / Cinéma (Cinematic) / Maximum), resolution (480p / 720p / 1080p), frames per second (15 / 24 / 30), JPEG quality, list of addresses to copy;
- **🌙** turns the display off (black screen) while streaming continues: less battery use and less heat. Touch the screen to turn it back on.
- The interface stays in portrait. **The stream follows the real orientation of the iPhone**: hold it horizontally for a 16:9 image.

## Using the stream on the Mac

**Browser**: open `http://IP:8080`.

**VLC**: File › Open Network… › `http://IP:8080/stream`.

**OBS**: add a **Browser** source with the URL `http://IP:8080/view` and the size of the stream (e.g. 1280×720).

**As a webcam in Zoom, Meet, Teams or FaceTime**: in OBS, click **Start Virtual Camera**, then choose "OBS Virtual Camera" as the camera in the video call application.

**ffmpeg** (record 30 s as MP4):
```bash
ffmpeg -i http://IP:8080/stream -t 30 -c:v libx264 -pix_fmt yuv420p sortie.mp4
```

**Python / OpenCV**:
```python
import cv2
cap = cv2.VideoCapture("http://IP:8080/stream")
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("iPhone", frame)
    if cv2.waitKey(1) == 27: break
```

## Good to know

- **The app must stay in the foreground.** iOS cuts the camera as soon as an app goes to the background. The screen does not lock by itself while streaming. When you return to the app, the stream resumes and the web page reconnects by itself.
- **Latency**: if the network slows down, late images are skipped rather than queued. So the delay does not build up: you always see the most recent image.
- **With no viewer**, the app encodes only one image per second, to keep `/snapshot.jpg` up to date without draining the battery.
- **Security**: any device on the local network can see the stream, with no password. Avoid it on public Wi-Fi.
- **Native alternative**: to use the iPhone only as the Mac's webcam, Apple's *Continuity Camera* feature does it without an app. CamStream is useful for everything else: another computer, browser, OBS, scripts, several viewers…

## Wide angle and overhead camera (Polaris)

The wide angle sees about twice as wide at the same height: useful to cover the whole arena from a low point. The choice is remembered, and the lens in use is given by `/status` (`"lens": "wide"`, `"ultrawide"`, or `"none"` for the test pattern).

- **One lens only, always the same.** The app chooses the physical camera (1× or 0.5×), never the "virtual" iOS camera that changes lens by itself depending on distance and light: so the scale of the image does not change in the middle of a mission.
- **Distortion**: iOS corrects that of the wide angle by default (`isGeometricDistortionCorrectionEnabled` is `true` when the device allows it, according to Apple's documentation). The overhead camera chain (`docs/overhead-camera.md`, `docs/demo-reelle.md`) corrects no distortion and until now recommended the main lens only: check on an image from the stream that the straight edges of the arena stay straight right into the corners before relying on it.
- **Scale**: measure the metres per pixel again (`python -m guidance_global_map.measure_scale`) after every change of lens, resolution or height; the old value is wrong by a factor of about 2.
- The lens is changed on the iPhone only, not over the network: the server has no password, and otherwise anyone on the Wi-Fi could change the scale during a mission.

## LiDAR

On an iPhone Pro (from the iPhone 12 Pro), **Réglages › LiDAR › Profondeur LiDAR** (Settings › LiDAR › LiDAR depth) captures the distance of each point in addition to the image. The app then uses the iOS LiDAR camera: the main 1× lens and the LiDAR scanner together; the wide angle is not available in this mode. Range of about 5 m. The setting is remembered; `/status` says `"depth": "lidar"` when depth is captured (`"off"`: turned off, `"unavailable"`: no LiDAR).

`GET /depth` returns the latest depth map (at most 1/5 s old), raw:

- the body: little-endian `float32` values, in **metres** along the optical axis, row by row from the top left corner; 0 or NaN = no measurement;
- the headers: `X-Depth-Width`, `X-Depth-Height`, `X-Depth-Intrinsics` (`fx,fy,cx,cy` in pixels of the depth map, when iOS gives them), `X-Video-Rotation` (rotation applied to the video, in degrees), `X-Age-Ms`.

The depth map is in the sensor's own orientation (landscape) and is **not** rotated like the video: to overlay them, rotate it by `X-Video-Rotation`. It is smaller than the video image and filtered by iOS (holes filled, edges smoothed).

```python
import numpy as np, requests
r = requests.get("http://IP:8080/depth", timeout=3)
w, h = int(r.headers["X-Depth-Width"]), int(r.headers["X-Depth-Height"])
profondeur = np.frombuffer(r.content, "<f4").reshape(h, w)   # metres; 0 / NaN = no measurement
fx, fy, cx, cy = map(float, r.headers["X-Depth-Intrinsics"].split(","))
```

For the overhead camera (Polaris), iPhone flat above the arena: the floor is at the most frequent distance, which gives the height of the camera and therefore the scale (metres per pixel) without a tape measure; an obstacle is whatever rises above the floor by a few centimetres, whatever its colour (neither shadows nor lines on the floor count). This processing is not yet written on the Python side.

Not yet tried on a real iPhone: the code compiles (GitHub's Mac, see `.github/workflows/camstream-ipa.yml`), but the LiDAR capture has never run.

## Stabilisation

**Réglages › Stabilisation** (Settings › Stabilisation) smooths the shaking of a hand-held iPhone. Default: **Aucune** (None; as before, the image is not stabilised). The choice is remembered, and `/status` gives the mode actually used (`"stabilization"`).

| Setting | iOS mode requested |
|---|---|
| Aucune (None) | none (`off`) |
| Standard | `standard` |
| Cinéma (Cinematic) | `cinematic`, otherwise `standard` |
| Maximum | the strongest the iPhone accepts: `cinematic-extended-enhanced` (iOS 18), otherwise `cinematic-extended`, `cinematic`, `standard` |

Not every camera and resolution accepts every mode: the app takes the first one in the list that the active format accepts, and `off` if none. iOS stabilises by cropping and shifting the image:

- **the field of view shrinks**: for the overhead camera (Polaris), the scale (metres per pixel) changes, to be measured again;
- **the image warps from one image to the next**: awkward for a map, where the geometry must stay fixed;
- **Cinéma (Cinematic) and Maximum delay the stream** by a fraction of a second (iOS keeps images ahead to smooth).

iPhone fixed above the area: leave on **Aucune** (None). Hand-held iPhone: **Standard** is a good compromise. With the **LiDAR**, stabilisation is turned off: the depth must stay aligned with the image.

Not yet tried on a real iPhone: the code compiles on GitHub's Mac (see `.github/workflows/camstream-ipa.yml`).

## Testing without an iPhone

The iOS simulator has no camera: there the app streams an **animated test pattern** on the same server. Run the app in the simulator, then open `http://localhost:8080` on the Mac.

## Code structure

| File | Role |
|---|---|
| `CameraManager.swift` | AVFoundation capture, orientation, JPEG encoding (Core Image), frame rate |
| `MJPEGServer.swift` | HTTP server (Network.framework), MJPEG stream, Bonjour, viewer counting |
| `StreamModel.swift` | App state, remembered settings, camera ↔ server link |
| `ContentView.swift` / `SettingsView.swift` | SwiftUI interface |
| `WebPage.swift` | HTML pages served to the browser |
| `NetworkInfo.swift` | Local IP addresses of the iPhone |
| `TestPattern.swift` | Test pattern for the simulator |

Requirements: Xcode 16 or later, iOS 17 or later. Swift 6, no external dependencies.
