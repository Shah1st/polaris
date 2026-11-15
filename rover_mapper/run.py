"""Live 3D map + third-person view from the rover camera.

    python run.py --robot 192.168.1.42          # RaspRover (MJPEG on :5000/video_feed)
    python run.py --source http://host/stream   # any MJPEG URL
    python run.py --source 0                    # this computer's webcam
    python run.py --source video.mp4            # recorded video
    python run.py --sim http://127.0.0.1:5060 --cam-height 0.212
                                                # simulator (sim/run_world.py): menu
                                                # "Capteurs" (caméra du robot / D405
                                                # simulée / les deux) + mesures vs
                                                # vérité terrain
    python run.py --d405 http://127.0.0.1:8091 --sensors d405 --d405-height 0.06
                                                # vraie D405 du rover (d405_server.py par
                                                # tunnel SSH), tout le calcul ici
    python run.py --source http://127.0.0.1:8082/video_rect --rgb-fov 100 \\
        --d405 http://127.0.0.1:8091 --sensors both --d405-mount 0 0.15 0.08 10
                                                # caméra du robot + profondeur D405

Then open http://localhost:8080
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import warnings

warnings.filterwarnings("ignore")
# Port scanners / health probes hitting the viewer port are not worth a traceback.
logging.getLogger("websockets").setLevel(logging.CRITICAL)


# --outdoor: chosen on two recorded forest runs (scripts/vm/eval_forest.py,
# HANDOFF_VM.md). Beyond ~10 m DA3 squeezes depth to ~0.5-0.7 of the truth.
OUTDOOR = {"max_depth": 10.0, "voxel": 0.05, "kf_min_dist": 0.15, "mask_sky": True,
           "sensor_anchor": True, "max_jump": 0.3, "lost_extrapolate": True}


def is_blank(rgb) -> bool:
    """True for black / covered-lens frames (measured on an iPhone session:
    black frames have 99th-percentile brightness < 4, real ones > 100)."""
    import cv2
    import numpy as np
    g = cv2.cvtColor(cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return float(np.percentile(g, 99)) < 20.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", help="IP du RaspRover (utilise http://IP:5000/video_feed)")
    ap.add_argument("--source", help="URL MJPEG, index de webcam, ou fichier vidéo")
    ap.add_argument("--model", default="depth-anything/DA3-BASE",
                    help="DA3-SMALL (plus rapide) / DA3-BASE / DA3-LARGE-1.1 (plus précis)")
    ap.add_argument("--cam-height", type=float, default=0.10,
                    help="hauteur de l'objectif au-dessus du sol, en mètres (fixe l'échelle)")
    ap.add_argument("--res", type=int, default=336,
                    help="résolution de traitement (336 ≈ 3 Hz sur M5 ; 504 plus précis mais ≈ 1,3 Hz)")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270),
                    help="rotation de l'image si la caméra est montée de travers")
    ap.add_argument("--robot-scale", type=float, default=1.0, help="taille du robot affiché")
    ap.add_argument("--render", choices=("splats", "points"), default="splats",
                    help="rendu de la carte au démarrage (modifiable dans la vue)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--loop", action="store_true", help="rejouer un fichier vidéo en boucle")
    ap.add_argument("--bridge-port", type=int, default=7400,
                    help="port TCP où diffuser la pose et les points pour ROS (0 = désactivé)")
    ap.add_argument("--ground", action="store_true",
                    help="mode robot au sol : caméra à hauteur fixe, sans roulis ni tangage "
                         "(pose gardée dans le plan, échelle réancrée sur le sol)")
    ap.add_argument("--no-mask", action="store_true",
                    help="ne pas retirer les personnes / objets mobiles de la carte")
    ap.add_argument("--no-record", action="store_true",
                    help="ne pas enregistrer la vidéo en direct (par défaut : out/rec_<date>.avi)")
    ap.add_argument("--sim", help="URL du simulateur (sim/run_world.py), ex. http://127.0.0.1:5060 : "
                                  "images de la caméra du robot ET de la D405 simulée d'un même pas de "
                                  "simulation, menu « Capteurs » et mesures vs vérité terrain")
    ap.add_argument("--sensors", choices=("rgb", "d405", "both"), default="rgb",
                    help="capteurs au démarrage avec --sim ou --d405 (modifiable dans la vue) : caméra du "
                         "robot, D405 seule, ou les deux")
    # Real D405: the robot only captures (d405_server.py), alignment and reprojection run here.
    ap.add_argument("--d405", metavar="URL",
                    help="serveur de la vraie D405 (rover_mapper/d405_server.py, par tunnel SSH), "
                         "ex. http://127.0.0.1:8091 ; rend --sensors d405|both disponibles sans --sim")
    ap.add_argument("--d405-height", type=float, default=0.06,
                    help="avec --d405 : hauteur de l'objectif de la D405 au-dessus du sol, m (« D405 seule » : "
                         "fixe le repère du sol ; l'échelle vient du capteur)")
    ap.add_argument("--d405-mount", type=float, nargs="+", metavar="V",
                    help="avec --d405, pour « Les deux » : position de la caméra couleur de la D405 dans le "
                         "repère de la caméra du robot, TX TY TZ en m [TANGAGE [LACET [ROULIS]] en degrés] ; "
                         "axes OpenCV x à droite, y vers le bas, z devant ; tangage > 0 = la D405 regarde plus "
                         "bas, lacet > 0 = plus à gauche, roulis > 0 = son côté droit plus bas. Valable "
                         "tourelle à pan 0 et au tilt de la mesure")
    ap.add_argument("--rgb-fov", type=float, default=100.0,
                    help="avec --d405 : champ horizontal (degrés) de l'image de --source, une image sténopé "
                         "centrée comme /video_rect du tableau de bord (son --rect-fov)")
    ap.add_argument("--d405-max-age", type=float, default=1.0,
                    help="avec --d405 : une image D405 plus vieille que ça (s) n'est pas utilisée")
    ap.add_argument("--rover", metavar="URL",
                    help="avec --d405 (facultatif) : serveur du rover (GET /state) pour prévenir si la "
                         "tourelle de la caméra n'est pas à pan 0 en « Les deux »")
    # Outdoors (forest). Without these flags the mapper keeps its indoor defaults.
    ap.add_argument("--outdoor", action="store_true",
                    help="préréglage extérieur (forêt) : " + " ".join(
                        f"--{k.replace('_', '-')}" + ("" if v is True else f" {v}") for k, v in OUTDOOR.items()))
    ap.add_argument("--max-depth", type=float, help="profondeur max. des points de la carte, m (défaut 6)")
    ap.add_argument("--voxel", type=float, help="taille de cellule de dédoublonnage de la carte, m (défaut 0,015)")
    ap.add_argument("--kf-min-dist", type=float,
                    help="distance min. entre images clés, m (défaut 0 : seulement 7 %% de la profondeur de la scène)")
    ap.add_argument("--mask-sky", action="store_true", default=None, help="retirer le ciel de la carte")
    ap.add_argument("--sensor-anchor", action="store_true", default=None,
                    help="réancrer l'échelle de chaque image clé sur le capteur de profondeur (D405)")
    ap.add_argument("--max-jump", type=float,
                    help="déplacement max. crédible par image, m (défaut 0 : pas de contrôle) ; au-delà la pose "
                         "est rejetée (fausses relocalisations en forêt)")
    ap.add_argument("--lost-extrapolate", action="store_true", default=None,
                    help="perdu : avancer la pose à la dernière vitesse mesurée au lieu de la figer")
    ap.add_argument("--cpu-threads", type=int,
                    help="threads CPU de torch et d'OpenBLAS (défaut : les leurs) ; 1 sur la VM : mêmes "
                         "résultats, inférence ~25 %% plus rapide, CPU presque divisé par deux")
    ap.add_argument("--max-kf-data", type=int,
                    help="images clés qui gardent leurs images (~2,5 Mo chacune) ; au-delà, les moins utilisées "
                         "les libèrent mais restent dans la carte (défaut 0 : toutes ; longue démo : 800)")
    ap.add_argument("--point-stride", type=int,
                    help="un point de carte tous les N pixels (défaut 3 ; plus grand = carte plus légère à "
                         "afficher à travers une liaison lente)")
    ap.add_argument("--live-hz", type=float, default=12.0,
                    help="cadence de l'image caméra dans le panneau de la vue 3D (liaison lente : 2)")
    ap.add_argument("--true-k", action="store_true", default=None,
                    help="avec --sim : intrinsèques vraies de la caméra au lieu de celles estimées par DA3 "
                         "(avec --d405 : celles d'usine de la D405, ou celles de --rgb-fov)")
    ap.add_argument("--log", help="journal par image (défaut : out/live_log.csv en direct, "
                                  "out/file_log.csv pour un fichier)")
    args = ap.parse_args()
    if args.cpu_threads:
        # numpy's OpenBLAS starts one thread per core at import and they spin
        # after every small matrix product: on the VM 7 threads burned ~0.9
        # core for nothing. Must be set before numpy is first imported.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.cpu_threads))

    if args.sim and args.d405:
        ap.error("--d405 (vraie D405) et --sim (D405 simulée) s'excluent")
    if args.d405_mount is not None and not 3 <= len(args.d405_mount) <= 6:
        ap.error("--d405-mount : TX TY TZ [TANGAGE [LACET [ROULIS]]] (3 à 6 nombres)")
    if args.sim:
        uri = args.sim
    elif args.robot:
        uri = f"http://{args.robot}:5000/video_feed"
    elif args.source:
        uri = args.source
    elif args.d405:
        uri = None              # D405 alone: no robot camera
        if args.sensors != "d405":
            ap.error("--sensors rgb|both demande aussi la caméra du robot : --source ou --robot")
    else:
        ap.error("donner --robot IP, --source, --sim ou --d405")
    if args.d405 and args.sensors == "both" and args.d405_mount is None:
        ap.error("--sensors both avec --d405 demande --d405-mount (position de la D405 vue de la caméra du robot)")

    from rovermap.mapper import Mapper
    from rovermap.source import (MODE_BOTH, MODE_D405, MODE_RGB, D405Source, FrameSource, SimSource,
                                 TruthTracker, mode_settings, mount_pose, sensor_inputs)
    from rovermap.viewer import CAM_FORWARD, Viewer

    # Moving-object masking needs a small segmentation model; use it only if
    # its weights are already downloaded, so startup never waits on the network.
    import glob
    import torch
    weights_ready = bool(glob.glob(os.path.join(torch.hub.get_dir(), "checkpoints",
                                                "deeplabv3_mobilenet_v3_large-*.pth")))
    mask_moving = weights_ready and not args.no_mask
    if not args.no_mask and not weights_ready:
        print("Masquage des personnes désactivé : modèle de segmentation pas encore téléchargé.")
    print(f"Chargement du modèle {args.model}…")
    # Only what was asked for is passed: the Mapper's defaults stay the indoor ones.
    extra = dict(OUTDOOR) if args.outdoor else {}
    for k in ("max_depth", "voxel", "kf_min_dist", "mask_sky", "sensor_anchor", "max_jump", "lost_extrapolate",
              "cpu_threads", "max_kf_data", "point_stride"):
        if getattr(args, k) is not None:
            extra[k] = getattr(args, k)
    # Measured in the forest: the true K made tracking worse, so --outdoor leaves it off.
    true_k = bool(args.true_k)
    if extra or true_k:
        print("Réglages :", ", ".join(f"{k}={v}" for k, v in extra.items()) + (", K vraie" if true_k else ""))
    mapper = Mapper(model_name=args.model, process_res=args.res, cam_height=args.cam_height,
                    mask_moving=mask_moving, ground_robot=args.ground, **extra)
    sim = bool(args.sim)
    real_d405 = bool(args.d405)
    # Simulator and real D405 both give snapshots (several streams + sensor
    # mode); only the simulator gives the truth.
    snaps = sim or real_d405
    mode = {"rgb": MODE_RGB, "d405": MODE_D405, "both": MODE_BOTH}[args.sensors] if snaps else MODE_RGB
    default_range = tuple(mapper.sensor_range)
    # D405 mount (body frame, metres) as the simulator declares it: an assumption.
    mount = None
    if sim:
        import requests
        for _ in range(10):
            try:
                mount = tuple(requests.get(uri.rstrip("/") + "/state", timeout=3).json()["d405_camera"]["mount"])
                break
            except Exception:
                time.sleep(1.0)
    viewer = Viewer(mapper, port=args.port, robot_scale=args.robot_scale, render=args.render,
                    sensor_mode=mode, d405_available=snaps, d405_mount=mount)
    viewer.live_hz = max(0.2, args.live_hz)
    d405_T = mount_pose(*args.d405_mount) if args.d405_mount else None   # D405 colour cam -> robot cam
    if real_d405:
        viewer.sensor_note.content = (
            "<small>Vraie D405 : alignement et reprojection calculés sur cette machine. "
            + ("« Les deux » : les deux images ne sont pas prises au même instant (caméra du robot "
               "~0,5 s en retard), rouler lentement ; tourelle à pan 0." if d405_T is not None
               else "« Les deux » indisponible sans --d405-mount.") + "</small>")
    print(f"\n  Vue 3D : http://localhost:{args.port}\n  Source : {uri or '(pas de caméra du robot)'}"
          + (f"\n  D405 : {args.d405}" if real_d405 else "") + "\n")
    import signal
    os.makedirs("out", exist_ok=True)
    is_live = uri is None or uri.isdigit() or "://" in uri
    tracker = TruthTracker() if sim else None

    def set_mode(m: str) -> None:
        """What the mapper must know about the camera it is fed (call before a reset)."""
        d405_h = args.d405_height if real_d405 else (mount[2] if mount else 0.06)
        mapper.cam_height, mapper.sensor_range = mode_settings(m, args.cam_height, d405_h, default_range)
        viewer.cam_forward = mount[0] if (m == MODE_D405 and mount) else CAM_FORWARD
        if snaps:
            src.mode = m

    def open_source():
        if sim:
            s = SimSource(uri, mode)
            viewer.attach_source(s)
            return s
        rec = None
        if uri is not None and is_live and not args.no_record:
            rec = time.strftime("out/rec_%Y%m%d_%H%M%S.avi")
            print(f"  Enregistrement : {rec}")
        s = FrameSource(uri, rotate=args.rotate, record=rec) if uri is not None else None
        if real_d405:
            s = D405Source(args.d405, mode, rgb_source=s, rgb_fov=args.rgb_fov, mount=d405_T,
                           max_age=args.d405_max_age, rover_url=args.rover)
        viewer.attach_source(s)
        return s

    src = open_source()
    set_mode(mode)
    if sim:
        print(f"  Capteurs : {mode} (menu « Capteurs » dans la vue)"
              + (f" · D405 supposée à x={mount[0]:.3f} m, z={mount[2]:.3f} m" if mount else "") + "\n")
    elif real_d405:
        print(f"  Capteurs : {mode} (menu « Capteurs » dans la vue) · D405 à {args.d405_height:.3f} m du sol"
              + (" · position vue de la caméra du robot : " + " ".join(f"{v:g}" for v in args.d405_mount)
                 if args.d405_mount else "") + "\n")
    last_reopen = time.time()

    # Pose + local points for the ROS side (polaris guidance_rover_vo).
    bridge = None
    if args.bridge_port:
        from rovermap.bridge import StateServer
        from rovermap.export import rover_state
        try:
            bridge = StateServer(args.bridge_port)
            print(f"  Pont ROS : tcp://0.0.0.0:{args.bridge_port} "
                  f"(depuis Docker : host.docker.internal:{args.bridge_port})\n")
        except OSError as e:
            print(f"  Pont ROS désactivé : port {args.bridge_port} indisponible ({e})\n")
    # Closed terminal / kill: stop cleanly so the recording is finalised.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGHUP, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    last_seq, fps, last_err = -1, 0.0, None
    last_truth_print = 0.0
    n_real = n_lost = 0
    import numpy as np
    # One line per processed frame, to understand tracking losses afterwards.
    # Live sessions and file replays get separate logs (a replay must not
    # overwrite the log of a live session running at the same time).
    log_path = args.log or ("out/live_log.csv" if is_live else "out/file_log.csv")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "w", buffering=1)
    log.write("t,frame_seq,skipped,dt,quality,overlap,lost,keyframes,new_kf,sharpness,w,h,"
              "fx_norm,pos_err,rot_err,x,y,z,true_x,true_y,true_z\n")
    t_start = time.time()
    try:
        while True:
            if viewer.sensor_mode_requested is not None:
                # Another sensor set: a different camera / scale source, so a new map.
                mode = viewer.sensor_mode_requested
                viewer.sensor_mode_requested = None
                print("Capteurs :", mode)
                set_mode(mode)
                viewer.reset_requested = True
            if viewer.reset_requested:
                viewer.reset_requested = False
                mapper.reset()
                viewer.clear()
                if tracker is not None:
                    tracker.reset()
                    viewer.show_metrics(None)
            # Reconnect on request (button), or by itself when a live camera
            # stops (e.g. Continuity Camera dropped the iPhone).
            dropped = is_live and src.ended and time.time() - last_reopen > 2.0
            if viewer.reconnect_requested or dropped:
                viewer.reconnect_requested = False
                print("Reconnexion de la caméra…")
                src.stop()
                src = open_source()
                last_reopen = time.time()
                last_seq, last_err = -1, None
                continue
            sd = T_true = d_true = K_true = None
            if snaps:
                # One simulation step: images, depth and true poses together
                # (real D405: the same dict, without any truth).
                seq, snap = src.snapshot()
                frame = None
                if snap is not None and seq != last_seq:
                    if sim and mount is None:
                        mount = tuple(float(v) for v in snap["d405_mount"])
                        set_mode(mode)
                    frame, sd, K_true, T_true, d_true = sensor_inputs(snap, mode)
                    if real_d405:
                        T_true = d_true = None   # identity placeholders, not a truth to log
            else:
                seq, frame = src.latest()
            if frame is None or seq == last_seq:
                if src.error and src.error != last_err:
                    last_err = src.error
                    print("Caméra :", src.error)
                    viewer.status.content = f"**Caméra injoignable** — `{src.error}`"
                if src.ended and src.is_file and args.loop:
                    src = open_source()
                    last_seq = -1
                time.sleep(0.005)
                continue
            skipped = seq - last_seq - 1 if last_seq >= 0 else 0
            last_seq = seq
            if is_blank(frame):
                # Covered lens / paused Continuity Camera sends black frames:
                # mapping them only produces garbage and "lost".
                viewer.status.content = "**Pas d'image** — caméra noire ou masquée, carte en pause"
                time.sleep(0.05)
                continue
            t0 = time.time()
            result = mapper.step(frame, sd, K_true if true_k else None)
            if result is None:
                viewer.status.content = "**Image inexploitable** — floue, surexposée ou sans détail"
                if tracker is not None:
                    tracker.n_none += 1
                continue
            viewer.update(result, fps)
            if tracker is not None:
                tracker.add(mapper, result, T_true, d_true)
                met = tracker.metrics()
                viewer.show_metrics(met)
                # Same numbers in the text log every 30 s, to follow a run without the page.
                if time.time() - last_truth_print > 30.0:
                    last_truth_print = time.time()
                    print(f"[vérité] {time.strftime('%H:%M:%S')} images {met['frames']} perdues "
                          f"{100 * met['lost_rate']:.1f} % · images clés {len(mapper.keyframes)} · "
                          f"dt {met['dt']:.2f} s · " + (
                              f"ATE {met['ate']:.2f} m (sans échelle {met['ate_noscale']:.2f} m) · "
                              f"échelle {100 * met['scale_err']:+.1f} % · "
                              f"récent {100 * met.get('scale_recent', float('nan')):+.1f} % / "
                              f"{met.get('ate_recent', float('nan')):.2f} m · "
                              if met.get("ready") else "")
                          + (f"err. prof. {100 * met['depth_err']:.1f} %" if met.get("depth_err") is not None
                             else ""), flush=True)
            elif real_d405:
                # No truth on the robot: what the D405 says about the map, every 30 s.
                n_real, n_lost = n_real + 1, n_lost + int(result.lost)
                if time.time() - last_truth_print > 30.0:
                    last_truth_print = time.time()
                    r = mapper.sensor_ratios
                    print(f"[D405] {time.strftime('%H:%M:%S')} {mode} · images {n_real} perdues {n_lost} · "
                          f"images clés {len(mapper.keyframes)} · échelle "
                          + (f"calée sur le capteur, capteur/carte récent {np.median(r[-10:]):.2f}"
                             if mapper.sensor_scaled else f"pas encore calée ({len(r)} mesures)"), flush=True)
            if bridge is not None and bridge.n_clients:
                bridge.publish(rover_state(mapper, result, time.time(), cam_forward=viewer.cam_forward))
            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1.0 / max(dt, 1e-3)
            sharp = mapper._sharp_hist[-1] if mapper._sharp_hist else 0.0
            log.write(f"{t0 - t_start:.2f},{seq},{skipped},{result.dt:.3f},{result.quality:.2f},"
                      f"{result.overlap:.2f},{int(result.lost)},{len(mapper.keyframes)},"
                      f"{int(result.new_keyframe is not None)},{sharp:.0f},{frame.shape[1]},{frame.shape[0]},"
                      f"{result.K[0, 0] / result.image.shape[1]:.3f},{mapper.last_align[0]:.3f},"
                      f"{mapper.last_align[1]:.1f},"
                      # Camera position (map frame at that moment) and, in simulation, the truth.
                      + ",".join(f"{v:.3f}" for v in result.c2w[:3, 3]) + ","
                      + (",".join(f"{v:.3f}" for v in T_true[:3, 3]) if T_true is not None else ",,")
                      + "\n")
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()


if __name__ == "__main__":
    main()
