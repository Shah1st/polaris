#!/bin/bash
# Run the whole Polaris chain locally on a Mac (Apple GPU through MPS), talking to the rover and
# the phone on the local network. Works with the bash 3.2 of macOS.
#
#   scripts/mac/run_local.sh start [service...]    start everything (or the named services)
#   scripts/mac/run_local.sh stop [service...]     stop them (processes only: see below)
#   scripts/mac/run_local.sh restart [service...]
#   scripts/mac/run_local.sh status                processes, ports, one read of each service
#   scripts/mac/run_local.sh logs [service]        follow the logs
#
# Services: dashboard (8082), perception (8095), fusion (8096), mapper (8081).
# Logs: out/logs/<service>.log   pid files: out/run/<service>.pid
#
# Nothing here sends a motor command, and "stop" does NOT send /stop to the rover: it only ends
# the local processes. The rover moves only from the dashboard (keys, or "go" to a clicked goal).
#
# Settings (environment variables):
#   POLARIS_PY     python of the rovermap environment (default: ../.venv or .venv next to the repo)
#   ROVER          rover API                    (default http://172.20.10.2:8080)
#   PHONE          phone stream, '' = no drone  (default http://172.20.10.6:8080/stream)
#                  Default: the camera phone plugged into this Mac by USB (its .local name:
#                  under 1 ms, 15 pictures a second). Over Wi-Fi give its address, and
#                  .../snapshot.jpg rather than .../stream (a pushed stream takes all the air
#                  time of a slow Wi-Fi shared with the rover)
#   PHONE_SHARE    share of the time the phone may spend sending pictures (default 0.25)
#   PHONE_NAME     name shown on the drone map  (default: iPhone CamStream (<host>))
#   D405           D405 server of the rover, e.g. http://172.20.10.2:8091 ('' = rover camera only)
#   D405_MOUNT     "TX TY TZ [PITCH [YAW [ROLL]]]" of the D405 seen from the rover camera (with D405)
#   CAM_HEIGHT     lens height above the floor, m (default 0.22)
#   FUSION_PERIOD  s between two looks of the fusion (default 1.0)
#   SEGMENT        m driven in one go during a mission (default 2.0; 0: a stop every 25 cm)
#   SPEED          m/s during a mission (default 0.30, the rover's own maximum)
#   DRONE_HEIGHT   height of the phone above the floor, m (default 1.8): scale of the drone map
#   DRONE_MPP      metres per pixel of the phone's picture, measured (replaces DRONE_HEIGHT)
#   DRONE_DETECTOR ia (default: a vision model finds the rover, needs transformers) or marqueur
#   DRONE_MARKER   ArUco id on the rover's top (default 10): places the drone map on the rover's

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROVER="${ROVER:-http://172.20.10.2:8080}"
PHONE="${PHONE-http://iphone-de-paco.local:8080/stream}"
D405="${D405:-}"
D405_MOUNT="${D405_MOUNT:-}"
CAM_HEIGHT="${CAM_HEIGHT:-0.22}"
FUSION_PERIOD="${FUSION_PERIOD:-1.0}"
PHONE_SHARE="${PHONE_SHARE:-0.25}"
SEGMENT="${SEGMENT:-2.0}"
SPEED="${SPEED:-0.30}"
DRONE_HEIGHT="${DRONE_HEIGHT:-1.8}"
DRONE_MARKER="${DRONE_MARKER:-10}"
DRONE_MPP="${DRONE_MPP:-}"
DRONE_DETECTOR="${DRONE_DETECTOR:-ia}"
PHONE_HOST="$(printf '%s' "$PHONE" | sed -E 's#^[a-z]+://([^@/]*@)?([^:/]+).*#\2#')"
PHONE_NAME="${PHONE_NAME:-iPhone CamStream ($PHONE_HOST)}"
ALL="dashboard perception fusion mapper"
LOGS="$ROOT/out/logs"
RUN="$ROOT/out/run"

if [ -n "${POLARIS_PY:-}" ]; then
  PY="$POLARIS_PY"
elif [ -x "$ROOT/../.venv/bin/python" ]; then
  PY="$ROOT/../.venv/bin/python"
else
  PY="$ROOT/.venv/bin/python"
fi

port_of() {
  case "$1" in
    dashboard) echo 8082 ;;
    perception) echo 8095 ;;
    fusion) echo 8096 ;;
    mapper) echo 8081 ;;
    *) echo "unknown service: $1 (known: $ALL)" >&2; return 1 ;;
  esac
}

# The script each service runs: used to check that a pid file still points at our process.
script_of() {
  case "$1" in
    dashboard) echo "teleop/server.py" ;;
    perception) echo "perception_service.py" ;;
    fusion) echo "fusion_reelle.py" ;;
    mapper) echo "run.py" ;;
  esac
}

pid_of() {  # prints the pid when the service runs
  local f="$RUN/$1.pid" pid
  [ -f "$f" ] || return 1
  pid="$(cat "$f")"
  [ -n "$pid" ] || return 1
  ps -p "$pid" -o command= 2>/dev/null | grep -q "$(script_of "$1")" || return 1
  echo "$pid"
}

port_holder() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $1 " (pid " $2 ")"}' | sort -u | tr '\n' ' '
}

launch() {  # launch <service> <directory> <command...>
  local name="$1" dir="$2"
  shift 2
  ( cd "$dir" && PYTHONUNBUFFERED=1 exec nohup "$@" >>"$LOGS/$name.log" 2>&1 </dev/null ) &
  echo $! >"$RUN/$name.pid"
}

start_one() {
  local name="$1" port holder
  port="$(port_of "$name")" || return 1
  if pid_of "$name" >/dev/null; then
    echo "$name: already running (pid $(pid_of "$name"))"
    return 0
  fi
  holder="$(port_holder "$port")"
  if [ -n "$holder" ]; then
    echo "$name: port $port is taken by $holder- not started (an ssh tunnel to the VM?)"
    return 1
  fi
  echo "---- $(date '+%F %T') start ----" >>"$LOGS/$name.log"
  case "$name" in
    dashboard)
      set -- "$PY" teleop/server.py --port 8082 --robot "$ROVER" --robot-lan "$ROVER" \
        --map-port 8081 --fusion http://127.0.0.1:8096 --suivi out/suivi.json \
        --perception-debug http://127.0.0.1:8095/debug.jpg
      # no --drone-lan: the browser would open a second full-size stream on the phone
      if [ -n "$PHONE" ]; then set -- "$@" --drone "$PHONE" --drone-part "$PHONE_SHARE"; fi
      launch "$name" "$ROOT" "$@"
      ;;
    perception)
      launch "$name" "$ROOT" "$PY" rover_mapper/perception_service.py \
        --source http://127.0.0.1:8082/video_rect --cam-height "$CAM_HEIGHT" --fov 100 \
        --range 2.0 --port 8095
      ;;
    fusion)
      # --delai-http / --delai-liaison: 8 s and 15 s were for the 4G link to the cloud machine
      set -- "$PY" sim/fusion_reelle.py --port 8096 --rover "$ROVER" \
        --perception http://127.0.0.1:8095 --periode "$FUSION_PERIOD" \
        --delai-http 3 --delai-liaison 6 --segment "$SEGMENT" --vitesse "$SPEED"
      if [ -n "$PHONE" ]; then
        # the phone is read through the dashboard (one connection to the phone)
        set -- "$@" --drone-nom "$PHONE_NAME" --drone-source http://127.0.0.1:8082/drone_feed \
          --drone-hauteur "$DRONE_HEIGHT" --drone-marqueur "$DRONE_MARKER" \
          --drone-detecteur "$DRONE_DETECTOR" --drone-cumul
        if [ -n "$DRONE_MPP" ]; then set -- "$@" --drone-mpp "$DRONE_MPP"; fi
      fi
      launch "$name" "$ROOT" "$@"
      ;;
    mapper)
      set -- "$PY" run.py --source http://127.0.0.1:8082/video_rect --cam-height "$CAM_HEIGHT" \
        --res 504 --max-jump 0.3 --max-depth 4 --live-hz 2 --render points --port 8081 \
        --bridge-port 7400
      if [ -n "$D405" ]; then
        set -- "$@" --rgb-fov 100 --d405 "$D405" --sensors both --rover "$ROVER"
        # shellcheck disable=SC2086
        if [ -n "$D405_MOUNT" ]; then set -- "$@" --d405-mount $D405_MOUNT; fi
      fi
      launch "$name" "$ROOT/rover_mapper" "$@"
      ;;
  esac
  echo "$name: started (pid $(cat "$RUN/$name.pid"), port $port, log out/logs/$name.log)"
}

stop_one() {
  local name="$1" pid n=0
  port_of "$name" >/dev/null || return 1
  if ! pid="$(pid_of "$name")"; then
    echo "$name: not running"
    rm -f "$RUN/$name.pid"
    return 0
  fi
  kill "$pid" 2>/dev/null
  while kill -0 "$pid" 2>/dev/null && [ "$n" -lt 25 ]; do sleep 0.2; n=$((n + 1)); done
  if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null; fi
  rm -f "$RUN/$name.pid"
  echo "$name: stopped"
}

wait_port() {  # wait_port <port> <seconds>
  local n=0
  while [ "$n" -lt "$2" ]; do
    if curl -s -o /dev/null -m 1 "http://127.0.0.1:$1/"; then return 0; fi
    sleep 1
    n=$((n + 1))
  done
  return 1
}

status() {
  local name port pid
  for name in $ALL; do
    port="$(port_of "$name")"
    if pid="$(pid_of "$name")"; then
      echo "$name: running (pid $pid), port $port held by $(port_holder "$port")"
    else
      echo "$name: stopped; port $port held by: $(port_holder "$port")"
    fi
  done
  echo
  echo "rover $ROVER/state:"
  curl -s -m 3 "$ROVER/state" | "$PY" -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print('  battery %s V, camera %s fps, mode %s' % (d.get('battery_v'), (d.get('camera') or {}).get('fps'), (d.get('ctrl') or {}).get('mode')))
except Exception as e:
    print('  no answer')" 2>/dev/null
  echo "fusion http://127.0.0.1:8096/status.json:"
  curl -s -m 3 http://127.0.0.1:8096/status.json | "$PY" -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print('  %s, views fused %s, rover tiles %s, info: %s %s' % (d.get('etat'), d.get('vues_fusionnees'), d.get('tuiles_rover'), d.get('info'), d.get('erreur') or ''))
except Exception as e:
    print('  no answer')" 2>/dev/null
}

cmd="${1:-}"
[ $# -gt 0 ] && shift
services="${*:-$ALL}"
mkdir -p "$LOGS" "$RUN"

case "$cmd" in
  start)
    if [ ! -x "$PY" ]; then
      echo "python not found: $PY (set POLARIS_PY to the rovermap environment)" >&2
      exit 1
    fi
    for s in $services; do
      start_one "$s"
      # the others read the dashboard's /video_rect: wait for it
      if [ "$s" = dashboard ]; then wait_port 8082 15 || echo "dashboard: no answer yet on 8082"; fi
    done
    echo
    echo "Dashboard  http://localhost:8082        (rover, drone, maps; click on the drone map = goal)"
    echo "Mission    http://localhost:8082/mission"
    echo "3D map     http://localhost:8081        (ready about 20 s after the start)"
    ;;
  stop)
    # the readers first, the dashboard last
    for s in mapper fusion perception dashboard; do
      case " $services " in *" $s "*) stop_one "$s" ;; esac
    done
    ;;
  restart)
    "$0" stop $services
    "$0" start $services
    ;;
  status) status ;;
  logs)
    if [ $# -gt 0 ]; then tail -n 40 -f "$LOGS/$1.log"; else tail -n 20 -f "$LOGS"/*.log; fi
    ;;
  *)
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
