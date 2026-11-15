#!/bin/zsh
# Live 3D map from the iPhone camera (Continuity Camera), run from macOS Terminal
# so that macOS can ask for camera permission. View: http://localhost:8082
cd "$(dirname "$0")/.." || exit 1
CAM_HEIGHT="${CAM_HEIGHT:-1.0}"   # height of the phone above the floor, metres
echo "Carte 3D en direct depuis l'iPhone — vue : http://localhost:8082 (Ctrl+C pour arrêter)"
exec .venv/bin/python run.py --source 1 --cam-height "$CAM_HEIGHT" --port 8082
