#!/bin/bash
# On polaris-gpu (Ubuntu 22.04, NVIDIA L4): build the two Python 3.12 environments of the
# rovermap project, like .venv and .venv-ma on the Mac but with CUDA PyTorch.
#   ~/rovermap/.venv     DA3 mapper, simulation, navigation policy
#   ~/rovermap/.venv-ma  MapAnything benchmark
# Re-runnable. Log: ~/setup_envs.log. Ends with a CUDA check of both.
set -eo pipefail
cd ~/rovermap
python3 -m pip install -q --user uv
UV=~/.local/bin/uv
$UV python install 3.12
[ -d .venv ] || $UV venv -q -p 3.12 .venv
$UV pip install -q --python .venv/bin/python -r scripts/vm/requirements-rovermap.txt
$UV pip install -q --python .venv/bin/python -e third_party/da3 --no-deps
[ -d .venv-ma ] || $UV venv -q -p 3.12 .venv-ma
$UV pip install -q --python .venv-ma/bin/python -r scripts/vm/requirements-mapanything.txt
$UV pip install -q --python .venv-ma/bin/python -e third_party/map-anything --no-deps
for v in .venv .venv-ma; do
  $v/bin/python -c "import torch; print('$v', 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
done
.venv/bin/python -c "import depth_anything_3, mujoco, viser, stable_baselines3; print('rovermap imports ok')" \
  || .venv/bin/python -c "import mujoco, viser, stable_baselines3; print('rovermap imports ok (check the DA3 package name)')"
.venv-ma/bin/python -c "import mapanything; print('mapanything import ok')"
echo "ENVS OK $(date)"
