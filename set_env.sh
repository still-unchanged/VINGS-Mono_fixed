#!/usr/bin/env bash
# One-shot environment setup for VINGS-Mono.
#
# Run from the repo root, inside the target conda env (do NOT use sudo):
#   conda create -n vings_vio python=3.9 -y && conda activate vings_vio
#   bash set_env.sh
#
# Assumes `git clone --recursive`. If you cloned without it:
#   git submodule update --init --recursive
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
echo "== interpreter: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
[ "$($PY -c 'import sys; print(sys.version_info[:2] >= (3, 9))')" = "True" ] \
  || { echo "ERROR: need python >= 3.9"; exit 1; }

echo "== submodules"
git submodule update --init --recursive

echo "== torch (2.0.1 / cu118)"
$PY -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "== torch-scatter"
# The pyg wheel index ships 2.0.2 for cu118; that is what the original env used.
$PY -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

echo "== requirements"
$PY -m pip install -r requirements.txt

echo "== CUDA extensions"
# diff-surfel-rasterization (2DGS)
$PY -m pip install submodules/diff-surfel-rasterization --no-build-isolation
# dbaf -> droid_backends. Install into THIS interpreter, never with sudo.
$PY -m pip install submodules/dbaf --no-build-isolation

echo "== gtsam (VIO fork, must be on the 'vio' branch for mode: vio)"
# A plain `pip install gtsam` gives upstream 4.x, which has no BA2GTSAM /
# CustomHessianFactor / evaluateErrorCustom. Those live only in this fork.
if [ -d submodules/gtsam/build/python ]; then
  echo "   installing from submodules/gtsam/build/python"
  $PY -m pip install submodules/gtsam/build/python
else
  echo "   building gtsam from source (slow, ~20-30 min)"
  cmake -S submodules/gtsam -B submodules/gtsam/build \
        -DGTSAM_BUILD_PYTHON=ON -DGTSAM_PYTHON_VERSION=3.9 \
        -DGTSAM_WITH_TBB=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build submodules/gtsam/build -j"$(nproc)"
  $PY -m pip install submodules/gtsam/build/python
fi

echo "== verify"
$PY - <<'PY'
import gtsam, torch
print("torch       ", torch.__version__, "cuda", torch.cuda.is_available())
print("gtsam       ", gtsam.__file__)
missing = [n for n in ("BA2GTSAM", "CustomHessianFactor") if not hasattr(gtsam, n)]
if not hasattr(gtsam.CombinedImuFactor, "evaluateErrorCustom"):
    missing.append("CombinedImuFactor.evaluateErrorCustom")
if missing:
    raise SystemExit(f"ERROR: gtsam is missing {missing}. "
                     "You are on upstream gtsam, not the 'vio' fork.")
print("gtsam VIO interfaces: OK")
PY

cat <<'EOF'

Setup done. Next:
  1. weights -> ./ckpts/   (droid.pth is the only one needed for mode: vio/vo)
  2. datasets -> ./data/   (see docs/PREPARE_DATA.md; paths in the yamls are
                            rewritten onto <repo>/data automatically)
  3. run from the repo root:
       python scripts/run.py configs/kitti/sync/kitti_2011_09_30_drive_0027.yaml
EOF
