#!/usr/bin/env bash
# RTX 5070 Ti (Blackwell, sm_120) 학습 환경 설치 — Ubuntu 22.04/24.04 (또는 WSL2 Ubuntu)
# AGX bok 학습 환경과 같은 버전: torch 2.8.0 / mmcv 2.1.0 / mmdet 3.3.0 / mmengine 0.10.7 / numpy 1.26.4
# 차이: torch 는 CUDA 12.8 빌드(cu128), mmcv 는 sm_120 용으로 소스 빌드.
#
# 사전 조건 (README 1~2단계):
#   - NVIDIA 드라이버 570 이상 (nvidia-smi 로 확인)
#   - CUDA Toolkit 12.8 (nvcc) — mmcv 빌드에 필요. 기본 경로 /usr/local/cuda-12.8
#   - python3.10 + venv, build-essential, git, ninja-build
#
# 사용: bash install.sh [설치 폴더, 기본 ~/kmji_train]
set -euo pipefail

ROOT="${1:-$HOME/kmji_train}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[1/6] 확인: 드라이버, nvcc"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$CUDA_HOME/bin/nvcc" --version | tail -1

echo "[2/6] venv: $ROOT/.venv"
mkdir -p "$ROOT"
python3.10 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install -U pip "setuptools<70" wheel ninja

echo "[3/6] PyTorch 2.8.0 + CUDA 12.8"
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA 를 못 찾음 (드라이버 확인)"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "arch", torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list(), "이 torch 빌드는 sm_120(RTX 50) 을 지원하지 않음"
EOF

echo "[4/6] 나머지 패키지 (AGX 와 같은 버전)"
pip install -r "$HERE/requirements.txt"

echo "[5/6] mmcv 2.1.0 소스 빌드 (sm_120, 10~30분)"
mkdir -p "$ROOT/src"
if [ ! -d "$ROOT/src/mmcv" ]; then
  git clone --depth 1 --branch v2.1.0 https://github.com/open-mmlab/mmcv.git "$ROOT/src/mmcv"
fi
cd "$ROOT/src/mmcv"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
export MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS="${MAX_JOBS:-8}"
pip install -v --no-build-isolation . 2>&1 | tee "$ROOT/mmcv_build.log" | tail -5

echo "[6/6] mmdet 3.3.0"
pip install mmdet==3.3.0

echo "설치 끝. 확인: source $ROOT/.venv/bin/activate && python $HERE/check_env.py --config $HERE/model_config.py"
