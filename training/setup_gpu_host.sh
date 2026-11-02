#!/usr/bin/env bash
# One-time setup on a Linux GPU host (we used Runpod, image
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404). Builds Piper's trainer and
# downloads the base checkpoint, verified by checksum.
#
#   bash setup_gpu_host.sh /root/work
#
# Put training data at <work>/data/{wavs/,metadata.csv} (see prepare_dataset.py)
# on local disk: writing checkpoints to a network volume stalled training.
set -euo pipefail
W=${1:-/root/work}
mkdir -p "$W/base"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential cmake ninja-build >/dev/null

cd "$W"
[ -d piper1-gpl ] || git clone -q https://github.com/OHF-voice/piper1-gpl.git
cd piper1-gpl && git checkout -q 057093803b
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
# scikit-build is needed by the dev build below but not listed by Piper.
pip install -q -e '.[train]' scikit-build cmake ninja
./build_monotonic_align.sh >/dev/null 2>&1
python3 setup.py build_ext --inplace >/dev/null 2>&1
python3 -c "import torch, piper.train.vits.lightning; from piper import phonemize_espeak; print('trainer ok, cuda', torch.cuda.is_available())"

# Base voice: LJ Speech "high" (public-domain data), from rhasspy/piper-checkpoints.
curl -sSL -o "$W/base/ljspeech-2000.ckpt" \
  https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ljspeech/high/ljspeech-2000.ckpt
echo "f49e9af70265433aa03d403efc2fe161e0c42b9083c3768a17338be4b0af720a  $W/base/ljspeech-2000.ckpt" | sha256sum -c -
