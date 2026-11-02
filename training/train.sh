#!/usr/bin/env bash
# Fine-tune Piper's LJ Speech "high" voice on the prepared data, then export
# every stage snapshot to ONNX before exiting.
#
#   bash train.sh /root/work               # start (epoch 0 of 2000)
#   bash train.sh /root/work resume        # continue from runs/last.ckpt
#
# Log: <work>/runs/train.log. Voices: <work>/export/*.onnx (+ .onnx.json).
set -uo pipefail
W=${1:-/root/work}; MODE=${2:-start}
R=$W/runs; C=$R/lightning_logs/version_0/checkpoints
mkdir -p "$R" "$W/export"
cd "$W/piper1-gpl" && . .venv/bin/activate

ARGS=(
  --data.voice_name ships_computer --data.csv_path "$W/data/metadata.csv" --data.audio_dir "$W/data/wavs"
  --model.sample_rate 22050 --data.espeak_voice en-us --data.cache_dir "$W/cache"
  --data.config_path "$R/config.json" --data.batch_size 24 --data.length_bucketing true
  --model.warmstart_ckpt "$W/base/ljspeech-2000.ckpt" --model.mos_metric none
  # Piper's trainer defaults to the "medium" architecture; these are "high",
  # read from the base checkpoint's hyperparameters.
  --model.resblock 1 --model.resblock_kernel_sizes "(3, 7, 11)"
  --model.resblock_dilation_sizes "((1, 3, 5), (1, 3, 5), (1, 3, 5))"
  --model.upsample_rates "(8, 8, 2, 2)" --model.upsample_initial_channel 512
  --model.upsample_kernel_sizes "(16, 16, 4, 4)"
  --trainer.max_epochs 2000 --trainer.default_root_dir "$R"
)
[ "$MODE" = resume ] && ARGS+=(--ckpt_path "$C/last.ckpt")

python3 -u "$(dirname "$0")/train_wrap.py" fit "${ARGS[@]}" >> "$R/train.log" 2>&1
echo "training exited rc=$?" >> "$R/train.log"

for c in "$C"/stage-*.ckpt; do
  o="$W/export/$(basename "$c" .ckpt).onnx"
  [ -f "$o" ] && continue
  python3 -m piper.train.export_onnx --checkpoint "$c" --output-file "$o" && cp "$R/config.json" "$o.json"
done
echo "exported: $(ls "$W"/export/*.onnx | wc -l) voices" >> "$R/train.log"
