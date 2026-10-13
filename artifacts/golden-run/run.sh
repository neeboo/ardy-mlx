#!/usr/bin/env bash
# THE single authorised reference inference run (nv-tlabs/ardy @ 693f74d).
# Read-only w.r.t. the system: no installs, no sudo, no config changes.
# Writes only under /home/mc/ardy-golden-2026-10-13/.
set -uo pipefail

cd /home/mc/services/ardy-engine/ardy

export HF_HOME=/home/mc/services/ardy-engine/hf-cache
export CHECKPOINTS_DIR=/home/mc/services/ardy-engine/checkpoints
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DEVICE=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENABLE_PARALLEL_LOADING=YES
export PYTHONPATH=/home/mc/ardy-golden-2026-10-13/hook
export ARDY_GOLDEN_DUMP=/home/mc/ardy-golden-2026-10-13/raw
export ARDY_MERGED_BASE=/home/mc/services/ardy-engine/llm2vec-base-merged
export ARDY_SUPERVISED_ADAPTER=/home/mc/services/ardy-engine/source-models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

PY=/home/mc/services/ardy-engine/venv/bin/python
ROOT=/home/mc/ardy-golden-2026-10-13
OUT=$ROOT/out/golden_core_h40_2s_seed0

# peak-VRAM sampler (2 s period)
(
  while true; do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
    sleep 2
  done
) > "$ROOT/gpu_samples.txt" 2>&1 &
MON=$!
echo "MONITOR_PID=$MON"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GIT=$(git -C /home/mc/services/ardy-engine/ardy rev-parse HEAD)"
echo "CMD: $PY scripts/generate.py \"A person walks forward.\" --model ARDY-Core-RP-20FPS-Horizon40 --duration 2.0 --num_samples 1 --seed 0 --no-postprocess --checkpoints_dir $CHECKPOINTS_DIR --output $OUT"

/usr/bin/time -v "$PY" scripts/generate.py \
  "A person walks forward." \
  --model ARDY-Core-RP-20FPS-Horizon40 \
  --duration 2.0 \
  --num_samples 1 \
  --seed 0 \
  --no-postprocess \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --output "$OUT"
RC=$?

kill "$MON" 2>/dev/null
wait "$MON" 2>/dev/null
echo "EXIT_CODE=$RC"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "DONE"
