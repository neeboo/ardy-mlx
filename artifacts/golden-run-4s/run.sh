#!/usr/bin/env bash
# THE second (and final) authorised reference inference run
# (nv-tlabs/ardy @ 693f74d), --duration 4.0 = 80 frames = two autoregressive windows.
#
# Identical to artifacts/golden-run/run.sh except:
#   * --duration 4.0 (was 2.0)
#   * ROOT is a NEW directory (the run-1 directory is left untouched)
#   * the hook is the fixed/extended v2 (incremental writes, per-window records)
#   * an extra memory sampler next to the GPU sampler
#
# Read-only w.r.t. the system: no installs, no sudo, no config changes.
# Writes only under $ROOT.
set -uo pipefail

cd /home/mc/services/ardy-engine/ardy

export HF_HOME=/home/mc/services/ardy-engine/hf-cache
export CHECKPOINTS_DIR=/home/mc/services/ardy-engine/checkpoints
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DEVICE=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENABLE_PARALLEL_LOADING=YES
export PYTHONPATH=/home/mc/ardy-golden-2026-10-13-4s/hook
export ARDY_GOLDEN_DUMP=/home/mc/ardy-golden-2026-10-13-4s/raw
export ARDY_MERGED_BASE=/home/mc/services/ardy-engine/llm2vec-base-merged
export ARDY_SUPERVISED_ADAPTER=/home/mc/services/ardy-engine/source-models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

PY=/home/mc/services/ardy-engine/venv/bin/python
ROOT=/home/mc/ardy-golden-2026-10-13-4s
OUT=$ROOT/out/golden_core_h40_4s_seed0

# peak-VRAM sampler (2 s period)
(
  while true; do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
    sleep 2
  done
) > "$ROOT/gpu_samples.txt" 2>&1 &
MON=$!

# memory/swap sampler (2 s period)
(
  while true; do
    echo "--- $(date -u +%H:%M:%S) ---"
    free -m | sed -n '2,3p'
    pgrep -f 'scripts/generate.py' | head -1 | while read -r p; do
      awk '/VmRSS|VmSwap/ {printf "%s ", $0} END {print ""}' "/proc/$p/status" 2>/dev/null
    done
    sleep 2
  done
) > "$ROOT/mem_samples.txt" 2>&1 &
MMON=$!

echo "MONITOR_PID=$MON MEM_MONITOR_PID=$MMON"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GIT=$(git -C /home/mc/services/ardy-engine/ardy rev-parse HEAD)"
echo "GIT_STATUS_BEFORE=$(git -C /home/mc/services/ardy-engine/ardy status --porcelain | wc -l)"
echo "CMD: $PY scripts/generate.py \"A person walks forward.\" --model ARDY-Core-RP-20FPS-Horizon40 --duration 4.0 --num_samples 1 --seed 0 --no-postprocess --checkpoints_dir $CHECKPOINTS_DIR --output $OUT"

/usr/bin/time -v "$PY" scripts/generate.py \
  "A person walks forward." \
  --model ARDY-Core-RP-20FPS-Horizon40 \
  --duration 4.0 \
  --num_samples 1 \
  --seed 0 \
  --no-postprocess \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --output "$OUT"
RC=$?

kill "$MON" "$MMON" 2>/dev/null
wait "$MON" "$MMON" 2>/dev/null
echo "EXIT_CODE=$RC"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GIT_STATUS_AFTER=$(git -C /home/mc/services/ardy-engine/ardy status --porcelain | wc -l)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "DONE"
