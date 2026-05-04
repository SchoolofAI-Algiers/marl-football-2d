#!/bin/bash
export PYTHONWARNINGS="ignore::DeprecationWarning:pkg_resources"
export TF_CPP_MIN_LOG_LEVEL=3
export PYGAME_HIDE_SUPPORT_PROMPT=1

NAME="${1:?Usage: $0 <experiment-name> [extra train args...]}"
shift
RUN_ID="$(date +%m_%d_%H_%M)_${NAME}"
LOG_DIR="logs/sac/$RUN_ID"
mkdir -p "$LOG_DIR"
echo "Logging to $LOG_DIR"

tensorboard --logdir "logs/sac" --bind_all 2>/dev/null &
TB_PID=$!
trap 'kill $TB_PID 2>/dev/null' EXIT
echo "TensorBoard running at http://localhost:6006/ (PID: $TB_PID)"

python -m rl.sac.train \
    --experiment-name "$RUN_ID" \
    --num-workers 0 \
    --num-envs-per-worker 4 \
    --num-gpus 1 \
    --opponent none \
    --ball-placement random \
    --render \
    "$@"

# Usage:
#   ./rl/sac/train.sh reward_fix_v1
#   ./rl/sac/train.sh baseline --warmup-steps 100000 --alpha-lr 1e-4
#   ./rl/sac/train.sh selfplay_2v2 --opponent agents --num-workers 4
