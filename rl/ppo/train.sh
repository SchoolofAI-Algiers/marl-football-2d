#!/bin/bash
export PYTHONWARNINGS="ignore::DeprecationWarning:pkg_resources"
export TF_CPP_MIN_LOG_LEVEL=3
export PYGAME_HIDE_SUPPORT_PROMPT=1

NAME="${1:?Usage: $0 <experiment-name> [extra train args...]}"
shift
RUN_ID="$(date +%m_%d_%H_%M)_${NAME}"
LOG_DIR="logs/ppo/$RUN_ID"
mkdir -p "$LOG_DIR"
echo "Logging to $LOG_DIR"

tensorboard --logdir "logs/ppo" --bind_all --port 6007 2>/dev/null &
TB_PID=$!
trap 'kill $TB_PID 2>/dev/null' EXIT
echo "TensorBoard running at http://localhost:6007/ (PID: $TB_PID)"

python -m rl.ppo.train \
    --experiment-name "$RUN_ID" \
    --num-workers 0 \
    --num-envs-per-worker 4 \
    --num-gpus 1 \
    --opponent none \
    --ball-placement random \
    --render \
    "$@"

# Usage:
#   ./rl/ppo/train.sh reward_fix_v1
#   ./rl/ppo/train.sh baseline --lr 1e-4 --entropy-coeff 0.005
#   ./rl/ppo/train.sh selfplay_2v2 --opponent agents --num-workers 4
