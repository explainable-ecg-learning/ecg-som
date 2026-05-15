#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Re-runs only the 24 previously-failed ablation variants across 4 GPUs.
#
# Usage:
#   cd /nfs/data8/schlegel/git/ecg-som
#   bash ablation/run_ablation_retry.sh
# ---------------------------------------------------------------------------

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

VENV="$ROOT_DIR/DisentangledECG/.venv/bin/activate"
ABLATION="$SCRIPT_DIR/ablation.py"

# 24 failed variants split across 4 GPUs (6 each)
GPU0_VARIANTS="no_som_smooth no_kl small_latent large_latent small_som eta_high"
GPU1_VARIANTS="narrow_encoder wide_encoder small_fc large_fc small_kernel large_kernel"
GPU2_VARIANTS="small_age_latent large_age_latent delta_age_low delta_age_high delta_sex_low delta_sex_high"
GPU3_VARIANTS="lr_low lr_high wd_low wd_high dropout_low dropout_high"

SESSION_PREFIX="abl_retry"

launch_session() {
    local gpu=$1
    local variants=$2
    local session="${SESSION_PREFIX}_gpu${gpu}"
    local log="$SCRIPT_DIR/ablation_retry_gpu${gpu}.log"
    local csv="$SCRIPT_DIR/ablation_results_retry_gpu${gpu}.csv"

    # Kill any previous session with the same name
    tmux kill-session -t "$session" 2>/dev/null || true

    tmux new-session -d -s "$session" bash
    tmux send-keys -t "$session" \
        "source $VENV && CUDA_VISIBLE_DEVICES=$gpu nohup python $ABLATION --variants $variants --output $csv > $log 2>&1; echo 'GPU$gpu DONE' >> $log" \
        Enter

    echo "[launcher] Started tmux session '$session' (GPU $gpu) → log: $log  csv: $csv"
}

launch_session 0 "$GPU0_VARIANTS"
launch_session 1 "$GPU1_VARIANTS"
launch_session 2 "$GPU2_VARIANTS"
launch_session 3 "$GPU3_VARIANTS"

echo ""
echo "All 4 retry sessions launched.  Monitor with:"
echo "  tmux ls"
echo "  tail -f ablation/ablation_retry_gpu0.log"
echo ""
echo "When all sessions are done, merge results with:"
echo "  python ablation/merge_ablation_results.py"
