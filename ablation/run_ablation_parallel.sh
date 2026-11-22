#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runs all ablation variants in parallel across 2 GPUs using tmux.
# Each GPU gets its own tmux session + nohup log; results land in
# ablation/ablation_results_gpu{0,1}.csv and are merged into
# ablation/ablation_results_all.csv at the end.
#
# Usage:
#   cd /nfs/data8/schlegel/git/ecg-som
#   bash ablation/run_ablation_parallel.sh
# ---------------------------------------------------------------------------

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

VENV="$ROOT_DIR/.venv/bin/activate"
ABLATION="$SCRIPT_DIR/ablation.py"

# 48 variants split evenly across 2 GPUs (24 each)
GPU0_VARIANTS="baseline no_disentangle no_age no_sex no_temporal no_record_attn no_som_commit no_som_smooth no_kl small_latent large_latent small_som large_som narrow_encoder wide_encoder small_fc large_fc small_kernel large_kernel small_age_latent large_age_latent no_age_corr more_topk less_topk"
GPU1_VARIANTS="fast_ramp slow_ramp theta_low theta_high alpha_low alpha_high beta_low beta_high gamma_low gamma_high tau_low tau_high eta_low eta_high delta_age_low delta_age_high delta_sex_low delta_sex_high lr_low lr_high wd_low wd_high dropout_low dropout_high"

SESSION_PREFIX="abl"

launch_session() {
    local gpu=$1
    local variants=$2
    local session="${SESSION_PREFIX}_gpu${gpu}"
    local log="$SCRIPT_DIR/ablation_gpu${gpu}.log"
    local csv="$SCRIPT_DIR/ablation_results_gpu${gpu}.csv"

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

echo ""
echo "Both GPU sessions launched. Monitor with:"
echo "  tmux attach -t ${SESSION_PREFIX}_gpu0    (Ctrl-B D to detach)"
echo "  tail -f ablation/ablation_gpu{0,1}.log"
echo ""
echo "After both finish, merge results with:"
echo "  python ablation/merge_ablation_results.py"
