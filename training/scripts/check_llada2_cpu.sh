#!/bin/bash
#SBATCH --job-name=orl-llada2-check
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/jacopo04/ORL/.logs/%x-%j.out
#SBATCH --error=/scratch/jacopo04/ORL/.logs/%x-%j.out
#SBATCH -D /scratch/jacopo04/ORL
#SBATCH --account=def-rudner

set -euo pipefail

REPO_ROOT=/scratch/jacopo04/ORL
RUNTIME_ROOT="${ORL_RUNTIME_ROOT:-$REPO_ROOT/.runtime}"
SIF="${ORL_SIF:-$RUNTIME_ROOT/sglang-paper-cu129.sif}"
VENV="${ORL_VENV:-$RUNTIME_ROOT/venv}"
MODEL_PATH="${ORL_MODEL_PATH:-$RUNTIME_ROOT/models/LLaDA2.1-mini-actor}"
DATA_DIR="${ORL_DATA_DIR:-$RUNTIME_ROOT/data/ORL-training-data}"

cd "$REPO_ROOT"
module --force purge
module load StdEnv/2023 python/3.11.5 apptainer/1.4.5

# Checkpointing imports Torch's compiler dependencies; use the training runtime
# rather than the login node's incomplete user-site Torch installation.
if [[ -s "$SIF" && -x "$VENV/bin/python3" ]]; then
  OMP_NUM_THREADS=1 apptainer exec -B "$REPO_ROOT:$REPO_ROOT" \
    "$SIF" "$VENV/bin/python3" training/tests/test_llada2_bptt.py
  OMP_NUM_THREADS=1 apptainer exec -B "$REPO_ROOT:$REPO_ROOT" \
    "$SIF" "$VENV/bin/python3" training/tests/test_llada2_bptt_fsdp.py
  OMP_NUM_THREADS=1 apptainer exec -B "$REPO_ROOT:$REPO_ROOT" \
    "$SIF" "$VENV/bin/python3" training/tests/test_llada2_bptt_packing.py
else
  python training/tests/test_llada2_bptt.py
  python training/tests/test_llada2_bptt_fsdp.py
  python training/tests/test_llada2_bptt_packing.py
fi
python -m py_compile \
  training/verl/verl/trainer/ppo/llada2_bptt.py \
  training/verl/verl/trainer/ppo/ray_trainer.py \
  training/verl/verl/workers/actor/dp_actor.py
bash -n training/scripts/train_llada2.sh
bash -n training/scripts/setup_llada2_cluster.sh

common_args=(
  training/recipes/llada2_mini/orl.yaml
  --run-dir "$RUNTIME_ROOT/preflight/dry-run"
  --set "MODEL_PATH=$MODEL_PATH"
  --set "TRAIN_FILE=$DATA_DIR/gsm8k-math-code_l0-mix-20260909/train.parquet"
  --set "VAL_FILE=$DATA_DIR/code_he_clean_rebuild/val.parquet"
  --sif "$SIF"
  --python "$VENV/bin/python3"
  --gpus 0,1,2,3
  --steps 650
  --override ++trainer.n_gpus_per_node=4
  --override ++actor_rollout_ref.actor.fsdp_config.fsdp_size=4
  --override ++actor_rollout_ref.rollout.agent.num_workers=4
)

baseline_command="$(python training/scripts/orl_train.py "${common_args[@]}" --dry-run)"
grep -q 'trainer.total_training_steps=650' <<<"$baseline_command"
grep -q 'trainer.n_gpus_per_node=4' <<<"$baseline_command"

bptt_command="$(python training/scripts/orl_train.py "${common_args[@]}" \
  --override ++actor_rollout_ref.actor.generation_consistency_jsd.bptt.enable=true \
  --override ++actor_rollout_ref.actor.generation_consistency_jsd.bptt.horizon=2 \
  --override ++actor_rollout_ref.actor.generation_consistency_jsd.trace_all_ce_packing=all_blocks_per_round \
  --override ++actor_rollout_ref.actor.generation_consistency_jsd.trace_all_ce_packing_materialization=dense \
  --override ++actor_rollout_ref.actor.generation_consistency_jsd.trace_all_ce_parallelism=none \
  --dry-run)"
grep -q 'generation_consistency_jsd.bptt.enable=true' <<<"$bptt_command"
grep -q 'generation_consistency_jsd.trace_all_ce_packing=all_blocks_per_round' <<<"$bptt_command"

if [[ -s "$SIF" && -x "$VENV/bin/python3" ]]; then
  runtime_pythonpath="$REPO_ROOT/training/kernels/fast-diffusion-attention:$REPO_ROOT/training/sglang:$REPO_ROOT/training/verl"
  apptainer exec -B "$REPO_ROOT:$REPO_ROOT" \
    --env "PYTHONPATH=$runtime_pythonpath" "$SIF" "$VENV/bin/python3" -c \
    'from verl.trainer.ppo import llada2_bptt, ray_trainer; from verl.workers.actor import dp_actor; print("ORL runtime imports passed")'
fi

echo "CPU preflight passed for baseline and horizon-2 BPTT"
