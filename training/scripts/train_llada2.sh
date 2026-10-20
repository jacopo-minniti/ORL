#!/bin/bash
#SBATCH --job-name=orl-llada2
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=/scratch/jacopo04/ORL/.logs/%x-%j.out
#SBATCH --error=/scratch/jacopo04/ORL/.logs/%x-%j.out
#SBATCH -D /scratch/jacopo04/ORL
#SBATCH --account=def-rudner
#SBATCH --requeue

set -euo pipefail

REPO_ROOT=/scratch/jacopo04/ORL
RUNTIME_ROOT="${ORL_RUNTIME_ROOT:-$REPO_ROOT/.runtime}"
SIF="${ORL_SIF:-$RUNTIME_ROOT/sglang-paper-cu129.sif}"
VENV="${ORL_VENV:-$RUNTIME_ROOT/venv}"
MODEL_PATH="${ORL_MODEL_PATH:-$RUNTIME_ROOT/models/LLaDA2.1-mini-actor}"
DATA_DIR="${ORL_DATA_DIR:-$RUNTIME_ROOT/data/ORL-training-data}"
ENV_FILE="${ORL_ENV_FILE:-$REPO_ROOT/../structured-diffusion/.env}"
VARIANT="${VARIANT:-baseline}"
BPTT_HORIZON="${BPTT_HORIZON:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-650}"
[[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'TRAIN_STEPS must be positive' >&2; exit 2; }

cd "$REPO_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Apply the explicit run destination after the shared environment file, which
# may define its own RUNS_DIR for other experiments.
RUNS_DIR="${ORL_RUNS_DIR:-${RUNS_DIR:-$RUNTIME_ROOT/runs}}"
mkdir -p .logs "$RUNS_DIR"

module --force purge
module load StdEnv/2023 python/3.11.5 apptainer/1.4.5

for path in "$SIF" "$VENV/bin/python3" "$MODEL_PATH/config.json" \
  "$DATA_DIR/gsm8k-math-code_l0-mix-20260909/train.parquet" \
  "$DATA_DIR/code_he_clean_rebuild/val.parquet"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing $path; run: sbatch training/scripts/setup_llada2_cluster.sh" >&2
    exit 2
  fi
done

case "$VARIANT" in
  baseline)
    experiment_name=llada2-mini-baseline-650
    variant_overrides=()
    ;;
  bptt)
    experiment_name="llada2-mini-bptt-h${BPTT_HORIZON}-650"
    variant_overrides=(
      --override ++actor_rollout_ref.actor.generation_consistency_jsd.bptt.enable=true
      --override "++actor_rollout_ref.model.enable_gradient_checkpointing=${BPTT_GRADIENT_CHECKPOINTING:-true}"
      --override "++actor_rollout_ref.actor.generation_consistency_jsd.bptt.checkpoint_projection=${BPTT_CHECKPOINT_PROJECTION:-true}"
      --override "++actor_rollout_ref.actor.generation_consistency_jsd.bptt.projection_chunk_size=${BPTT_PROJECTION_CHUNK_SIZE:-8}"
      --override "++actor_rollout_ref.actor.generation_consistency_jsd.stream_replay_chunk_size=${BPTT_REPLAY_CHUNK_SIZE:-1}"
      --override "++actor_rollout_ref.actor.fsdp_config.optimizer_offload=${BPTT_OPTIMIZER_OFFLOAD:-false}"
      --override ++actor_rollout_ref.actor.generation_consistency_jsd.bptt.profile_memory=true
      --override "++actor_rollout_ref.actor.generation_consistency_jsd.bptt.horizon=$BPTT_HORIZON"
      --override "++actor_rollout_ref.actor.generation_consistency_jsd.trace_all_ce_packing=${BPTT_PACKING:-all_blocks_per_round}"
      --override "++actor_rollout_ref.actor.generation_consistency_jsd.bptt.packing_max_blocks=${BPTT_PACKING_MAX_BLOCKS:-32}"
      --override ++actor_rollout_ref.actor.generation_consistency_jsd.trace_all_ce_packing_materialization=dense
      --override ++actor_rollout_ref.actor.generation_consistency_jsd.trace_all_ce_parallelism=none
    )
    ;;
  *)
    echo "VARIANT must be baseline or bptt, got: $VARIANT" >&2
    exit 2
    ;;
esac

RUN_DIR="$RUNS_DIR/$experiment_name"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$RUNTIME_ROOT/apptainer-cache}"

# The recipe is the paper LLaDA2.1-mini recipe. Only the requested training
# endpoint and the resource-count adaptations from 8 to 4 H100s are overridden
# for baseline; BPTT additionally enables activation/projection recomputation.
launcher=(
  python training/scripts/orl_train.py training/recipes/llada2_mini/orl.yaml
  --run-dir "$RUN_DIR"
  --set "MODEL_PATH=$MODEL_PATH"
  --set "TRAIN_FILE=$DATA_DIR/gsm8k-math-code_l0-mix-20260909/train.parquet"
  --set "VAL_FILE=$DATA_DIR/code_he_clean_rebuild/val.parquet"
  --sif "$SIF"
  --python "$VENV/bin/python3"
  --cache-dir "$RUNTIME_ROOT/cache"
  --gpus 0,1,2,3
  --logger console,wandb
  --steps "$TRAIN_STEPS"
  --override "++trainer.save_freq=${SAVE_FREQ:-50}"
  --env VERL_SDAR_GPU_REPLAY_RELAY_VISIBLE_DEVICES=0,1,2,3
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
  --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
  --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
  --override "++trainer.total_epochs=$TRAIN_STEPS"
  --override ++trainer.n_gpus_per_node=4
  --override ++actor_rollout_ref.actor.fsdp_config.fsdp_size=4
  --override ++actor_rollout_ref.rollout.agent.num_workers=4
  --override ++trainer.split_gpu_streaming.rollout_gpus=4
  --override ++trainer.split_gpu_streaming.actor_gpus=4
  --override ++ray_kwargs.ray_init.num_cpus=32
  --override ++ray_kwargs.ray_init.num_gpus=4
  --override ++ray_kwargs.ray_init.object_store_memory=120000000000
  --override ++trainer.resume_mode=auto
  --override ++trainer.initial_global_step=null
  "${variant_overrides[@]}"
)

echo "Launching $VARIANT in $RUN_DIR"
srun --ntasks=1 "${launcher[@]}" "$@"
