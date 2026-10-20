#!/bin/bash
#SBATCH --job-name=orl-llada2-setup
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/jacopo04/ORL/.logs/%x-%j.out
#SBATCH --error=/scratch/jacopo04/ORL/.logs/%x-%j.out
#SBATCH -D /scratch/jacopo04/ORL
#SBATCH --account=def-rudner

set -euo pipefail

REPO_ROOT=/scratch/jacopo04/ORL
RUNTIME_ROOT="${ORL_RUNTIME_ROOT:-$REPO_ROOT/.runtime}"
SIF="${ORL_SIF:-$RUNTIME_ROOT/sglang-paper-cu129.sif}"
VENV="${ORL_VENV:-$RUNTIME_ROOT/venv}"
MODELS_DIR="${ORL_MODELS_DIR:-$RUNTIME_ROOT/models}"
DATA_DIR="${ORL_DATA_DIR:-$RUNTIME_ROOT/data/ORL-training-data}"
ENV_FILE="${ORL_ENV_FILE:-$REPO_ROOT/../structured-diffusion/.env}"
MODEL_REVISION=20e64e2ad21644d0e5248586ed9c942cdd45de0f
IMAGE_URI='docker://docker.io/lmsysorg/sglang@sha256:065d403274324cc9dec4f830ca62540e6ac586746e1aed268c98a2b347d5853f'

cd "$REPO_ROOT"
mkdir -p "$RUNTIME_ROOT" "$MODELS_DIR" "$DATA_DIR" .logs

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

module --force purge
module load StdEnv/2023 apptainer/1.4.5

CONTAINER_CA=/etc/ssl/certs/ca-certificates.crt
container_exec=(
  apptainer exec
  -B "$REPO_ROOT:$REPO_ROOT"
  --env "SSL_CERT_FILE=$CONTAINER_CA"
  --env "REQUESTS_CA_BUNDLE=$CONTAINER_CA"
  --env "CURL_CA_BUNDLE=$CONTAINER_CA"
  --env "PIP_CERT=$CONTAINER_CA"
  "$SIF"
)

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$RUNTIME_ROOT/apptainer-cache}"
mkdir -p "$APPTAINER_CACHEDIR"

if [[ ! -s "$SIF" ]]; then
  partial_sif="${SIF}.partial.${SLURM_JOB_ID:-$$}"
  trap 'rm -f "$partial_sif"' EXIT
  apptainer build "$partial_sif" "$IMAGE_URI"
  mv -f "$partial_sif" "$SIF"
  trap - EXIT
fi

if [[ ! -x "$VENV/bin/python3" ]]; then
  "${container_exec[@]}" python3 -m venv --system-site-packages "$VENV"
fi

"${container_exec[@]}" \
  "$VENV/bin/python3" -m pip install --upgrade pip
"${container_exec[@]}" \
  "$VENV/bin/python3" -m pip install -r training/env/requirements-venv.txt

# Downloads happen during setup; the actual training recipe is deliberately offline.
unset HF_HUB_OFFLINE
export HF_HOME="$RUNTIME_ROOT/cache/hf_home"
mkdir -p "$HF_HOME"

hf_download() {
  local repo_id=$1 repo_type=$2 revision=$3 local_dir=$4
  "${container_exec[@]}" env "HF_HOME=$HF_HOME" \
    "$VENV/bin/python3" -c \
    'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], repo_type=sys.argv[2] or None, revision=sys.argv[3] or None, local_dir=sys.argv[4])' \
    "$repo_id" "$repo_type" "$revision" "$local_dir"
}

if [[ ! -f "$DATA_DIR/SHA256SUMS" ]]; then
  hf_download yifanyu/ORL-training-data dataset '' "$DATA_DIR"
fi

MODEL_SNAPSHOT="$MODELS_DIR/LLaDA2.1-mini"
MODEL_ACTOR="$MODELS_DIR/LLaDA2.1-mini-actor"
if [[ ! -f "$MODEL_SNAPSHOT/config.json" ]]; then
  hf_download inclusionAI/LLaDA2.1-mini model "$MODEL_REVISION" \
    "$MODEL_SNAPSHOT"
fi

if [[ ! -f "$MODEL_ACTOR/modeling_llada2_moe.stock_backup.py" ]] || \
  ! cmp -s training/overlays/llada2/modeling_llada2_moe.py \
    "$MODEL_ACTOR/modeling_llada2_moe.py"; then
  bash training/tools/prepare_llada2_actor.sh "$MODEL_SNAPSHOT" "$MODEL_ACTOR"
fi

(cd "$DATA_DIR" && sha256sum --quiet -c SHA256SUMS)
"${container_exec[@]}" \
  "$VENV/bin/python3" -c \
  'import ray, tensordict, torch, transformers, wandb; print("runtime", torch.__version__, transformers.__version__, ray.__version__, wandb.__version__)'

echo "LLaDA2 runtime ready"
echo "SIF=$SIF"
echo "VENV=$VENV"
echo "MODEL_PATH=$MODEL_ACTOR"
echo "DATA_DIR=$DATA_DIR"
