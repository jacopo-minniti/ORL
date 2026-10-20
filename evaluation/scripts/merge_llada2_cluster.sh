#!/bin/bash
#SBATCH --job-name=orl-llada2-merge
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --account=def-rudner
#SBATCH --output=/scratch/jacopo04/ORL/.logs/%x-%j.out
#SBATCH --chdir=/scratch/jacopo04/ORL
# CPU-only; merge any number of checkpoints sequentially to bound peak RAM.
# Usage: sbatch merge_llada2_cluster.sh CHECKPOINT [CHECKPOINT ...]
set -euo pipefail
REPO_ROOT=/scratch/jacopo04/ORL
(( $# > 0 )) || { echo 'Supply at least one FSDP checkpoint' >&2; exit 2; }
module --force purge
module load StdEnv/2023 python/3.11.5 apptainer/1.4.5
export ORL_EVAL_WORK="${ORL_EVAL_WORK:-$REPO_ROOT/.eval}"
export CUDA_VISIBLE_DEVICES=''
mkdir -p "$ORL_EVAL_WORK/models" "$ORL_EVAL_WORK/locks"
for source_checkpoint in "$@"; do
  checkpoint=$(realpath -e "$source_checkpoint")
  [[ ! -d "$checkpoint/actor" ]] || checkpoint="$checkpoint/actor"
  [[ -f "$checkpoint/fsdp_config.json" ]] || { echo "Not an FSDP actor checkpoint: $checkpoint" >&2; exit 2; }
  identity=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])' "$checkpoint")
  label=$(basename "$(dirname "$checkpoint")")
  [[ "$(basename "$checkpoint")" = actor ]] || label=$(basename "$checkpoint")
  label=${label//[^a-zA-Z0-9._-]/-}
  name="llada-orl-$label-$identity"
  model="$ORL_EVAL_WORK/models/$name"
  exec 9> "$ORL_EVAL_WORK/locks/$name.lock"
  flock 9
  if [[ -f "$model/export_report.json" && -f "$model/model.safetensors.index.json" ]]; then
    echo "Already exported: $model"
  else
    echo "Merging $checkpoint -> $model (CPU only)"
    apptainer exec -B /scratch "$REPO_ROOT/.runtime/sglang-paper-cu129.sif" \
      env CUDA_VISIBLE_DEVICES= PYTHONPATH="$REPO_ROOT/training/verl" \
      "$REPO_ROOT/.runtime/venv/bin/python3" "$REPO_ROOT/training/tools/export_hf.py" \
      --actor-dir "$checkpoint" --target "$model" \
      --reference "$REPO_ROOT/.runtime/models/LLaDA2.1-mini" --llada2-base-reference
  fi
  flock -u 9
done
echo 'All requested checkpoint exports are ready.'
