#!/bin/bash
#SBATCH --job-name=orl-llada2-eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --account=def-rudner
#SBATCH --output=/scratch/jacopo04/ORL/.logs/%x-%j.out
#SBATCH --chdir=/scratch/jacopo04/ORL
# Usage: sbatch [--gpus-per-node=h100:N] this_script CHECKPOINT [N=1] ["he gsm8k math500 mbpp"=he]
# CHECKPOINT is a global_step_N directory, its actor directory, or an HF export.
set -euo pipefail
REPO_ROOT=/scratch/jacopo04/ORL
checkpoint=${1:?Usage: sbatch eval_llada2_cluster.sh CHECKPOINT [NUM_GPUS] [BENCHMARKS]}
num_gpus=${2:-1}
benchmarks=${3:-he}
[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_GPUS must be positive' >&2; exit 2; }
benchmarks=${benchmarks//,/ }
benchmarks=${benchmarks//humaneval/he}
for benchmark in $benchmarks; do
  case "$benchmark" in he|gsm8k|math500|mbpp) ;; *) echo "Unknown benchmark: $benchmark" >&2; exit 2 ;; esac
done
[[ -n "$benchmarks" ]] || { echo 'Select at least one benchmark' >&2; exit 2; }
checkpoint=$(realpath -e "$checkpoint")
[[ ! -d "$checkpoint/actor" ]] || checkpoint="$checkpoint/actor"
module --force purge
module load StdEnv/2023 python/3.11.5 apptainer/1.4.5
export ORL_EVAL_WORK="${ORL_EVAL_WORK:-$REPO_ROOT/.eval}"
export ORL_EVAL_GPU_BINDING=slurm
export ORL_EVAL_SCORE_ENGINE=apptainer-rlimits
export BENCHMARKS="$benchmarks"
export PYTHONDONTWRITEBYTECODE=1
# Use only Slurm-allocated device indices (Slurm may expose nonzero physical IDs).
allocated=${CUDA_VISIBLE_DEVICES:?Run this script inside a Slurm GPU allocation}
IFS=, read -ra devices <<< "$allocated"
(( ${#devices[@]} >= num_gpus )) || { echo "Requested $num_gpus eval workers but Slurm exposed $allocated" >&2; exit 2; }
GPUS=$(IFS=,; echo "${devices[*]:0:num_gpus}")
export GPUS
for required in images/sglang-gen.sif images/python-scorer.sif runtime/llada/runtime_manifest.json scoring/runtime/staging_receipt.json; do
  [[ -f "$ORL_EVAL_WORK/$required" ]] || { echo "Missing $required; first sbatch evaluation/scripts/setup_llada2_cluster.sh" >&2; exit 2; }
done
# Fail before loading weights if the node cannot provide scoring isolation.
unshare --user --map-user=65534 --map-group=65534 --net -- true
identity=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])' "$checkpoint")
label=$(basename "$(dirname "$checkpoint")")
[[ "$(basename "$checkpoint")" = actor ]] || label=$(basename "$checkpoint")
label=${label//[^a-zA-Z0-9._-]/-}
name="llada-orl-$label-$identity"
model="$ORL_EVAL_WORK/models/$name"
mkdir -p "$ORL_EVAL_WORK/models" "$ORL_EVAL_WORK/locks" "$ORL_EVAL_WORK/submissions"
# flock serializes concurrent evaluations/exports of this checkpoint.
exec 9> "$ORL_EVAL_WORK/locks/$name.lock"
flock 9
if [[ ! -d "$model" ]]; then
  if [[ -f "$checkpoint/fsdp_config.json" ]]; then
    echo "Checkpoint must be merged in a CPU job first: $checkpoint" >&2
    echo 'Use submit_llada2_eval.sh to submit the merge and evaluation with dependencies.' >&2
    exit 2
  elif [[ -f "$checkpoint/config.json" ]] && compgen -G "$checkpoint/*.safetensors" >/dev/null; then
    # Materialize symlinks once so the paper recipe can inventory all weights.
    cp -rL "$checkpoint" "$model.partial"
    mv "$model.partial" "$model"
  else
    echo "Not an FSDP actor checkpoint or HF safetensors export: $checkpoint" >&2; exit 2
  fi
fi
echo "Evaluating $checkpoint as $name; GPUs=$GPUS benchmarks=$BENCHMARKS"
python3 - "$ORL_EVAL_WORK" "$name" "$checkpoint" "$GPUS" "$benchmarks" <<'PY'
import json, os, pathlib, sys
root, name, checkpoint, gpus, benchmarks = sys.argv[1:]
pathlib.Path(root, 'submissions', name + '-' + os.environ['SLURM_JOB_ID'] + '.json').write_text(json.dumps({
    'checkpoint': checkpoint, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
    'gpus': gpus, 'benchmarks': benchmarks.split(), 'role': 'llada-orl',
    'seed': 12345, 'sampling_temperature': 0, 'prob_threshold': 0.85,
    'max_new_tokens': 4096, 'score_engine': 'apptainer-rlimits',
}, indent=2) + '\n')
PY
bash "$REPO_ROOT/evaluation/scripts/main_table.sh" llada-orl "$model" "$name"
python3 "$REPO_ROOT/evaluation/scripts/summarize.py" --name "$name" \
  --out "$ORL_EVAL_WORK/$name.summary.json"
echo "Results: $ORL_EVAL_WORK/$name.summary.json"
