#!/bin/bash
# Submit from the login node; resource count and worker count always agree.
# Usage: bash evaluation/scripts/submit_llada2_eval.sh CHECKPOINT [NUM_GPUS=1] [BENCHMARKS=he]
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
checkpoint=${1:?Usage: submit_llada2_eval.sh CHECKPOINT [NUM_GPUS] ["he gsm8k math500 mbpp"]}
checkpoint=$(realpath -e "$checkpoint")
num_gpus=${2:-1}
benchmarks=${3:-he}
[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] && (( num_gpus <= 8 )) || { echo 'NUM_GPUS must be 1..8' >&2; exit 2; }
benchmarks=${benchmarks//,/ }
benchmarks=${benchmarks//humaneval/he}
for benchmark in $benchmarks; do
  case "$benchmark" in he|gsm8k|math500|mbpp) ;; *) echo "Unknown benchmark: $benchmark" >&2; exit 2 ;; esac
done
[[ -n "$benchmarks" ]] || { echo 'Select at least one benchmark' >&2; exit 2; }
[[ -d "$checkpoint/actor" ]] && checkpoint="$checkpoint/actor"
if [[ ! -f "$checkpoint/fsdp_config.json" && ! -f "$checkpoint/config.json" ]]; then
  echo "Expected a global_step_N directory, FSDP actor directory, or HF export: $checkpoint" >&2; exit 2
fi
export ORL_EVAL_WORK="${ORL_EVAL_WORK:-$repo/.eval}"
mkdir -p "$repo/.logs" "$ORL_EVAL_WORK"
dependency=()
if [[ ! -f "$ORL_EVAL_WORK/images/sglang-gen.sif" || ! -f "$ORL_EVAL_WORK/images/python-scorer.sif" ||
      ! -f "$ORL_EVAL_WORK/runtime/llada/runtime_manifest.json" || ! -f "$ORL_EVAL_WORK/scoring/runtime/staging_receipt.json" ]]; then
  # Share a single pending/running setup across concurrent submissions.
  exec 8> "$ORL_EVAL_WORK/setup.lock"
  flock 8
  setup_id=''
  if [[ -f "$ORL_EVAL_WORK/setup_job_id" ]]; then
    previous=$(cat "$ORL_EVAL_WORK/setup_job_id")
    state=$(squeue -h -j "$previous" -o '%T' 2>/dev/null || true)
    case "$state" in PENDING|RUNNING|CONFIGURING) setup_id=$previous ;; esac
  fi
  if [[ -z "$setup_id" ]]; then
    setup_id=$(sbatch --parsable "$repo/evaluation/scripts/setup_llada2_cluster.sh")
    setup_id=${setup_id%%;*}
    echo "$setup_id" > "$ORL_EVAL_WORK/setup_job_id"
  fi
  dependency=(--dependency="afterok:$setup_id" --kill-on-invalid-dep=yes)
  echo "Evaluation setup job: $setup_id"
  flock -u 8
fi
if [[ -f "$checkpoint/fsdp_config.json" ]]; then
  merge_id=$(sbatch --parsable "$repo/evaluation/scripts/merge_llada2_cluster.sh" "$checkpoint")
  merge_id=${merge_id%%;*}
  echo "CPU merge job: $merge_id (200 GiB, no GPU)"
  if (( ${#dependency[@]} )); then
    dependency[0]="${dependency[0]}:$merge_id"
  else
    dependency=(--dependency="afterok:$merge_id" --kill-on-invalid-dep=yes)
  fi
fi
job_id=$(sbatch --parsable --gpus-per-node="h100:$num_gpus" --time="${EVAL_TIME:-02:00:00}" --mem="${EVAL_MEM:-40G}" \
  "${dependency[@]}" "$repo/evaluation/scripts/eval_llada2_cluster.sh" "$checkpoint" "$num_gpus" "$benchmarks")
echo "Evaluation job: ${job_id%%;*} ($num_gpus H100, ${EVAL_TIME:-02:00:00}, benchmarks: $benchmarks)"
echo "Log: $repo/.logs/orl-llada2-eval-${job_id%%;*}.out"
