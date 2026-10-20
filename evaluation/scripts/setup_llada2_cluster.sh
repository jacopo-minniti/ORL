#!/bin/bash
#SBATCH --job-name=orl-eval-setup
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-rudner
#SBATCH --output=/scratch/jacopo04/ORL/.logs/%x-%j.out
#SBATCH --chdir=/scratch/jacopo04/ORL
set -euo pipefail
REPO_ROOT=/scratch/jacopo04/ORL
module --force purge
module load StdEnv/2023 python/3.11.5 apptainer/1.4.5
export ORL_EVAL_WORK="${ORL_EVAL_WORK:-$REPO_ROOT/.eval}"
export APPTAINER_CACHEDIR="$REPO_ROOT/.runtime/apptainer-cache"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export SSL_CERT_FILE=$(python3 -c 'import ssl; path = ssl.get_default_verify_paths().cafile; assert path, "No host CA bundle found"; print(path)')
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PIP_CERT="$SSL_CERT_FILE"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"
mkdir -p "$ORL_EVAL_WORK/images"
# Same digest as the paper's generation image; derive only the YAML dependency.
if [[ ! -f "$ORL_EVAL_WORK/images/sglang-gen.sif" ]]; then
  definition=$(mktemp)
  cat > "$definition" <<EOF
Bootstrap: localimage
From: $REPO_ROOT/.runtime/sglang-paper-cu129.sif

%post
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE
    export PIP_CERT=\$SSL_CERT_FILE
    python3 -m pip install --no-cache-dir PyYAML==6.0.3
EOF
  apptainer build "$ORL_EVAL_WORK/images/sglang-gen.sif.partial-${SLURM_JOB_ID}" "$definition"
  mv "$ORL_EVAL_WORK/images/sglang-gen.sif.partial-${SLURM_JOB_ID}" "$ORL_EVAL_WORK/images/sglang-gen.sif"
  rm "$definition"
fi
if [[ ! -f "$ORL_EVAL_WORK/images/python-scorer.sif" ]]; then
  apptainer build "$ORL_EVAL_WORK/images/python-scorer.sif.partial-${SLURM_JOB_ID}" \
    docker://docker.io/library/python@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0
  mv "$ORL_EVAL_WORK/images/python-scorer.sif.partial-${SLURM_JOB_ID}" "$ORL_EVAL_WORK/images/python-scorer.sif"
fi
cd "$REPO_ROOT/evaluation"
export PYTHONPATH="$PWD"
if [[ ! -f "$ORL_EVAL_WORK/runtime/llada/runtime_manifest.json" ]]; then
  python3 -m runtime.stage --family llada --out "$ORL_EVAL_WORK/runtime/llada.partial-${SLURM_JOB_ID}"
  mv "$ORL_EVAL_WORK/runtime/llada.partial-${SLURM_JOB_ID}" "$ORL_EVAL_WORK/runtime/llada"
fi
bash scripts/prepare.sh inputs scorer
echo "LLaDA evaluation environment ready: $ORL_EVAL_WORK"
