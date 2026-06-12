#!/usr/bin/env bash
# Simple submitter for GSM-8K finetuning sweeps.

set -euo pipefail
info() { echo "[INFO] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SWEEP="${SWEEP:-1}"
SUBMIT_TRAIN="${SUBMIT_TRAIN:-1}"
LOCAL_RUN="${LOCAL_RUN:-0}"
SBATCH_ARGS_BASE="${SBATCH_ARGS_BASE:-${SBATCH_ARGS:-}}"
RUN_NAME_BASE="${RUN_NAME_BASE:-gsm8k}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/slurm}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/run_gsm8k.sh}"
SWEEP_PARAMS_YAML="${SWEEP_PARAMS_YAML:-$REPO_ROOT/config/sweep_params_gsm8k.yaml}"

_is_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

_build_run_tag() {
  local base_name="$1"
  shift
  local -a args=("$@")
  local params_str=""
  local i
  for ((i=0; i<${#args[@]}; i+=2)); do
    local name="${args[$i]}"
    local value="${args[$i+1]}"
    params_str="${params_str}${name}=${value};"
  done
  local hash
  hash="$(echo -n "$params_str" | md5sum | cut -c1-12)"
  echo "${base_name}_${hash}"
}

_parse_sbatch_args() {
  local -a out=()
  local combined=""
  if [ -n "${SBATCH_ARGS_BASE:-}" ]; then
    combined="${SBATCH_ARGS_BASE}"
  fi
  if [ -n "${SBATCH_ARGS_EXTRA:-}" ]; then
    if [ -n "$combined" ]; then
      combined="${combined} ${SBATCH_ARGS_EXTRA}"
    else
      combined="${SBATCH_ARGS_EXTRA}"
    fi
  fi
  if [ -n "$combined" ]; then
    while IFS= read -r tok; do
      [ -n "$tok" ] || continue
      out+=("$tok")
    done < <(SBATCH_ARGS_COMBINED="$combined" python3 - <<'PY'
import os, shlex
s = os.environ.get("SBATCH_ARGS_COMBINED", "") or ""
for tok in shlex.split(s):
    print(tok)
PY
    )
  fi
  printf '%s\n' "${out[@]}"
}


submit_run() {
  local run_tag="$1"
  local train_job="(skipped)"

  mkdir -p "$LOG_DIR/$run_tag"
  [ -f "$TRAIN_SCRIPT" ] || die "Missing train script: $TRAIN_SCRIPT"

  if _is_true "$LOCAL_RUN"; then
    info "Running locally (LOCAL_RUN=1) for tag: $run_tag"
    "$TRAIN_SCRIPT"
    return
  fi

  if _is_true "$SUBMIT_TRAIN"; then
    local -a extra_args=()
    while IFS= read -r tok; do
      [ -n "$tok" ] || continue
      extra_args+=("$tok")
    done < <(_parse_sbatch_args)

    if [ "${#extra_args[@]}" -gt 0 ]; then
      info "Train sbatch overrides: ${extra_args[*]}"
    fi

    train_job="$(sbatch --export=ALL --parsable \
      --job-name="gsm8k_${run_tag}" \
      --output="$LOG_DIR/${run_tag}/train-%j.out" \
      --error="$LOG_DIR/${run_tag}/train-%j.err" \
      "${extra_args[@]}" \
      "$TRAIN_SCRIPT")"

    [ -n "$train_job" ] || die "Train submission failed"
    info "Train job id: $train_job"
  else
    info "Skipping train submission (SUBMIT_TRAIN=0)"
  fi
}

_process_yaml_run() {
  local -a tag_parts=()
  local override_tag=""
  export SBATCH_ARGS_EXTRA=""

  # Set defaults for run_name generation (matching run_gsm8k.sh)
  local _model_name_or_path="HuggingFaceTB/SmolLM2-135M"
  local _muon_backend="halfpower"
  local _muon_exponent="1/3"
  local _learning_rate="2e-5"
  local _run_name_postfix=""
  local _dataset_name=""

  local -a control_params=("RUN_TAG" "SUBMIT_TRAIN" "LOCAL_RUN" "SBATCH_ARGS" "WANDB_RESUME_RUN_ID")
  local i=1
  while [ "$i" -le "$#" ]; do
    local param_name="${!i}"
    i=$((i + 1))
    local param_value="${!i}"
    i=$((i + 1))

    if [ "$param_name" = "RUN_TAG" ]; then
      override_tag="$param_value"
      continue
    fi

    if [ "$param_name" = "WANDB_RESUME_RUN_ID" ]; then
      export WANDB_RUN_ID="$param_value"
      export WANDB_RESUME="${WANDB_RESUME:-allow}"
      continue
    fi

    if [ "$param_name" = "SBATCH_ARGS" ]; then
      export SBATCH_ARGS_EXTRA="$param_value"
      continue
    fi

    # Export parameter as environment variable (lowercase for run_gsm8k.sh)
    local lower_name=$(echo "$param_name" | tr '[:upper:]' '[:lower:]')
    export "$lower_name"="$param_value"

    # Track values for run_name generation
    case "$lower_name" in
      model_name_or_path) _model_name_or_path="$param_value" ;;
      muon_backend) _muon_backend="$param_value" ;;
      muon_exponent) _muon_exponent="$param_value" ;;
      learning_rate) _learning_rate="$param_value" ;;
      run_name_postfix) _run_name_postfix="$param_value" ;;
      dataset_name) _dataset_name="$param_value" ;;
    esac

    local is_control=0
    local control_param
    for control_param in "${control_params[@]}"; do
      if [ "$param_name" = "$control_param" ]; then
        is_control=1
        break
      fi
    done
    if [ "$is_control" -eq 0 ]; then
      tag_parts+=("$param_name" "$param_value")
    fi
  done

  # Generate run_name in the same format as run_gsm8k.sh
  local exponent_clean=$(echo "${_muon_exponent}" | tr '/' '-')
  local model_slug=$(echo "${_model_name_or_path}" | sed 's|.*/||')
  local dataset_lower=$(echo "${_dataset_name}" | tr '[:upper:]' '[:lower:]')
  local run_name_prefix="gsm8k"
  case "$dataset_lower" in
    *numina*) run_name_prefix="numina" ;;
    *code*) run_name_prefix="code" ;;
    *gsm8k*) run_name_prefix="gsm8k" ;;
  esac
  local generated_run_name="${run_name_prefix}_${model_slug}_${_muon_backend}_exp${exponent_clean}_lr${_learning_rate}"
  if [ -n "$_run_name_postfix" ]; then
    generated_run_name="${generated_run_name}_${_run_name_postfix}"
  fi
  export run_name="$generated_run_name"

  local tag=""
  if [ -n "$override_tag" ]; then
    tag="$override_tag"
  else
    tag="$generated_run_name"
  fi

  submit_run "$tag"
}

if [ "$SWEEP" = "1" ] || [ "$SWEEP" = "true" ] || [ "$SWEEP" = "True" ]; then
  [ -f "$SWEEP_PARAMS_YAML" ] || die "Missing sweep params YAML: $SWEEP_PARAMS_YAML"

  temp_runs="$(mktemp)"
  python3 <<PYTHON_EOF > "$temp_runs" || die "Failed to parse YAML"
import yaml
import sys

try:
    with open("$SWEEP_PARAMS_YAML", 'r') as f:
        data = yaml.safe_load(f)

    if 'runs' not in data or not isinstance(data['runs'], list):
        print("Error: YAML must have a 'runs' key with a list of runs", file=sys.stderr)
        sys.exit(1)

    for run in data['runs']:
        if not isinstance(run, dict):
            continue
        for param_name, param_value in sorted(run.items()):
            print(param_name)
            print(str(param_value))
        print()
except Exception as e:
    print(f"Error parsing YAML: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

  run_args=()
  run_count=0
  while IFS= read -r line; do
    if [ -z "$line" ]; then
      if [ ${#run_args[@]} -gt 0 ]; then
        run_count=$((run_count + 1))
        (_process_yaml_run "${run_args[@]}")
        run_args=()
      fi
    else
      param_name="$line"
      IFS= read -r param_value || param_value=""
      if [ -n "$param_name" ] && [ -n "$param_value" ]; then
        run_args+=("$param_name" "$param_value")
      fi
    fi
  done < "$temp_runs"

  if [ ${#run_args[@]} -gt 0 ]; then
    run_count=$((run_count + 1))
    (_process_yaml_run "${run_args[@]}")
  fi

  rm -f "$temp_runs"
  info "Sweep submission complete. Processed $run_count run(s)."
else
  run_tag="${RUN_TAG:-run}"
  submit_run "$run_tag"
fi
