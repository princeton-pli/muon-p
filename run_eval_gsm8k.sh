#!/usr/bin/env bash
#SBATCH --job-name=gsm8k-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=80G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=outputs/slurm/gsm8k-eval-%j.out
#SBATCH --error=outputs/slurm/gsm8k-eval-%j.err

# pass@k evaluation of a saved checkpoint on GSM-8K (or any GSM-8K-style
# dataset). Override settings via environment variables, e.g.:
#   checkpoint_path=outputs/<run_name> ./run_eval_gsm8k.sh
# Carries an optional SBATCH header so it can also be submitted with sbatch.
set -eo pipefail

mkdir -p outputs/slurm

# ============================================================================
# CONFIGURATION
# ============================================================================

checkpoint_path=${checkpoint_path:-""}
k=${k:-"8"}
n_samples=${n_samples:-10}
temperature=${temperature:-0.8}
top_p=${top_p:-0.95}
max_new_tokens=${max_new_tokens:-512}
max_eval_samples=${max_eval_samples:-""}
batch_size=${batch_size:-8}
dataset_name=${dataset_name:-"openai/gsm8k"}
use_chat_template=${use_chat_template:-""}
device=${device:-"cuda"}
eval_seed=${eval_seed:-42}   # seed generation-sampling RNG for reproducible pass@k

# Dataset-specific defaults (case-insensitive on dataset_name), mirroring
# the auto-config logic in run_gsm8k.sh. Only sets vars not already set by
# the caller; explicit overrides win.
dataset_name_lc=$(echo "${dataset_name}" | tr '[:upper:]' '[:lower:]')
case "${dataset_name_lc}" in
    *numina*|*math*)
        : "${dataset_config:=default}"
        : "${eval_split:=train}"
        : "${prompt_column:=problem}"
        : "${response_column:=solution}"
        : "${eval_split:=test}"
        : "${split_seed:=42}"
        ;;
    *code*)
        : "${dataset_config:=train}"
        : "${eval_split:=train}"
        : "${prompt_column:=input}"
        : "${response_column:=output}"
        : "${eval_holdout_size:=500}"
        : "${split_seed:=42}"
        ;;
esac

dataset_config=${dataset_config:-"main"}
eval_split=${eval_split:-"test"}
prompt_column=${prompt_column:-"question"}
response_column=${response_column:-"answer"}
eval_holdout_size=${eval_holdout_size:-0}
split_seed=${split_seed:-42}

# ============================================================================
# VALIDATION
# ============================================================================

if [ -z "$checkpoint_path" ]; then
    echo "[ERROR] checkpoint_path is required. Set it via environment variable."
    echo "Usage: checkpoint_path=outputs/my_model sbatch run_eval_gsm8k.sh"
    exit 1
fi

echo "============================================================================"
echo "[INFO] pass@k Evaluation"
echo "============================================================================"
echo "[INFO] Checkpoint:        $checkpoint_path"
echo "[INFO] Dataset:           $dataset_name (config=$dataset_config, split=$eval_split)"
echo "[INFO] Columns:           prompt=$prompt_column, response=$response_column"
echo "[INFO] Holdout:           eval_holdout_size=$eval_holdout_size (seed=$split_seed)"
echo "[INFO] k values:          $k"
echo "[INFO] n_samples:         $n_samples"
echo "[INFO] temperature:       $temperature"
echo "[INFO] batch_size:        $batch_size"
echo "[INFO] max_new_tokens:    $max_new_tokens"
echo "[INFO] eval_seed:         ${eval_seed:-<unset, fresh sampling>}"
echo "============================================================================"

# ============================================================================
# BUILD COMMAND
# ============================================================================

PYTHON="${PYTHON:-python}"
CMD="${PYTHON} eval_gsm8k.py --checkpoint_path ${checkpoint_path} --k ${k} --n_samples ${n_samples}"
CMD="${CMD} --temperature ${temperature} --top_p ${top_p} --max_new_tokens ${max_new_tokens}"
CMD="${CMD} --batch_size ${batch_size} --dataset_name ${dataset_name} --dataset_config ${dataset_config}"
CMD="${CMD} --eval_split ${eval_split} --device ${device}"
CMD="${CMD} --prompt_column ${prompt_column} --response_column ${response_column}"
CMD="${CMD} --eval_holdout_size ${eval_holdout_size} --split_seed ${split_seed}"

[ -n "${max_eval_samples}" ] && CMD="${CMD} --max_eval_samples ${max_eval_samples}"
[ -n "${use_chat_template}" ] && CMD="${CMD} --use_chat_template"
[ -n "${eval_seed}" ] && CMD="${CMD} --eval_seed ${eval_seed}"

echo "[INFO] Running: $CMD"
eval $CMD

echo "[INFO] Evaluation complete for: $checkpoint_path"
