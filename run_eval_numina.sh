#!/usr/bin/env bash
#SBATCH --job-name=numina-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=80G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=outputs/slurm/numina-eval-%j.out
#SBATCH --error=outputs/slurm/numina-eval-%j.err

# pass@k evaluation of a saved checkpoint on NuminaMath-CoT (\boxed{} answers).
# Override settings via environment variables, e.g.:
#   checkpoint_path=outputs/<run_name> ./run_eval_numina.sh
# Carries an optional SBATCH header so it can also be submitted with sbatch.
set -eo pipefail

mkdir -p outputs/slurm

# ============================================================================
# CONFIGURATION
# ============================================================================

checkpoint_path=${checkpoint_path:-""}
k=${k:-"8"}
n_samples=${n_samples:-32}
temperature=${temperature:-0.8}
top_p=${top_p:-0.95}
max_new_tokens=${max_new_tokens:-1024}
max_eval_samples=${max_eval_samples:-""}
batch_size=${batch_size:-8}
dataset_name=${dataset_name:-"AI-MO/NuminaMath-CoT"}
dataset_config=${dataset_config-"default"}   # `-` (not `:-`) so empty stays empty
eval_split=${eval_split:-"test"}            
prompt_column=${prompt_column:-"problem"}
response_column=${response_column:-"solution"}
use_chat_template=${use_chat_template:-""}
device=${device:-"cuda"}

# ============================================================================
# VALIDATION
# ============================================================================

if [ -z "$checkpoint_path" ]; then
    echo "[ERROR] checkpoint_path is required. Set it via environment variable."
    echo "Usage: checkpoint_path=outputs/my_model sbatch run_eval_numina.sh"
    exit 1
fi

echo "============================================================================"
echo "[INFO] NuminaMath-CoT pass@k Evaluation"
echo "============================================================================"
echo "[INFO] Checkpoint:       $checkpoint_path"
echo "[INFO] Dataset:          $dataset_name (config=$dataset_config, split=$eval_split)"
echo "[INFO] Columns:          prompt=$prompt_column, response=$response_column"
echo "[INFO] k values:         $k"
echo "[INFO] n_samples:        $n_samples"
echo "[INFO] temperature:      $temperature"
echo "[INFO] batch_size:       $batch_size"
echo "[INFO] max_new_tokens:   $max_new_tokens"
echo "============================================================================"

# ============================================================================
# BUILD COMMAND
# ============================================================================

PYTHON="${PYTHON:-python}"
CMD="${PYTHON} eval_gsm8k.py --checkpoint_path ${checkpoint_path} --k ${k} --n_samples ${n_samples}"
CMD="${CMD} --temperature ${temperature} --top_p ${top_p} --max_new_tokens ${max_new_tokens}"
CMD="${CMD} --batch_size ${batch_size} --dataset_name ${dataset_name}"
CMD="${CMD} --eval_split ${eval_split} --prompt_column ${prompt_column} --response_column ${response_column}"
CMD="${CMD} --device ${device}"

[ -n "${dataset_config}" ] && CMD="${CMD} --dataset_config ${dataset_config}"
[ -n "${max_eval_samples}" ] && CMD="${CMD} --max_eval_samples ${max_eval_samples}"
[ -n "${use_chat_template}" ] && CMD="${CMD} --use_chat_template"

echo "[INFO] Running: $CMD"
eval $CMD

echo "[INFO] Evaluation complete for: $checkpoint_path"
