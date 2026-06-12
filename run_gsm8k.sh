#!/usr/bin/env bash
#SBATCH --job-name=gsm8k-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=60G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/gsm8k-%j.out
#SBATCH --error=outputs/slurm/gsm8k-%j.err

# Supervised finetuning on math / reasoning datasets (GSM-8K, NuminaMath,
# MATH-500, code, ...) with the Muon + AdamW dual optimizer.
#
# Runs train_gsm8k.py via torchrun. Every variable below can be overridden from
# the environment (this is how submit_gsm8k.sh launches sweeps), e.g.:
#
#   muon_backend=newtonschulz5 dataset_name=openai/gsm8k ./run_gsm8k.sh
#
# This file carries an optional SBATCH header so it can also be submitted with
# `sbatch run_gsm8k.sh` on a Slurm cluster; it runs fine as a plain script too.

set -eo pipefail
export WANDB_MODE="${WANDB_MODE:-online}"

# ============================================================================
# PATHS AND SETUP
# ============================================================================

mkdir -p outputs/slurm

num_gpus=${num_gpus:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
echo "[INFO] Number of GPUs: $num_gpus"

# Point PYTHON at the interpreter for your environment (see README / setup.sh).
PYTHON="${PYTHON:-python}"
master_port=${master_port:-$(${PYTHON} -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}

# ============================================================================
# MODEL
# ============================================================================

model_name_or_path=${model_name_or_path:-"HuggingFaceTB/SmolLM2-135M"}
from_pretrained=${from_pretrained:-True}
tokenizer_name=${tokenizer_name:-""}
num_hidden_layers=${num_hidden_layers:-""}

# ============================================================================
# DATASET
# ============================================================================

# dataset_name controls both the dataset loaded and (in train_gsm8k.py) the
# answer-extraction style at generation-eval time:
#   *gsm8k*    -> '####' / '###' marker extraction
#   *numina*   -> \boxed{...} extraction
#   *math*     -> \boxed{...} extraction (covers NuminaMath, MATH-500, etc.)
#   *code*     -> ``` fenced code-block extraction
dataset_name=${dataset_name:-"openai/gsm8k"}

# Auto-apply dataset-specific defaults based on dataset_name (case-insensitive).
# Only fills in values the caller hasn't already set; explicit overrides win.
dataset_name_lc=$(echo "${dataset_name}" | tr '[:upper:]' '[:lower:]')
case "${dataset_name_lc}" in
    *numina*|*math*)
        : "${prompt_column:=problem}"
        : "${response_column:=solution}"
        : "${max_seq_length:=1024}"
        : "${max_new_tokens:=512}"
        : "${eval_holdout_size:=500}"
        : "${train_split:=train}"
        : "${eval_split:=test}"  # not actually loaded when eval_holdout_size>0
        dataset_config="default"
        ;;
    *code*)
        : "${prompt_column:=input}"
        : "${response_column:=output}"
        : "${max_seq_length:=1024}"
        : "${max_new_tokens:=512}"
        : "${eval_holdout_size:=500}"
        : "${train_split:=train}"
        : "${eval_split:=train}"  # not actually loaded when eval_holdout_size>0
        : "${dataset_config:=train}"
        ;;
esac

dataset_config=${dataset_config:-"main"}   # NOTE: `-` (not `:-`) so empty string is preserved
train_split=${train_split:-"train"}
eval_split=${eval_split:-"test"}
max_seq_length=${max_seq_length:-512}
prompt_column=${prompt_column:-"question"}
response_column=${response_column:-"answer"}
eval_holdout_size=${eval_holdout_size:-""}
train_tokenized_cache=${train_tokenized_cache:-""}
eval_tokenized_cache=${eval_tokenized_cache:-""}
use_chat_template=${use_chat_template:-False}
max_eval_samples=${max_eval_samples:--1}

# ============================================================================
# MUON HYPERPARAMETERS
# ============================================================================

muon_backend=${muon_backend:-halfpower}
muon_exponent=${muon_exponent:-1/3}
muon_lr_factor=${muon_lr_factor:-0.1}
muon_momentum=${muon_momentum:-0.95}
muon_backend_steps=${muon_backend_steps:-6}
halfpower_c=${halfpower_c:-0.66}
muon_curriculum=${muon_curriculum:-False}
muon_curriculum_switch_step=${muon_curriculum_switch_step:-""}
muon_curriculum_target_backend=${muon_curriculum_target_backend:-halfpower}
muon_curriculum_target_exponent=${muon_curriculum_target_exponent:-1/3}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

learning_rate=${learning_rate:-2e-5}
num_train_epochs=${num_train_epochs:-3}
max_steps=${max_steps:-3000}
per_device_train_batch_size=${per_device_train_batch_size:-64}
per_device_eval_batch_size=${per_device_eval_batch_size:-128}
total_batch_size=${total_batch_size:-128}
gradient_accumulation_steps="${gradient_accumulation_steps:-$((total_batch_size / num_gpus / per_device_train_batch_size))}"
warmup_steps=${warmup_steps:-300}
warmup_ratio=${warmup_ratio:-0.1}
weight_decay=${weight_decay:-0.01}
logging_steps=${logging_steps:-300}
eval_strategy=${eval_strategy:-steps}
eval_steps=${eval_steps:-1000}
save_strategy=${save_strategy:-steps}
save_steps=${save_steps:-1000}
seed=${seed:-42}
min_lr_rate=${min_lr_rate:-0.5}
max_new_tokens=${max_new_tokens:-128}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

exponent_clean=$(echo ${muon_exponent} | tr '/' '-')
model_slug=$(echo ${model_name_or_path} | sed 's|.*/||')
run_name_postfix=${run_name_postfix:-""}
run_name=${run_name:-"gsm8k_${model_slug}_${muon_backend}_exp${exponent_clean}_lr${learning_rate}${run_name_postfix:+_${run_name_postfix}}"}
output_dir=${output_dir:-"outputs/${run_name}"}
output_file="${output_dir}/train.log"
project_name=${project_name:-"gsm8k-finetune"}

mkdir -p ${output_dir}

# ============================================================================
# BUILD OPTIONAL ARGS
# ============================================================================

OPTIONAL_ARGS=""
[ "$num_gpus" -gt 1 ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --disable_tqdm True"
[ -n "${tokenizer_name}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --tokenizer_name ${tokenizer_name}"
[ -n "${num_hidden_layers}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --num_hidden_layers ${num_hidden_layers}"
[ -n "${dataset_config}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --dataset_config ${dataset_config}"
[ -n "${eval_holdout_size}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval_holdout_size ${eval_holdout_size}"
[ -n "${train_tokenized_cache}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --train_tokenized_cache ${train_tokenized_cache}"
[ -n "${eval_tokenized_cache}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval_tokenized_cache ${eval_tokenized_cache}"
[ -n "${halfpower_c}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --halfpower_c ${halfpower_c}"
[ -n "${max_eval_samples}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --max_eval_samples ${max_eval_samples}"
[ -n "${muon_curriculum_switch_step}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --muon_curriculum_switch_step ${muon_curriculum_switch_step}"

# Handle max_steps vs num_train_epochs
STEP_ARGS=""
if [ "${max_steps}" -gt 0 ] 2>/dev/null; then
    STEP_ARGS="--max_steps ${max_steps}"
else
    STEP_ARGS="--num_train_epochs ${num_train_epochs}"
fi

# Handle warmup: prefer warmup_steps if set, otherwise use warmup_ratio
WARMUP_ARGS=""
if [ "${warmup_steps}" -gt 0 ] 2>/dev/null; then
    WARMUP_ARGS="--warmup_steps ${warmup_steps}"
else
    WARMUP_ARGS="--warmup_ratio ${warmup_ratio}"
fi

# ============================================================================
# TRAINING COMMAND
# ============================================================================

echo "============================================================================"
echo "[INFO] GSM-8K Finetuning"
echo "============================================================================"
echo "[INFO] Run name:           $run_name"
echo "[INFO] Model:              $model_name_or_path"
echo "[INFO] From pretrained:    $from_pretrained"
echo "[INFO] Dataset:            $dataset_name (config=$dataset_config)"
echo "[INFO] Optimizer backend:  $muon_backend (exp=$muon_exponent)"
echo "[INFO] Curriculum:         $muon_curriculum"
echo "[INFO] Learning rate:      $learning_rate (muon_factor=$muon_lr_factor)"
echo "[INFO] Max steps:          $max_steps"
echo "[INFO] Num epochs:         $num_train_epochs"
echo "[INFO] Batch size:         $per_device_train_batch_size x $gradient_accumulation_steps x $num_gpus"
echo "[INFO] Num GPUs:           $num_gpus"
echo "[INFO] Output dir:         $output_dir"
echo "============================================================================"


${PYTHON} -m torch.distributed.run --standalone --nproc_per_node=${num_gpus} --master_port=${master_port} train_gsm8k.py \
    --model_name_or_path ${model_name_or_path} \
    --from_pretrained ${from_pretrained} \
    --dataset_name ${dataset_name} \
    --train_split ${train_split} \
    --eval_split ${eval_split} \
    --max_seq_length ${max_seq_length} \
    --prompt_column ${prompt_column} \
    --response_column ${response_column} \
    --use_chat_template ${use_chat_template} \
    --muon_backend ${muon_backend} \
    --muon_exponent ${muon_exponent} \
    --muon_lr_factor ${muon_lr_factor} \
    --muon_momentum ${muon_momentum} \
    --muon_backend_steps ${muon_backend_steps} \
    --muon_curriculum ${muon_curriculum} \
    --muon_curriculum_target_backend ${muon_curriculum_target_backend} \
    --muon_curriculum_target_exponent ${muon_curriculum_target_exponent} \
    --output_dir ${output_dir} \
    --run_name ${run_name} \
    --do_train True \
    --do_eval True \
    --learning_rate ${learning_rate} \
    ${STEP_ARGS} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    ${WARMUP_ARGS} \
    --weight_decay ${weight_decay} \
    --max_grad_norm 0 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\"min_lr_rate\": ${min_lr_rate}}" \
    --logging_steps ${logging_steps} \
    --eval_strategy ${eval_strategy} \
    --eval_steps ${eval_steps} \
    --save_strategy ${save_strategy} \
    --save_steps ${save_steps} \
    --save_total_limit 4 \
    --bf16 True \
    --report_to wandb \
    --remove_unused_columns False \
    --project_name ${project_name} \
    --seed ${seed} \
    --max_new_tokens ${max_new_tokens} \
    ${OPTIONAL_ARGS} \
    2>&1 | tee ${output_file}

echo "[INFO] Finished: $run_name"
