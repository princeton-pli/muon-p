#!/usr/bin/env bash
#SBATCH --job-name=muon-hf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=60G
#SBATCH --gres=gpu:2
#SBATCH --time=20:00:00
#SBATCH --output=outputs/slurm/muon-hf-%j.out
#SBATCH --error=outputs/slurm/muon-hf-%j.err

# Language-model pretraining with the Muon + AdamW dual optimizer.
#
# Runs train_hf.py via torchrun. Every variable below can be overridden from
# the environment (this is how submit_hf.sh launches sweeps), e.g.:
#
#   muon_backend=newtonschulz5 learning_rate=3.6e-3 ./run_hf.sh
#
# This file carries an optional SBATCH header so it can also be submitted with
# `sbatch run_hf.sh` on a Slurm cluster; it runs fine as a plain script too.

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
from_pretrained=${from_pretrained:-False}
num_hidden_layers=${num_hidden_layers:-0}
resume_from_checkpoint=${resume_from_checkpoint:-""}

# ============================================================================
# DATASET
# ============================================================================

dataset_name=${dataset_name:-"HuggingFaceFW/fineweb-edu"}
dataset_config=${dataset_config:-"sample-10BT"}
dataset_split=${dataset_split:-"train"}
max_seq_length=${max_seq_length:-1024}
text_column=${text_column:-"text"}
eval_holdout_size=${eval_holdout_size:-1000}
# Optional path to a cached tokenized dataset (saves re-tokenizing every run).
train_tokenized_cache=${train_tokenized_cache:-""}
eval_tokenized_cache=${eval_tokenized_cache:-""}

# ============================================================================
# MUON HYPERPARAMETERS
# ============================================================================

muon_backend=${muon_backend:-halfpower}
muon_exponent=${muon_exponent:-1/3}
muon_lr_factor=${muon_lr_factor:-0.1}
muon_momentum=${muon_momentum:-0.95}
muon_backend_steps=${muon_backend_steps:-6}
halfpower_c=${halfpower_c:-0.66}
halfpower_d=${halfpower_d:-0.795918}
log_effective_rank=${log_effective_rank:-False}

# Optional curriculum: switch the Muon backend partway through training.
muon_curriculum=${muon_curriculum:-False}
muon_curriculum_switch_step=${muon_curriculum_switch_step:-0}
muon_curriculum_target_backend=${muon_curriculum_target_backend:-halfpower}
muon_curriculum_target_exponent=${muon_curriculum_target_exponent:-1/3}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

learning_rate=${learning_rate:-3.6e-3}
max_steps=${max_steps:-5000}
per_device_train_batch_size=${per_device_train_batch_size:-8}
per_device_eval_batch_size=${per_device_eval_batch_size:-32}
total_batch_size=${total_batch_size:-512}
gradient_accumulation_steps="${gradient_accumulation_steps:-$((total_batch_size / num_gpus / per_device_train_batch_size))}"
warmup_steps=${warmup_steps:-400}
logging_steps=${logging_steps:-200}
eval_strategy=${eval_strategy:-steps}
eval_steps=${eval_steps:-500}
save_strategy=${save_strategy:-steps}
save_steps=${save_steps:-3000}
seed=${seed:-42}
min_lr_rate=${min_lr_rate:-0.5}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

exponent_clean=$(echo ${muon_exponent} | tr '/' '-')
model_slug=$(echo ${model_name_or_path} | sed 's|.*/||')
run_name_postfix=${run_name_postfix:-""}
run_name=${run_name:-"${model_slug}_${muon_backend}_exp${exponent_clean}_lr${learning_rate}${run_name_postfix:+_${run_name_postfix}}"}
output_dir=${output_dir:-"outputs/${run_name}"}
output_file="${output_dir}/train.log"
project_name=${project_name:-"muon-hf"}

mkdir -p ${output_dir}

# ============================================================================
# BUILD OPTIONAL ARGS
# ============================================================================

OPTIONAL_ARGS=""
[ "$num_gpus" -gt 1 ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --disable_tqdm True"
[ -n "${train_tokenized_cache}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --train_tokenized_cache ${train_tokenized_cache}"
[ -n "${eval_tokenized_cache}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval_tokenized_cache ${eval_tokenized_cache}"
[ -n "${dataset_config}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --dataset_config ${dataset_config}"
[ -n "${num_hidden_layers}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --num_hidden_layers ${num_hidden_layers}"
[ -n "${halfpower_c}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --halfpower_c ${halfpower_c}"
[ -n "${halfpower_d}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --halfpower_d ${halfpower_d}"
[ -n "${resume_from_checkpoint}" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --resume_from_checkpoint ${resume_from_checkpoint}"

# ============================================================================
# TRAINING COMMAND
# ============================================================================

echo "============================================================================"
echo "[INFO] Muon HF Training"
echo "============================================================================"
echo "[INFO] Run name:           $run_name"
echo "[INFO] Model:              $model_name_or_path"
echo "[INFO] From pretrained:    $from_pretrained"
echo "[INFO] Dataset:            $dataset_name ($dataset_config)"
echo "[INFO] Muon backend:       $muon_backend (exp=$muon_exponent)"
echo "[INFO] Learning rate:      $learning_rate (muon_factor=$muon_lr_factor)"
echo "[INFO] Max steps:          $max_steps"
echo "[INFO] Batch size:         $per_device_train_batch_size x $gradient_accumulation_steps x $num_gpus"
echo "[INFO] Num GPUs:           $num_gpus"
echo "[INFO] Output dir:         $output_dir"
echo "============================================================================"


${PYTHON} -m torch.distributed.run --standalone --nproc_per_node=${num_gpus} --master_port=${master_port} train_hf.py \
    --model_name_or_path ${model_name_or_path} \
    --from_pretrained ${from_pretrained} \
    --dataset_name ${dataset_name} \
    --dataset_split ${dataset_split} \
    --max_seq_length ${max_seq_length} \
    --text_column ${text_column} \
    --eval_holdout_size ${eval_holdout_size} \
    --muon_backend ${muon_backend} \
    --muon_exponent ${muon_exponent} \
    --muon_lr_factor ${muon_lr_factor} \
    --muon_momentum ${muon_momentum} \
    --muon_backend_steps ${muon_backend_steps} \
    --output_dir ${output_dir} \
    --run_name ${run_name} \
    --do_train True \
    --do_eval True \
    --learning_rate ${learning_rate} \
    --max_steps ${max_steps} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --warmup_steps ${warmup_steps} \
    --max_grad_norm 0 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\"min_lr_rate\": ${min_lr_rate}}" \
    --logging_steps ${logging_steps} \
    --eval_strategy ${eval_strategy} \
    --eval_steps ${eval_steps} \
    --save_strategy ${save_strategy} \
    --save_steps ${save_steps} \
    --save_total_limit 5 \
    --bf16 True \
    --report_to wandb \
    --remove_unused_columns False \
    --project_name ${project_name} \
    --muon_curriculum ${muon_curriculum} \
    --muon_curriculum_switch_step ${muon_curriculum_switch_step} \
    --muon_curriculum_target_backend ${muon_curriculum_target_backend} \
    --muon_curriculum_target_exponent ${muon_curriculum_target_exponent} \
    --log_effective_rank ${log_effective_rank} \
    --seed ${seed} \
    ${OPTIONAL_ARGS} \
    2>&1 | tee ${output_file}

# max_grad_norm=0 because Muon already normalizes the matrix-arm updates.

echo "[INFO] Finished: $run_name"
