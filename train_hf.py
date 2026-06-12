"""
HuggingFace Trainer-based language-model pretraining with the Muon + AdamW
dual optimizer.

Muon is used for all 2-D weight matrices (excluding embeddings and lm_head).
AdamW is used for everything else: embeddings, unembeddings (lm_head),
scalars, biases, LayerNorm / RMSNorm parameters, and any 1-D weights.

Examples are packed into fixed-length sequences and trained with
FlashAttention varlen, so flash-attn must be installed (see README).
"""

import os
import logging

import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
    set_seed,
)
from datasets import load_dataset, load_from_disk

import utils
from all_arguments import CustomTrainingArguments, ModelArguments, LMDataArguments
from dual_optimizer import (
    MuonArguments,
    MuonTrainer,
    SwitchBackendCallback,
    _is_embedding_or_lm_head,
)
from perf_bench import PerfBenchmarkArguments, PerfBenchmarkCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Effective-rank logging
# =============================================================================

class EffectiveRankCallback(TrainerCallback):
    """Log the mean effective rank of the Muon-optimized weight matrices and
    their gradients to wandb during training.

    Mirrors the effective-rank logging in ``train_gpt2.py``: the metric is
    computed over the 2-D weight matrices that Muon updates (i.e. all >=2-D
    parameters that are not embeddings or the LM head). Gradients are read at
    ``on_pre_optimizer_step``, after gradient accumulation / DDP sync but before
    they are zeroed, so they reflect the update actually applied this step.
    """

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        logging_steps = max(int(args.logging_steps), 1)
        # global_step is incremented after the optimizer step; the upcoming step
        # is state.global_step + 1, which is also the step the loss is logged at.
        next_step = state.global_step + 1
        if next_step % logging_steps != 0:
            return
        if model is None:
            return

        import wandb
        if wandb.run is None:
            return

        param_erank_sum = 0.0
        grad_erank_sum = 0.0
        param_count = 0
        grad_count = 0
        for name, p in model.named_parameters():
            if not p.requires_grad or p.ndim < 2 or _is_embedding_or_lm_head(name):
                continue
            param_erank_sum += utils.effective_rank(p.data).item()
            param_count += 1
            if p.grad is not None:
                grad_erank_sum += utils.effective_rank(p.grad).item()
                grad_count += 1

        log_dict = {}
        if param_count > 0:
            log_dict["erank/mean"] = param_erank_sum / param_count
        if grad_count > 0:
            log_dict["erank_grad/mean"] = grad_erank_sum / grad_count
        if log_dict:
            wandb.log(log_dict, step=next_step)


# =============================================================================
# Dataset preparation
# =============================================================================

class GroupingCollator:
    """Padding-free packed collator for causal LM training with FlashAttention varlen support.

    Packs all examples from a DataLoader batch into a single flat sequence of
    length <= max_seq_len, emitting cumulative sequence lengths and position_ids
    for use with flash_attn_varlen_func.  Sequences that would exceed max_seq_len
    are dropped; leftover tokens at the end of a sequence that doesn't fit are
    also dropped (they appear in other batches via shuffling).
    """

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len

    def __call__(self, examples: list) -> dict:
        flat_input_ids: list = []
        flat_labels: list = []
        flat_pos: list = []
        cu_seq_lens = [0]
        max_len = 0

        for ex in examples:
            ids = ex["input_ids"] if isinstance(ex["input_ids"], list) else ex["input_ids"].tolist()
            if not ids:
                continue
            # Skip if adding this sequence would exceed max_seq_len
            if len(flat_input_ids) + len(ids) > self.max_seq_len:
                continue

            flat_input_ids.extend(ids)
            # First token of each packed sequence has no previous context -> ignore
            flat_labels.append(-100)
            flat_labels.extend(ids[1:])
            flat_pos.extend(list(range(len(ids))))

            cu_seq_lens.append(cu_seq_lens[-1] + len(ids))
            max_len = max(max_len, len(ids))

        if not flat_input_ids:
            # Edge case: nothing fit — return a single padded sample
            return {
                "input_ids": torch.zeros(1, 1, dtype=torch.long),
                "labels": torch.full((1, 1), -100, dtype=torch.long),
            }

        input_ids_t = torch.tensor([flat_input_ids], dtype=torch.long)
        labels_t = torch.tensor([flat_labels], dtype=torch.long)
        position_ids_t = torch.tensor([flat_pos], dtype=torch.long)
        cu = torch.tensor(cu_seq_lens, dtype=torch.int32)

        return {
            "input_ids": input_ids_t,
            "labels": labels_t,
            "position_ids": position_ids_t,
            "cu_seq_lens_q": cu,
            "cu_seq_lens_k": cu,
            "max_length_q": max_len,
            "max_length_k": max_len,
        }


def _tokenize_raw(raw_dataset, tokenizer, text_column, num_proc):
    """Tokenize raw text into per-document token IDs with EOS appended."""
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples[text_column],
            add_special_tokens=False,
            truncation=True,
            max_length=10_000_000,
        )
        # Keep one row per document, append EOS to each
        result = []
        for ids in tokenized["input_ids"]:
            result.append(ids + [tokenizer.eos_token_id])
        return {"input_ids": result}

    tokenized = raw_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
        num_proc=num_proc,
    )
    return tokenized


def prepare_datasets(data_args, tokenizer):
    """Load and tokenize datasets."""
    load_kwargs = {}
    if data_args.dataset_config:
        load_kwargs["name"] = data_args.dataset_config

    num_proc = min(os.cpu_count() or 1, 8)
    max_length = data_args.max_seq_length

    # --- Tokenize training set (with cache support) ---
    if data_args.train_tokenized_cache and os.path.exists(data_args.train_tokenized_cache):
        logger.info(f"Loading tokenized training dataset from cache: {data_args.train_tokenized_cache}")
        train_tokenized = load_from_disk(data_args.train_tokenized_cache)
        logger.info(f"Loaded {len(train_tokenized)} tokenized training rows from cache")
        raw_dataset = None
    else:
        raw_dataset = load_dataset(data_args.dataset_name, split=data_args.dataset_split, **load_kwargs)
        logger.info("Tokenizing training dataset...")
        train_tokenized = _tokenize_raw(raw_dataset, tokenizer, data_args.text_column, num_proc)
        if data_args.train_tokenized_cache:
            logger.info(f"Saving tokenized training dataset to cache: {data_args.train_tokenized_cache}")
            train_tokenized.save_to_disk(data_args.train_tokenized_cache)

    train_dataset = train_tokenized
    logger.info(f"Train: {len(train_dataset)} documents (collator will chunk into blocks of {max_length})")

    # --- Eval set ---
    eval_dataset = None
    if data_args.eval_tokenized_cache and os.path.exists(data_args.eval_tokenized_cache):
        logger.info(f"Loading tokenized eval dataset from cache: {data_args.eval_tokenized_cache}")
        eval_dataset = load_from_disk(data_args.eval_tokenized_cache)
        logger.info(f"Loaded {len(eval_dataset)} tokenized eval rows from cache")
    elif raw_dataset is not None:
        raw_eval = None
        if data_args.eval_holdout_size is not None and data_args.eval_holdout_size > 0:
            split = raw_dataset.train_test_split(test_size=data_args.eval_holdout_size, seed=42)
            raw_eval = split["test"]
            logger.info(f"Held out {data_args.eval_holdout_size} examples from train for eval")
        elif data_args.eval_split:
            raw_eval = load_dataset(data_args.dataset_name, split=data_args.eval_split, **load_kwargs)

        if raw_eval is not None:
            logger.info("Tokenizing evaluation dataset...")
            eval_dataset = _tokenize_raw(raw_eval, tokenizer, data_args.text_column, num_proc)
            if data_args.eval_tokenized_cache:
                logger.info(f"Saving tokenized eval dataset to cache: {data_args.eval_tokenized_cache}")
                eval_dataset.save_to_disk(data_args.eval_tokenized_cache)
    elif raw_dataset is None and data_args.eval_holdout_size and data_args.eval_holdout_size > 0:
        # Loaded tokenized train data directly (no raw_dataset), carve eval from it
        split = train_dataset.train_test_split(test_size=data_args.eval_holdout_size, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Carved {len(eval_dataset)} eval rows from tokenized train data")

    if eval_dataset is not None:
        logger.info(f"Eval: {len(eval_dataset)} documents (collator will chunk into blocks of {max_length})")

    return train_dataset, eval_dataset


# =============================================================================
# Main
# =============================================================================

def main():
    parser = HfArgumentParser(
        (ModelArguments, LMDataArguments, MuonArguments, PerfBenchmarkArguments, CustomTrainingArguments)
    )
    model_args, data_args, muon_args, perf_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)


    # -------------------------------------------------------------------------
    # Wandb
    # -------------------------------------------------------------------------
    os.environ["WANDB_PROJECT"] = muon_args.project_name
    wandb_dir = training_args.output_dir
    if wandb_dir is not None:
        os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir
    if getattr(training_args, "run_name", None):
        os.environ["WANDB_NAME"] = training_args.run_name
    # -------------------------------------------------------------------------
    # Tokenizer & Model — use HF_HUB_OFFLINE to force cache-only loading,
    # then restore so dataset downloading still works.
    # -------------------------------------------------------------------------
    _prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Packed varlen training requires FlashAttention-2.
    attn_impl = "flash_attention_2"
    if model_args.from_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        if model_args.num_hidden_layers > 0:
            config.num_hidden_layers = model_args.num_hidden_layers
        config._attn_implementation = attn_impl
        model = AutoModelForCausalLM.from_config(config)
        n_params = utils.count_parameters(model, trainable_only=False)
        logger.info(f"Model: {model_args.model_name_or_path}  ({n_params / 1e6:.1f}M parameters)")

    if _prev_offline is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = _prev_offline

    model.resize_token_embeddings(len(tokenizer))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_args.model_name_or_path}  ({n_params / 1e6:.1f}M parameters)")

    # -------------------------------------------------------------------------
    # Dataset & Collator
    # -------------------------------------------------------------------------
    train_dataset, eval_dataset = prepare_datasets(data_args, tokenizer)
    collator = GroupingCollator(max_seq_len=data_args.max_seq_length)

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    if eval_dataset is not None:
        logger.info(f"Eval dataset: {len(eval_dataset)} samples")

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    trainer = MuonTrainer(
        muon_args=muon_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    if muon_args.log_effective_rank:
        trainer.add_callback(EffectiveRankCallback())
        logger.info("Registered EffectiveRankCallback: logging param/grad effective rank to wandb")

    if muon_args.muon_curriculum:
        switch_step = muon_args.muon_curriculum_switch_step
        trainer.add_callback(SwitchBackendCallback(
            trainer,
            switch_step=switch_step,
            target_backend=muon_args.muon_curriculum_target_backend,
            target_exponent=muon_args.muon_curriculum_target_exponent,
        ))
        logger.info(
            f"Registered SwitchBackendCallback: will switch to "
            f"{muon_args.muon_curriculum_target_backend} "
            f"(exponent={muon_args.muon_curriculum_target_exponent})"
        )

    # --- Performance benchmark callback (opt-in via --perf_benchmark) ---
    if perf_args.perf_benchmark:
        trainer.add_callback(PerfBenchmarkCallback(perf_args))
        logger.info(
            f"Registered PerfBenchmarkCallback: warmup={perf_args.perf_warmup_steps}, "
            f"measure={perf_args.perf_measure_steps}, "
            f"stop_after_measure={perf_args.perf_stop_after_measure}"
        )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    if training_args.do_train:
        resume_ckpt = training_args.resume_from_checkpoint
        if resume_ckpt:
            logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
        else:
            logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model()
        
    # -------------------------------------------------------------------------
    # Log configs
    # -------------------------------------------------------------------------
    from dataclasses import fields as dc_fields
    for label, args_obj in [("Model", model_args), ("Data", data_args), ("Muon", muon_args)]:
        vals = " | ".join(f"{f.name}={getattr(args_obj, f.name)}" for f in dc_fields(args_obj))
        logger.info(f"[{label}] {vals}")
    logger.info(f"[Training] lr={training_args.learning_rate} bs={training_args.per_device_train_batch_size} "
                f"epochs={training_args.num_train_epochs} warmup={training_args.warmup_steps} "
                f"wd={training_args.weight_decay} seed={training_args.seed} max_steps={training_args.max_steps}")

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    if training_args.do_eval and eval_dataset is not None:
        logger.info("Running evaluation...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

