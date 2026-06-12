"""
GSM-8K finetuning script with Muon + AdamW dual optimizer support.

Finetunes a pretrained causal LM on the GSM-8K math reasoning dataset
(and other math / code datasets) with options for:
  - Optimizer selection: AdamW-only, dual AdamW+Muon, or curriculum switching
  - Curriculum learning: switch Muon backend mid-training
  - Generation-based evaluation with exact-match accuracy on final answers
"""

import os
import json
import logging
from dataclasses import asdict, fields as dc_fields

import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_from_disk

from all_arguments import CustomTrainingArguments, ModelArguments, GSM8KDataArguments
from dual_optimizer import (
    MuonArguments,
    MuonTrainer,
    SwitchBackendCallback,
)
from eval_gsm8k import (
    compute_gsm8k_metrics,
    preprocess_logits_for_gsm8k,
    GenerationEvalCallback,
    extract_gsm8k_answer,
    extract_code_answer,
    normalize_number,
    normalize_code_answer,
)
from eval_math500 import extract_math500_answer, normalize_math_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data collator
# =============================================================================

class GSM8KCollator:
    """Data collator for GSM-8K finetuning.

    Pads input_ids and labels to the longest sequence in the batch.
    Labels are masked with -100 on the prompt portion so that loss is
    computed only on the chain-of-thought answer.
    """

    def __init__(self, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, examples: list) -> dict:
        batch_input_ids = []
        batch_labels = []
        batch_lengths = []

        for ex in examples:
            input_ids = ex["input_ids"]
            labels = ex["labels"]

            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_lengths.append(len(input_ids))

        max_len = max(batch_lengths)

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for i in range(len(batch_input_ids)):
            pad_len = max_len - batch_lengths[i]
            padded_input_ids.append(batch_input_ids[i] + [self.pad_token_id] * pad_len)
            padded_labels.append(batch_labels[i] + [-100] * pad_len)
            padded_attention_mask.append([1] * batch_lengths[i] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# =============================================================================
# Dataset preparation
# =============================================================================

def _format_gsm8k_example(question: str, answer: str, tokenizer, use_chat_template: bool):
    """Format a single GSM-8K example into input_ids and labels.

    The prompt is masked in labels (-100) so loss is only on the answer.
    """
    if use_chat_template and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True)

        # Find where the assistant response starts by tokenizing just the prompt
        prompt_messages = [{"role": "user", "content": question}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True,
        )
        prefix_len = len(prompt_ids)
    else:
        prompt_text = f"Question: {question}\nAnswer: "
        full_text = prompt_text + answer + tokenizer.eos_token

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prefix_len = len(prompt_ids)

    labels = [-100] * prefix_len + full_ids[prefix_len:]

    return full_ids, labels


_CACHE_META_FILENAME = "cache_meta.json"
_SPLIT_SEED = 42


def _tokenized_cache_meta(data_args, tokenizer, *, split: str) -> dict:
    """Params that fully determine the contents of a tokenized cache.

    Two caches are interchangeable iff they share this dict. `eval_holdout_size`
    and `split_seed` are intentionally NOT included: the train cache always
    holds the full tokenized train_split, and the holdout is applied AFTER
    loading the cache (see `prepare_gsm8k_datasets`). The eval cache is only
    consulted when no holdout is used, so it's similarly independent of those.
    """
    common = {
        "split": split,
        "dataset_name": data_args.dataset_name,
        "dataset_config": data_args.dataset_config,
        "prompt_column": data_args.prompt_column,
        "response_column": data_args.response_column,
        "use_chat_template": data_args.use_chat_template,
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }
    if split == "train":
        common["train_split"] = data_args.train_split
    else:
        common["eval_split"] = data_args.eval_split
    return common


def _cache_is_compatible(cache_path: str, expected_meta: dict) -> bool:
    meta_path = os.path.join(cache_path, _CACHE_META_FILENAME)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            on_disk = json.load(f)
    except Exception:
        return False
    return on_disk == expected_meta


def _load_or_tokenize(
    raw_ds, cache_path, expected_meta, tokenize_fn, num_proc, *, desc: str,
):
    """Load a tokenized dataset from cache if compatible, otherwise tokenize and save."""
    if cache_path and os.path.exists(cache_path):
        if _cache_is_compatible(cache_path, expected_meta):
            logger.info(f"Loading tokenized {desc} from cache: {cache_path}")
            return load_from_disk(cache_path)
        logger.warning(
            f"Tokenized {desc} cache at {cache_path} is incompatible with current "
            "data_args (tokenizer / columns / split / use_chat_template / etc). "
            "Regenerating."
        )

    logger.info(f"Tokenizing {desc} dataset...")
    ds = raw_ds.map(
        tokenize_fn, batched=True, remove_columns=raw_ds.column_names,
        desc=f"Tokenizing {desc}", num_proc=num_proc,
    )
    if cache_path:
        logger.info(f"Saving tokenized {desc} to cache: {cache_path}")
        ds.save_to_disk(cache_path)
        with open(os.path.join(cache_path, _CACHE_META_FILENAME), "w") as f:
            json.dump(expected_meta, f, indent=2, sort_keys=True)
    return ds


def prepare_gsm8k_datasets(data_args, tokenizer):
    """Load and preprocess GSM-8K dataset.

    Strategy: the train cache holds the *full* tokenized train_split. The
    eval_holdout_size split is applied AFTER loading the cache, so toggling
    or resizing the holdout never invalidates the tokenization cache.

    Alignment between tokenized and raw eval: HuggingFace `train_test_split`
    with a fixed seed and a dataset of fixed length picks the same row indices
    regardless of column contents, so splitting the tokenized full-train and
    raw full-train with `(seed=_SPLIT_SEED, test_size=K)` yields tokenized
    eval rows that correspond 1:1 (by index) to the raw eval rows fed to
    `GenerationEvalCallback`.
    """
    num_proc = min(os.cpu_count() or 1, 8)

    load_kwargs = {}
    if data_args.dataset_config:
        load_kwargs["name"] = data_args.dataset_config

    raw_train_full = load_dataset(data_args.dataset_name, split=data_args.train_split, **load_kwargs)

    prompt_col = data_args.prompt_column
    response_col = data_args.response_column
    use_chat = data_args.use_chat_template

    def tokenize_fn(examples):
        all_input_ids = []
        all_labels = []
        for q, a in zip(examples[prompt_col], examples[response_col]):
            input_ids, labels = _format_gsm8k_example(q, a, tokenizer, use_chat)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        return {"input_ids": all_input_ids, "labels": all_labels}

    tokenized_train_full = _load_or_tokenize(
        raw_train_full, data_args.train_tokenized_cache,
        _tokenized_cache_meta(data_args, tokenizer, split="train"),
        tokenize_fn, num_proc, desc="train",
    )

    use_holdout = data_args.eval_holdout_size is not None and data_args.eval_holdout_size > 0
    if use_holdout:
        token_split = tokenized_train_full.train_test_split(
            test_size=data_args.eval_holdout_size, seed=_SPLIT_SEED,
        )
        train_dataset = token_split["train"]
        eval_dataset = token_split["test"]

        raw_split = raw_train_full.train_test_split(
            test_size=data_args.eval_holdout_size, seed=_SPLIT_SEED,
        )
        raw_eval = raw_split["test"]
        logger.info(
            f"Held out {data_args.eval_holdout_size} examples from train for eval "
            "(split applied after loading tokenized cache)"
        )
    else:
        train_dataset = tokenized_train_full
        raw_eval = load_dataset(data_args.dataset_name, split=data_args.eval_split, **load_kwargs)
        eval_dataset = _load_or_tokenize(
            raw_eval, data_args.eval_tokenized_cache,
            _tokenized_cache_meta(data_args, tokenizer, split="eval"),
            tokenize_fn, num_proc, desc="eval",
        )

    logger.info(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")
    return train_dataset, eval_dataset, raw_eval


# =============================================================================
# Wandb helpers
# =============================================================================

def _should_log_to_wandb(args: CustomTrainingArguments) -> bool:
    report_to = getattr(args, "report_to", None)
    if report_to is None:
        return False
    if isinstance(report_to, str):
        lowered = report_to.strip().lower()
        if lowered in {"", "none"}:
            return False
        return "wandb" in lowered or lowered == "all"
    if isinstance(report_to, (list, tuple, set)):
        lowered = {str(t).strip().lower() for t in report_to}
        return "wandb" in lowered or "all" in lowered
    return False


def _log_wandb_config(training_args, model_args, data_args, muon_args):
    if os.environ.get("WANDB_DISABLED", "").strip().lower() in {"true", "1", "yes"}:
        return
    try:
        import wandb
    except Exception:
        return
    config_payload = {
        "training": getattr(training_args, "to_dict", lambda: asdict(training_args))(),
        "model": asdict(model_args),
        "data": asdict(data_args),
        "muon": asdict(muon_args),
    }
    try:
        if wandb.run is None:
            return
        wandb.config.update(config_payload, allow_val_change=True)
    except Exception as e:
        logging.warning("Failed to upload config to W&B. (%s)", e)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = HfArgumentParser((ModelArguments, GSM8KDataArguments, MuonArguments, CustomTrainingArguments))
    model_args, data_args, muon_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # -------------------------------------------------------------------------
    # Wandb
    # -------------------------------------------------------------------------
    os.environ["WANDB_PROJECT"] = muon_args.project_name
    wandb_dir = training_args.output_dir
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
    if getattr(training_args, "run_name", None):
        os.environ["WANDB_NAME"] = training_args.run_name

    # -------------------------------------------------------------------------
    # Checkpoint resume
    # -------------------------------------------------------------------------
    resume_checkpoint_path = None
    resume_flag = training_args.resume_from_checkpoint
    if resume_flag is not None:
        if isinstance(resume_flag, str):
            flag = resume_flag.strip().lower()
            if flag in {"true", "1", "yes"}:
                resume_checkpoint_path = get_last_checkpoint(training_args.output_dir)
                if resume_checkpoint_path is None:
                    raise FileNotFoundError(
                        f"resume_from_checkpoint=True but no checkpoint found in: {training_args.output_dir}"
                    )
            elif flag not in {"false", "0", "no", "none"}:
                resume_checkpoint_path = resume_flag
        elif resume_flag is True:
            resume_checkpoint_path = get_last_checkpoint(training_args.output_dir)

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        logger.error("Hub fetch failed for tokenizer, falling back to local cache")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        # raise Exception("Hub fetch failed for tokenizer, falling back to local cache")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    if model_args.from_pretrained:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            logger.error(f"Hub fetch failed for model, falling back to local cache: {e}")
            _prev_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            if _prev_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = _prev_offline
    else:
        try:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        except Exception:
            logger.error("Hub fetch failed for config, falling back to local cache")
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, local_files_only=True)
        if model_args.num_hidden_layers is not None:
            config.num_hidden_layers = model_args.num_hidden_layers
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_args.model_name_or_path}  ({n_params / 1e6:.1f}M parameters)")

    # -------------------------------------------------------------------------
    # Dataset & Collator
    # -------------------------------------------------------------------------
    train_dataset, eval_dataset, raw_eval = prepare_gsm8k_datasets(data_args, tokenizer)
    collator = GSM8KCollator(tokenizer=tokenizer, max_seq_len=data_args.max_seq_length)
    training_args.remove_unused_columns = False

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
        compute_metrics=compute_gsm8k_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_gsm8k,
    )

    # --- Curriculum callback ---
    if muon_args.muon_curriculum:
        switch_step = muon_args.muon_curriculum_switch_step
        if switch_step is None:
            switch_step = max(1, training_args.max_steps - 500)
        trainer.add_callback(SwitchBackendCallback(
            trainer,
            switch_step=switch_step,
            target_backend=muon_args.muon_curriculum_target_backend,
            target_exponent=muon_args.muon_curriculum_target_exponent,
        ))
        logger.info(
            f"Registered SwitchBackendCallback: will switch to "
            f"{muon_args.muon_curriculum_target_backend} (exp={muon_args.muon_curriculum_target_exponent}) "
            f"at step {switch_step}"
        )

    # --- Generation eval callback ---
    # Pick the answer-extractor based on dataset_name: math-style benchmarks
    # (NuminaMath, MATH-500, MATH, etc.) all use \boxed{...} answers; code
    # datasets (OpenCodeInstruct) use ``` fenced blocks; GSM-8K uses
    # '####'/'###' markers. Detection is case-insensitive substring match.
    ds_lower = (data_args.dataset_name or "").lower()
    if "numina" in ds_lower or "math" in ds_lower:
        extract_fn = extract_math500_answer
        normalize_fn = normalize_math_answer
        logger.info(f"[Generation Eval] Using \\boxed{{}} answer extraction (dataset={data_args.dataset_name})")
    elif "code" in ds_lower:
        extract_fn = extract_code_answer
        normalize_fn = normalize_code_answer
        logger.info(f"[Generation Eval] Using ``` code-block answer extraction (dataset={data_args.dataset_name})")
    else:
        extract_fn = extract_gsm8k_answer
        normalize_fn = normalize_number
        logger.info(f"[Generation Eval] Using ####/### answer extraction (dataset={data_args.dataset_name})")

    trainer.add_callback(GenerationEvalCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        eval_dataset=raw_eval,
        data_args=data_args,
        max_new_tokens=data_args.max_new_tokens,
        gen_batch_size=training_args.per_device_eval_batch_size,
        extract_fn=extract_fn,
        normalize_fn=normalize_fn,
    ))

    # --- Wandb config ---
    if _should_log_to_wandb(training_args):
        _log_wandb_config(training_args, model_args, data_args, muon_args)

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    if training_args.do_train:
        logger.info("=" * 80)
        logger.info("[TRAINING] Starting GSM-8K finetuning")
        logger.info(f"  Model:        {model_args.model_name_or_path}")
        logger.info(f"  Pretrained:   {model_args.from_pretrained}")
        logger.info(f"  Optimizer:    {muon_args.muon_backend}")
        logger.info(f"  Curriculum:   {muon_args.muon_curriculum}")
        logger.info(f"  LR:           {training_args.learning_rate}")
        logger.info(f"  Max steps:    {training_args.max_steps}")
        logger.info(f"  Batch size:   {training_args.per_device_train_batch_size}")
        logger.info(f"  Grad accum:   {training_args.gradient_accumulation_steps}")
        logger.info("=" * 80)

        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint_path)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model()

    # -------------------------------------------------------------------------
    # Log all configs
    # -------------------------------------------------------------------------
    for label, args_obj in [("Model", model_args), ("Data", data_args), ("Muon", muon_args)]:
        vals = " | ".join(f"{f.name}={getattr(args_obj, f.name)}" for f in dc_fields(args_obj))
        logger.info(f"[{label}] {vals}")
    logger.info(
        f"[Training] lr={training_args.learning_rate} bs={training_args.per_device_train_batch_size} "
        f"epochs={training_args.num_train_epochs} warmup={training_args.warmup_steps} "
        f"wd={training_args.weight_decay} seed={training_args.seed} max_steps={training_args.max_steps}"
    )

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
