"""
GSM-8K evaluation utilities: answer extraction, metrics, and generation-based evaluation.

Used by train_gsm8k.py. Also usable standalone for evaluating saved checkpoints.
"""

import math
import re
import logging
from typing import Optional, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from datasets import load_dataset

from eval_math500 import extract_math500_answer, normalize_math_answer

logger = logging.getLogger(__name__)


def _select_extractor(dataset_name: str):
    """Pick (extract_fn, normalize_fn) for a dataset by name (case-insensitive).

    Math-style (NuminaMath, MATH-500, MATH, etc.) -> \\boxed{...} extraction.
    Code-style (OpenCodeInstruct, etc.) -> ``` fenced-block extraction.
    Default -> GSM-8K '####'/'###' numerical-answer extraction.
    """
    ds_lower = (dataset_name or "").lower()
    if "numina" in ds_lower or "math" in ds_lower:
        return extract_math500_answer, normalize_math_answer
    if "code" in ds_lower:
        return extract_code_answer, normalize_code_answer
    return extract_gsm8k_answer, normalize_number


# =============================================================================
# Answer extraction
# =============================================================================

_SPECIAL_TOKENS = ["<|eot_id|>", "<|endoftext|>", "<|end|>"]


def _clean_special_tokens(text: str) -> str:
    """Strip common special-token artifacts from decoded text."""
    for tok in _SPECIAL_TOKENS:
        if tok in text:
            text = text.split(tok)[0]
    return text.rstrip("! ").strip()


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the final numerical answer from a GSM-8K response.

    Tries #### then ### markers. Falls back to the last non-empty line.
    Strips special tokens and padding artifacts before returning.
    """
    for marker in ("####", "###"):
        if marker in text:
            candidate = text.rsplit(marker, 1)[-1].strip()
            candidate = _clean_special_tokens(candidate)
            if candidate:
                return candidate.replace(",", "")

    # Fallback: last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidate = _clean_special_tokens(lines[-1])
        if candidate:
            return candidate.replace(",", "")
    return None


def normalize_number(s: str) -> Optional[str]:
    """Normalize a number string for comparison."""
    try:
        return str(float(s))
    except (ValueError, TypeError):
        return s


def extract_code_answer(text: str) -> Optional[str]:
    """Extract a code answer for OpenCodeInstruct-style datasets.

    Pulls the contents of the last fenced ``` code block if present,
    otherwise returns the cleaned text. 
    """
    text = _clean_special_tokens(text)
    matches = list(re.finditer(r"```(?:[\w+\-.]*)\n?(.*?)```", text, re.DOTALL))
    if matches:
        candidate = matches[-1].group(1).strip()
        if candidate:
            return candidate
    stripped = text.strip()
    return stripped if stripped else None


def normalize_code_answer(s: Optional[str]) -> Optional[str]:
    """Collapse whitespace for code-string comparison."""
    if s is None:
        return None
    return re.sub(r"\s+", " ", s.strip())


# =============================================================================
# HF Trainer metrics
# =============================================================================

def compute_gsm8k_metrics(eval_pred):
    """Compute token-level and sequence-level accuracy for GSM-8K.

    Works with the standard HF Trainer evaluation pipeline where
    preprocess_logits_for_metrics reduces logits to argmax predictions.
    """
    preds = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids)

    valid = labels != -100
    token_total = int(valid.sum())
    if token_total == 0:
        return {"token_accuracy": 0.0, "sequence_accuracy": 0.0}

    token_correct = int(((preds == labels) & valid).sum())
    token_acc = token_correct / token_total

    per_example_total = valid.sum(axis=1)
    per_example_correct = ((preds == labels) & valid).sum(axis=1)
    nonempty = per_example_total > 0
    seq_acc = float(
        (per_example_correct[nonempty] == per_example_total[nonempty]).mean()
    ) if nonempty.any() else 0.0

    return {
        "token_accuracy": float(token_acc),
        "sequence_accuracy": float(seq_acc),
    }


def preprocess_logits_for_gsm8k(logits, labels):
    """Reduce logits to argmax predictions to save memory during eval."""
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)


# =============================================================================
# Generation-based evaluation callback
# =============================================================================

class GenerationEvalCallback(TrainerCallback):
    """Run generation-based exact-match evaluation at eval steps.

    ``extract_fn`` and ``normalize_fn`` control how the gold answer is parsed
    from ``response_column`` and how predicted answers are compared. Defaults
    are GSM-8K style (``####``-marker numerical answers); for MATH-style
    datasets (MATH-500, NuminaMath-CoT) pass ``extract_math500_answer`` and
    ``normalize_math_answer`` from ``eval_math500``.
    """

    def __init__(self, trainer, tokenizer, eval_dataset, data_args,
                 max_new_tokens=128, gen_batch_size=8,
                 extract_fn=None, normalize_fn=None):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.data_args = data_args
        self.max_new_tokens = max_new_tokens
        self.gen_batch_size = gen_batch_size
        self.extract_fn = extract_fn if extract_fn is not None else extract_gsm8k_answer
        self.normalize_fn = normalize_fn if normalize_fn is not None else normalize_number

    def _encode_prompt(self, question):
        """Encode a single question into prompt token ids."""
        if (self.data_args.use_chat_template
                and hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template):
            messages = [{"role": "user", "content": question}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
        else:
            prompt_text = f"Question: {question}\nAnswer:"
            return self.tokenizer.encode(prompt_text, add_special_tokens=False)

    def _left_pad_batch(self, prompt_ids_list, device):
        """Left-pad a list of token-id lists and return input_ids + attention_mask."""
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_len = max(len(ids) for ids in prompt_ids_list)

        padded_input_ids = []
        attention_masks = []
        for ids in prompt_ids_list:
            pad_len = max_len - len(ids)
            padded_input_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        return (
            torch.tensor(padded_input_ids, dtype=torch.long, device=device),
            torch.tensor(attention_masks, dtype=torch.long, device=device),
        )

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_dataset is None:
            return

        model = self.trainer.model
        model.eval()
        device = next(model.parameters()).device

        correct = 0
        total = 0
        examples = []

        dataset = self.eval_dataset
        if self.data_args.max_eval_samples > 0 and len(dataset) > self.data_args.max_eval_samples:
            dataset = dataset.select(range(self.data_args.max_eval_samples))

        # Prepare all prompts and gold answers
        all_prompt_ids = []
        all_gold_answers = []
        all_questions = []
        for ex in dataset:
            question = ex[self.data_args.prompt_column]
            gold_answer_text = ex[self.data_args.response_column]
            all_questions.append(question)
            all_gold_answers.append(self.extract_fn(gold_answer_text))
            all_prompt_ids.append(self._encode_prompt(question))

        # Process in batches with left-padding
        all_pred_answers = []
        # Use a safe pad token id: if pad == eos (common in LLaMA-family),
        # generation can terminate prematurely or produce warnings.
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None or pad_id == self.tokenizer.eos_token_id:
            pad_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
        for batch_start in range(0, len(all_prompt_ids), self.gen_batch_size):
            batch_prompt_ids = all_prompt_ids[batch_start:batch_start + self.gen_batch_size]
            input_ids, attention_mask = self._left_pad_batch(batch_prompt_ids, device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )

            # Decode only the generated portion for each example
            eos_id = self.tokenizer.eos_token_id
            for i, prompt_ids in enumerate(batch_prompt_ids):
                prompt_len = input_ids.shape[1]  # padded length
                generated_tokens = output_ids[i][prompt_len:].tolist()

                # Truncate at first EOS — generate() pads all seqs to same length
                if eos_id is not None and eos_id in generated_tokens:
                    generated_tokens = generated_tokens[:generated_tokens.index(eos_id)]

                # Filter out padding artifacts (pad_id, token 0)
                generated_tokens = [
                    t for t in generated_tokens
                    if t != pad_id and t != 0
                ]

                generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                all_pred_answers.append(self.extract_fn(generated))

        # Score
        for i in range(len(all_gold_answers)):
            gold_answer = all_gold_answers[i]
            pred_answer = all_pred_answers[i]

            is_correct = False
            if gold_answer is not None and pred_answer is not None:
                is_correct = self.normalize_fn(pred_answer) == self.normalize_fn(gold_answer)

            if is_correct:
                correct += 1
            total += 1

            if len(examples) < 3:
                examples.append({
                    "question": all_questions[i][:100],
                    "gold": gold_answer,
                    "pred": pred_answer,
                    "correct": is_correct,
                })

        accuracy = correct / total if total > 0 else 0.0

        logger.info(f"[Generation Eval] Step {state.global_step}: "
                     f"accuracy={accuracy:.4f} ({correct}/{total})")
        for i, ex in enumerate(examples):
            logger.info(f"  Example {i}: gold={ex['gold']}, pred={ex['pred']}, correct={ex['correct']}")

        if hasattr(self.trainer, "log"):
            self.trainer.log({
                "eval/generation_accuracy": accuracy,
                "eval/generation_correct": correct,
                "eval/generation_total": total,
            })


# =============================================================================
# Checkpoint loading
# =============================================================================

def load_checkpoint(checkpoint_path: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load a model and tokenizer from a saved checkpoint directory.

    Returns (model, tokenizer) ready for generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, tokenizer


# =============================================================================
# pass@k evaluation
# =============================================================================

def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Codex / Chen et al. 2021).

    n = total samples per problem, c = number correct, k = k.
    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def _left_pad_batch(token_ids_list: List[List[int]], pad_id: int, device: str):
    """Left-pad a list of variable-length token-id lists into tensors."""
    max_len = max(len(ids) for ids in token_ids_list)
    padded_ids, masks = [], []
    for ids in token_ids_list:
        pad_len = max_len - len(ids)
        padded_ids.append([pad_id] * pad_len + ids)
        masks.append([0] * pad_len + [1] * len(ids))
    return (
        torch.tensor(padded_ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def _decode_generated(output_ids, prompt_len: int, eos_id, pad_id):
    """Extract generated token ids from a single sequence, truncating at EOS."""
    tokens = output_ids[prompt_len:].tolist()
    if eos_id is not None and eos_id in tokens:
        tokens = tokens[:tokens.index(eos_id)]
    return [t for t in tokens if t != pad_id and t != 0]


def evaluate_pass_at_k(
    checkpoint_path: str,
    k_values: List[int] = [1, 5, 10],
    n_samples: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    max_eval_samples: int = -1,
    batch_size: int = 8,
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    eval_split: str = "test",
    prompt_column: str = "question",
    response_column: str = "answer",
    use_chat_template: bool = False,
    device: str = "cuda",
    eval_holdout_size: int = 0,
    split_seed: int = 42,
    eval_seed: Optional[int] = None,
):
    """Load a checkpoint and evaluate pass@k with temperature sampling.

    For each problem, generates `n_samples` responses at the given temperature,
    extracts the answer from each (style auto-selected by dataset_name: GSM-8K
    ####-marker, NuminaMath/MATH \\boxed{}, or code-block extraction), and
    computes pass@k for each requested k.

    ``eval_seed`` (when not None) seeds the generation sampling RNG so pass@k is
    reproducible across runs; leave None for fresh (non-deterministic) sampling.

    Returns a dict with per-k pass rates and per-problem details.
    """
    extract_fn, normalize_fn = _select_extractor(dataset_name)
    logger.info(
        f"[pass@k] dataset={dataset_name} extractor={extract_fn.__name__} "
        f"normalizer={normalize_fn.__name__}"
    )

    model, tokenizer = load_checkpoint(checkpoint_path, device=device)

    # Seed the sampling RNG for reproducible generations when requested.
    if eval_seed is not None:
        set_seed(eval_seed)
        logger.info(f"[pass@k] Seeded generation RNG with eval_seed={eval_seed}")

    dataset = load_dataset(dataset_name, dataset_config, split=eval_split)
    if eval_holdout_size and eval_holdout_size > 0:
        dataset = dataset.train_test_split(
            test_size=eval_holdout_size, seed=split_seed,
        )["test"]
        logger.info(
            f"[pass@k] Held out {eval_holdout_size} examples from {eval_split} "
            f"(seed={split_seed}) for eval"
        )
    if max_eval_samples > 0 and len(dataset) > max_eval_samples:
        dataset = dataset.select(range(max_eval_samples))

    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id == tokenizer.eos_token_id:
        pad_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    eos_id = tokenizer.eos_token_id

    # Encode all prompts
    all_prompt_ids = []
    all_gold = []
    for ex in dataset:
        question = ex[prompt_column]
        gold_text = ex[response_column]
        all_gold.append(extract_fn(gold_text))

        if (use_chat_template
                and hasattr(tokenizer, "chat_template")
                and tokenizer.chat_template):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True, add_generation_prompt=True,
            )
        else:
            ids = tokenizer.encode(
                f"Question: {question}\nAnswer:", add_special_tokens=False,
            )
        all_prompt_ids.append(ids)

    # For each problem, generate n_samples responses
    per_problem_correct = []  # list of int counts

    for prob_idx in range(len(all_prompt_ids)):
        prompt_ids = all_prompt_ids[prob_idx]
        gold = all_gold[prob_idx]
        n_correct = 0

        # Generate n_samples in batches
        samples_remaining = n_samples
        while samples_remaining > 0:
            cur_batch = min(batch_size, samples_remaining)
            # Replicate prompt for the batch
            batch_prompt_ids = [prompt_ids] * cur_batch
            input_ids, attention_mask = _left_pad_batch(
                batch_prompt_ids, pad_id, device,
            )

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                )

            prompt_len = input_ids.shape[1]
            for i in range(cur_batch):
                gen_tokens = _decode_generated(output_ids[i], prompt_len, eos_id, pad_id)
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                pred = extract_fn(gen_text)
                if gold is not None and pred is not None:
                    if normalize_fn(pred) == normalize_fn(gold):
                        n_correct += 1

            samples_remaining -= cur_batch

        per_problem_correct.append(n_correct)

        if prob_idx < 3 or (prob_idx + 1) % 100 == 0:
            logger.info(
                f"  Problem {prob_idx + 1}/{len(all_prompt_ids)}: "
                f"{n_correct}/{n_samples} correct"
            )

    # Compute pass@k for each requested k
    results = {}
    for k in k_values:
        if k > n_samples:
            logger.warning(f"k={k} > n_samples={n_samples}, skipping")
            continue
        per_problem_pass = [
            _estimate_pass_at_k(n_samples, c, k) for c in per_problem_correct
        ]
        results[f"pass@{k}"] = float(np.mean(per_problem_pass))

    results["n_samples"] = n_samples
    results["temperature"] = temperature
    results["eval_seed"] = eval_seed
    results["num_problems"] = len(per_problem_correct)
    results["per_problem_correct"] = per_problem_correct

    logger.info(f"[pass@k] checkpoint={checkpoint_path}")
    for k in k_values:
        key = f"pass@{k}"
        if key in results:
            logger.info(f"  {key} = {results[key]:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="GSM-8K pass@k evaluation")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--prompt_column", type=str, default="question")
    parser.add_argument("--response_column", type=str, default="answer")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--eval_holdout_size", type=int, default=0,
        help="If > 0, evaluate on train_test_split(test_size=N, seed=split_seed)['test']; "
             "matches train_gsm8k.py's holdout when N=500, split_seed=42.",
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--eval_seed", type=int, default=None,
        help="If set, seed the generation sampling RNG for reproducible pass@k. "
             "Leave unset for fresh (non-deterministic) sampling each run.",
    )
    args = parser.parse_args()

    results = evaluate_pass_at_k(
        checkpoint_path=args.checkpoint_path,
        k_values=args.k,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_eval_samples=args.max_eval_samples,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        eval_split=args.eval_split,
        prompt_column=args.prompt_column,
        response_column=args.response_column,
        use_chat_template=args.use_chat_template,
        device=args.device,
        eval_holdout_size=args.eval_holdout_size,
        split_seed=args.split_seed,
        eval_seed=args.eval_seed,
    )

    import json
    import os
    summary = {k: v for k, v in results.items() if k != "per_problem_correct"}
    summary.update({
        "checkpoint_path": args.checkpoint_path,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "eval_split": args.eval_split,
        "eval_holdout_size": args.eval_holdout_size,
        "split_seed": args.split_seed,
        "eval_seed": args.eval_seed,
    })
    print(json.dumps(summary, indent=2))
    if os.path.isdir(args.checkpoint_path):
        ds_slug = args.dataset_name.replace("/", "_")
        seed_suffix = f"_s{args.eval_seed}" if args.eval_seed is not None else ""
        out_file = os.path.join(
            args.checkpoint_path,
            f"pass_at_k_{ds_slug}_n{args.n_samples}_t{args.temperature}{seed_suffix}.json",
        )
        try:
            with open(out_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[pass@k] Wrote summary to {out_file}")
        except OSError as e:
            print(f"[pass@k] Could not write summary file: {e}")
