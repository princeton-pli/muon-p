"""
MATH-500 evaluation utilities: answer extraction, metrics, and generation-based evaluation.

Mirrors eval_gsm8k.py but targets the MATH-500 benchmark (HuggingFaceH4/MATH-500).
Answers are extracted from \\boxed{} in model output and compared to the ground-truth
`answer` field in the dataset.
"""

import math
import re
import logging
from typing import Optional, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import load_dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Answer extraction
# =============================================================================

_SPECIAL_TOKENS = ["<|eot_id|>", "<|endoftext|>", "<|end|>"]


def _clean_special_tokens(text: str) -> str:
    """Strip common special-token artifacts from decoded text."""
    for tok in _SPECIAL_TOKENS:
        if tok in text:
            text = text.split(tok)[0]
    return text.strip()


def _extract_boxed(text: str) -> Optional[str]:
    r"""Extract the content of the last \boxed{...} in `text`, handling nested braces."""
    # Find all \boxed{ occurrences and take the last one
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    start = matches[-1].end()  # position right after the opening {
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    if depth != 0:
        return None  # unbalanced braces

    return text[start : pos - 1].strip()


def _extract_hash_marker(text: str) -> Optional[str]:
    # 2-or-more #'s, optional whitespace, then capture everything up to newline.
    pattern = r"#{2,}[ \t]*([^\n]+)"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_math500_answer(text: str) -> Optional[str]:
    text = _clean_special_tokens(text)
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    hashed = _extract_hash_marker(text)
    if hashed is not None:
        return hashed

    # Fallback: last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def normalize_math_answer(s: str) -> str:
    """Light normalization for MATH answers: strip whitespace and collapse spaces."""
    s = s.strip()
    # Remove surrounding dollar signs (e.g. "$3$" -> "3")
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s


# =============================================================================
# Checkpoint loading
# =============================================================================

def load_checkpoint(checkpoint_path: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load a model and tokenizer from a saved checkpoint directory."""
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
    """Unbiased estimator of pass@k (Chen et al. 2021)."""
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
    max_new_tokens: int = 1024,
    max_eval_samples: int = -1,
    batch_size: int = 8,
    dataset_name: str = "HuggingFaceH4/MATH-500",
    dataset_config: str = "default",
    eval_split: str = "test",
    prompt_column: str = "problem",
    answer_column: str = "answer",
    use_chat_template: bool = False,
    device: str = "cuda",
):
    """Load a checkpoint and evaluate pass@k on MATH-500 with temperature sampling.

    For each problem, generates `n_samples` responses at the given temperature,
    extracts the answer from \\boxed{} in each, and computes pass@k for each
    requested k.

    Returns a dict with per-k pass rates and per-problem details.
    """
    model, tokenizer = load_checkpoint(checkpoint_path, device=device)

    dataset = load_dataset(dataset_name, dataset_config, split=eval_split)
    if max_eval_samples > 0 and len(dataset) > max_eval_samples:
        dataset = dataset.select(range(max_eval_samples))

    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id == tokenizer.eos_token_id:
        pad_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    eos_id = tokenizer.eos_token_id

    # Encode all prompts and collect gold answers
    all_prompt_ids = []
    all_gold = []
    for ex in dataset:
        question = ex[prompt_column]
        gold = ex[answer_column]
        all_gold.append(normalize_math_answer(str(gold)))

        if (use_chat_template
                and hasattr(tokenizer, "chat_template")
                and tokenizer.chat_template):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True, add_generation_prompt=True,
            )
        else:
            ids = tokenizer.encode(
                f"Problem: {question}\nSolution:", add_special_tokens=False,
            )
        all_prompt_ids.append(ids)

    # For each problem, generate n_samples responses
    per_problem_correct = []

    for prob_idx in range(len(all_prompt_ids)):
        prompt_ids = all_prompt_ids[prob_idx]
        gold = all_gold[prob_idx]
        n_correct = 0

        samples_remaining = n_samples
        while samples_remaining > 0:
            cur_batch = min(batch_size, samples_remaining)
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
                pred = extract_math500_answer(gen_text)
                if pred is not None:
                    if normalize_math_answer(pred) == gold:
                        n_correct += 1

            samples_remaining -= cur_batch

        per_problem_correct.append(n_correct)

        if prob_idx < 3 or (prob_idx + 1) % 50 == 0:
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

    parser = argparse.ArgumentParser(description="MATH-500 pass@k evaluation")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
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
        use_chat_template=args.use_chat_template,
        device=args.device,
    )

    import json
    print(json.dumps({k: v for k, v in results.items() if k != "per_problem_correct"}, indent=2))
