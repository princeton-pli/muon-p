"""
Shared argument dataclasses for training scripts.

Used by train_hf.py and train_gsm8k.py.
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


# =============================================================================
# Training arguments
# =============================================================================

class CustomTrainingArguments(TrainingArguments):
    disable_tqdm: Optional[bool] = field(
        default=True,
        metadata={"help": "Disable TQDM for cleaner logging"},
    )


# =============================================================================
# Model arguments
# =============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="gpt2",
        metadata={"help": "HuggingFace model name or path (e.g. gpt2, meta-llama/Llama-3.2-1B)"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name (defaults to model_name_or_path)"},
    )
    from_pretrained: bool = field(
        default=False,
        metadata={"help": "Load pretrained weights (True) or train from scratch (False)"},
    )
    num_hidden_layers: Optional[int] = field(
        default=-1,
        metadata={"help": "Override the number of hidden layers in the model config"},
    )


# =============================================================================
# Data arguments: language modeling (train_hf.py)
# =============================================================================

@dataclass
class LMDataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace dataset name (e.g. wikitext, HuggingFaceFW/fineweb-edu)"},
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config name (e.g. wikitext-2-raw-v1, sample-10BT)"},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training"},
    )
    eval_split: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset split for evaluation (default: None = no eval)"},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length for tokenization"},
    )
    text_column: str = field(
        default="text",
        metadata={"help": "Column name containing text in the dataset"},
    )
    eval_holdout_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of examples to holdout from train for eval"},
    )
    train_tokenized_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save/load tokenized training dataset cache"},
    )
    eval_tokenized_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save/load tokenized evaluation dataset cache"},
    )


# =============================================================================
# Data arguments: GSM-8K finetuning (train_gsm8k.py)
# =============================================================================

@dataclass
class GSM8KDataArguments:
    dataset_name: str = field(
        default="openai/gsm8k",
        metadata={"help": (
            "HuggingFace dataset name. Used both for loading and to select the "
            "answer-extraction style at generation-eval time: names containing "
            "'numina' or 'math' use \\boxed{...} extraction; everything else "
            "(e.g. 'openai/gsm8k') uses '####' extraction."
        )},
    )
    dataset_config: Optional[str] = field(
        default="main",
        metadata={"help": "Dataset config name (use empty string for datasets without configs, e.g. NuminaMath)"},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Training split name"},
    )
    eval_split: str = field(
        default="test",
        metadata={"help": "Evaluation split name"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization"},
    )
    prompt_column: str = field(
        default="question",
        metadata={"help": "Column name containing the math question"},
    )
    response_column: str = field(
        default="answer",
        metadata={"help": "Column name containing the chain-of-thought answer"},
    )
    eval_holdout_size: Optional[int] = field(
        default=None,
        metadata={"help": "If set, carve this many examples from train for eval (ignores eval_split)"},
    )
    train_tokenized_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save/load tokenized training dataset cache"},
    )
    eval_tokenized_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save/load tokenized evaluation dataset cache"},
    )
    use_chat_template: bool = field(
        default=False,
        metadata={"help": "Format examples using the tokenizer's chat template"},
    )
    max_eval_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "Limit evaluation to this many samples (for faster dev loops)"},
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum new tokens to generate during generation-based evaluation"},
    )
