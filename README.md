# $\text{Muon}^p$: Muon with Fractional Spectral Powers

[![arXiv](https://img.shields.io/badge/arXiv-2605.08478-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.13867)


$\text{Muon}^p$ is a Muon-based optimizer that extends it to arbitrary fractional spectral powers $p$ for $0 < p < 1$. $\text{Muon}^p$ is especially effective at finetuning, outperforming Muon and other baselines on a variety of tasks. $\text{Muon}^p$ also provides a valuable tool for understanding the optimal spectral geometry during various stages of training.

This repository contains an efficient, distributed implementaion of $\text{Muon}^p$.

## Backends

Select the optimizer with `--muon_backend`, assuming the gradient $G$ has the SVD decomposition $G = U S V^\top$:

| backend          | update                | notes                                             |
| ---------------- | --------------------- | ------------------------------------------------- |
| `newtonschulz5` (Muon)  | `U Vᵀ` (p = 0)        | standard Muon orthogonalization (quintic Newton Schulz 5)     |
| `halfpower` ($\text{Muon}^p$)      | `U Sᵖ Vᵀ`             | fractional power, set `p` via `--muon_exponent`    |
| `zeropower_svd`  | `U Vᵀ` (p = 0)        | exact SVD reference for `newtonschulz5`            |
| `halfpower_svd`  | `U Sᵖ Vᵀ`             | exact SVD reference for `halfpower`                |
| `adamw`          | —                     | AdamW on all parameters (baseline) |

`--muon_exponent` accepts rational strings, e.g. `1/2`, `1/3`, `1/5`, `1/7`, `3/5`,
`1/15`, `13/15`.

## Directory tree

```
.
├── README.md
├── requirements.txt
├── setup.sh                 # create a venv + install dependencies
├── all_arguments.py         # model / data / training argument dataclasses
├── dual_optimizer.py        # Muon, DualOptimizer, MuonTrainer, param splitting
├── utils.py                 # Newton-Schulz polynomials, effective rank, helpers
├── perf_bench.py            # optional optimizer-step performance benchmark
├── train_hf.py              # LM pretraining (FineWeb-Edu, etc.)
├── train_gsm8k.py           # math / reasoning SFT (GSM-8K, NuminaMath, MATH, code)
├── eval_gsm8k.py            # GSM-8K answer extraction, metrics, pass@k
├── eval_math500.py          # MATH-500 answer extraction, metrics, pass@k
├── eval_lm.py               # zero-shot benchmark eval via lm-eval-harness
├── optimize_power.py        # fit Newton-Schulz polynomial coeffs to gradient spectra
├── run_hf.sh                # launch one LM-pretraining run
├── run_gsm8k.sh             # launch one finetuning run
├── run_eval_gsm8k.sh        # launch GSM-8K pass@k eval for a checkpoint
├── run_eval_numina.sh       # launch NuminaMath-CoT pass@k eval for a checkpoint
├── summarize_numina_evals.py # aggregate pass@k results across seed runs
├── submit_hf.sh             # sweep launcher  (reads config/sweep_params_hf.yaml)
├── submit_gsm8k.sh          # sweep launcher  (reads config/sweep_params_gsm8k.yaml)
└── config/
    ├── sweep_params_hf.yaml
    └── sweep_params_gsm8k.yaml
```

## 🚀 Getting started

### 1. Set up the environment

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a virtual environment, installs `requirements.txt`, builds
FlashAttention-2, and runs an import smoke test. FlashAttention is **required by
`train_hf.py`** (it packs examples into variable-length sequences). If the
flash-attn build is slow or you only need the finetuning / eval scripts, skip it:

```bash
SKIP_FLASH=1 ./setup.sh
# install later with:  pip install flash-attn==2.8.3 --no-build-isolation
```

Training and evaluation expect at least one CUDA GPU.


## 📈 Training

Both `run_hf.sh` and `run_gsm8k.sh` read their settings from environment
variables (every variable in the script can be overridden) and launch the
training script with `torchrun`, auto-detecting the visible GPUs. They also carry
an optional `#SBATCH` header so you can `sbatch run_hf.sh` on a Slurm cluster.

### Language modeling tasks — `train_hf.py`

Trains a model on a streaming HF text dataset (default:
SmolLM2-135M on FineWeb-Edu).

```bash
# Standard Muon (Newton-Schulz orthogonalization, p = 0)
muon_backend=newtonschulz5 ./run_hf.sh

# Muon^p (halfpower, exponent 1/3)
muon_backend=halfpower muon_exponent=1/3 halfpower_c=0.66 ./run_hf.sh
```

Override any hyperparameter the same way, e.g.:

```bash
muon_backend=halfpower muon_exponent=1/2 \
  model_name_or_path=HuggingFaceTB/SmolLM2-360M \
  learning_rate=3.6e-3 max_steps=10000 per_device_train_batch_size=8 \
  ./run_hf.sh
```

### Math / reasoning finetuning — `train_gsm8k.py`

Finetunes a pretrained model. The `dataset_name` selects the answer-extraction style automatically (GSM-8K `####`, `\boxed{}` for NuminaMath / MATH, fenced blocks for code).

```bash
# Standard Muon on GSM-8K
muon_backend=newtonschulz5 dataset_name=openai/gsm8k max_steps=600 ./run_gsm8k.sh

# Muon^p on GSM-8K
muon_backend=halfpower muon_exponent=1/3 halfpower_c=0.66 max_steps=600 \
  dataset_name=openai/gsm8k ./run_gsm8k.sh
```

### Sweeps

Edit the YAML files in `config/` (one list item under `runs:` per job) and submit:

```bash
./submit_hf.sh        # one job per run in config/sweep_params_hf.yaml
./submit_gsm8k.sh     # one job per run in config/sweep_params_gsm8k.yaml
```

Run a sweep locally instead of submitting to Slurm with `LOCAL_RUN=1 ./submit_hf.sh`.
The shipped YAMLs each define a `newtonschulz5` run and a `halfpower` run as
templates.

## Evaluation

```bash
# Zero-shot benchmarks (piqa, winogrande, lambada, ...) via lm-eval-harness
python eval_lm.py --model_path outputs/<run_name>

# pass@k on GSM-8K for a saved checkpoint
python eval_gsm8k.py --checkpoint_path outputs/<run_name> --k 1 5 --n_samples 5

# pass@k on MATH-500
python eval_math500.py --checkpoint_path outputs/<run_name> --k 1 --n_samples 1
```

The `run_eval_*.sh` wrappers run the same pass@k evaluation with
per-dataset defaults (and an optional `#SBATCH` header for Slurm). Set the
checkpoint via an environment variable:

```bash
# GSM-8K pass@k
checkpoint_path=outputs/<run_name> ./run_eval_gsm8k.sh

# NuminaMath-CoT pass@k (\boxed{} answers)
checkpoint_path=outputs/<run_name> ./run_eval_numina.sh
```

`train_gsm8k.py` already reports greedy generation accuracy on a held-out split
during training; `eval_gsm8k.py` / `eval_math500.py` add temperature-sampled
pass@k for finished checkpoints. `summarize_numina_evals.py` aggregates the
resulting `pass_at_k_*.json` files across seed runs into a mean ± std table.

### Tuning $\text{Muon}^p$ polynomial coefficients

`optimize_power.py` fits the coefficients of the $\text{Muon}^p$ 
polynomial iteration to real gradient singular-value spectra. Save singular values
during training with `utils.record_grad`, then point the script at the dumped
`.pt` files (run `python optimize_power.py --help` for options).

## Additional Notes

- `--muon_curriculum` switches the backend at a chosen step (e.g. from
  `halfpower` to `newtonschulz5`) for a short orthogonalization "cooldown".
- `--perf_benchmark` (in `train_hf.py`) measures per-step optimizer time and
  memory for a few steps; see `perf_bench.py`.

### 👋 Citation

```
@article{muonp2026,
  title={MuonP: Muon with Fractional Spectral Powers},
  author={Dong, Yihe and Sawin, Will},
  journal={arXiv:2606.13867},
  year={2026}
}
```