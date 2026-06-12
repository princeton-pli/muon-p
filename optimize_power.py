import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments


# ---------------------------------------------------------------------------
# Dataset: load singular-value spectra saved by utils.record_grad
# ---------------------------------------------------------------------------

class GradSingularValueDataset(Dataset):
    """Dataset of singular-value vectors produced by ``utils.record_grad``.

    Each ``.pt`` file in *grad_dir* contains a list of 1-D tensors (the
    singular values of one gradient snapshot).  We normalise every vector so
    that the largest singular value is ≤ 1, matching the normalisation used
    inside ``halfpower_via_newtonschulz5``.
    """

    def __init__(self, grad_dir="grad_logs", prefix="grads"):
        self.samples = []
        pattern = os.path.join(grad_dir, f"{prefix}_*.pt")
        for filepath in sorted(glob.glob(pattern)):
            sv_list = torch.load(filepath, weights_only=True)
            self.samples.extend(sv_list)
            # break
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No gradient data found in '{grad_dir}' with prefix '{prefix}'. "
                "Run training with record_grad first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sv = self.samples[idx].float()
        return {"singular_values": sv}


def collate_fn(batch):
    """Pad singular-value vectors to the same length and stack them."""
    svs = [item["singular_values"] for item in batch]
    max_len = max(sv.shape[0] for sv in svs)
    padded = torch.zeros(len(svs), max_len)
    mask = torch.ones(len(svs), max_len, dtype=torch.bool)
    for i, sv in enumerate(svs):
        padded[i, : sv.shape[0]] = sv
        mask[i, sv.shape[0] :] = False
    return {"singular_values": padded, "mask": mask}


# ---------------------------------------------------------------------------
# Model: learnable scalar Newton-Schulz coefficients
# ---------------------------------------------------------------------------

class PowerApproxModel(nn.Module):
    """Thin ``nn.Module`` whose only learnable parameters are the scalar
    Newton-Schulz coefficients *c* and *d*.

    Forward pass:
        1. Run ``steps`` iterations of the scalar Newton-Schulz recurrence
           using the current (c, d).
        2. Compute the true power ``x ** exponent``.
        3. Return the MSE between them (masked to ignore padding).
    """

    def __init__(self, c_init=-0.2330, d_init=0.8015, steps=6, exponent="1/2"):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c_init))
        self.d = nn.Parameter(torch.tensor(d_init))
        self.steps = steps
        self.exponent = exponent

    # -- iterative approximation ------------------------------------------------

    def approx_power(self, x):
        """Scalar Newton-Schulz iteration with learnable coefficients."""
        y = x / 2
        for _ in range(self.steps):
            if self.exponent == "1/2":
                y = y - (y ** 4 - x ** 2) * (self.c * x + self.d * y)
            elif self.exponent == "1/3":
                y = y - self.c * (y ** 3 - x)
            elif self.exponent == "1/7":
                y = y - self.c * (y ** 7 - x)
            elif self.exponent == "1/15":
                y = y - self.c * (y ** 15 - x)
            elif self.exponent == "13/15":
                y = y - self.c * (y ** 15 - x ** 13)
        return y

    # -- ground truth -----------------------------------------------------------

    @staticmethod
    def true_power(x, exponent):
        if exponent == "1/2":
            return x ** 0.5
        elif exponent == "1/3":
            return x ** (1.0 / 3.0)
        elif exponent == "1/7":
            return x ** (1.0 / 7.0)
        elif exponent == "1/15":
            return x ** (1.0 / 15.0)
        elif exponent == "13/15":
            return x ** (13.0 / 15.0)
        raise ValueError(f"Invalid exponent: {exponent}")

    # -- forward ----------------------------------------------------------------

    def forward(self, singular_values, mask=None):
        approx = self.approx_power(singular_values)
        target = self.true_power(singular_values, self.exponent)
        # breakpoint()
        loss = (approx - target) ** 2
        if mask is not None:
            loss = (loss * mask).sum() #/ mask.sum()
        else:
            loss = loss.mean()
        return {"loss": loss}


# ---------------------------------------------------------------------------
# Grid search: brute-force over (c, d) or c
# ---------------------------------------------------------------------------

def _load_all_singular_values(grad_dir, prefix):
    """Load all singular-value vectors, pad, and return (data, mask) tensors."""
    dataset = GradSingularValueDataset(grad_dir=grad_dir, prefix=prefix)
    batch = collate_fn([dataset[i] for i in range(len(dataset))])
    # batch = collate_fn([dataset[i] for i in range(10)])
    return batch["singular_values"], batch["mask"]


def _approx_power_batch(x, c, d, steps, exponent, a=None, b=None):
    """Vectorized scalar Newton-Schulz over a whole (N, L) tensor."""
    if exponent == "0":
        # Zero power: quintic iteration x_new = a*x + b*x^3 + c*x^5
        y = x.clone()
        for _ in range(steps):
            y = a * y + b * y**3 + c * y**5
        return y
    # y = x / 2
    y = x
    for _ in range(steps):
        if exponent == "1/2":
            y = y - (y ** 4 - x ** 2) * (c * x + d * y)
        elif exponent == "1/3":
            y = y - c * (y ** 3 - x)
        elif exponent == "1/5":
            y = y - c * (y ** 5 - x)
        elif exponent == "1/7":
            y = y - c * (y ** 7 - x)
        elif exponent == "3/5":
            y = y - c * (y ** 5 - x**3)
        elif exponent == "1/15":
            y = y - c * (y ** 15 - x)
        elif exponent == "13/15":
            y = y - c * (y ** 15 - x ** 13)
    return y


def _true_power_batch(x, exponent):
    if exponent == "0":
        return torch.ones_like(x)  # x^0 = 1 
    elif exponent == "1/2":
        return x ** 0.5
    elif exponent == "1/3":
        return x ** (1.0 / 3.0)
    elif exponent == "1/5":
        return x ** (1.0 / 5.0)
    elif exponent == "1/7":
        return x ** (1.0 / 7.0)
    elif exponent == "3/5":
        return x ** (3.0 / 5.0)
    elif exponent == "1/15":
        return x ** (1.0 / 15.0)
    elif exponent == "13/15":
        return x ** (13.0 / 15.0)
    raise ValueError(f"Invalid exponent: {exponent}")


def _masked_mse(approx, target, mask):
    """MSE over valid (non-padded) entries."""
    err = (approx - target) ** 2
    return (err * mask).sum() / mask.sum()


def grid_search(grad_dir, prefix, exponent, steps, grid_size, device="cpu"):
    """Brute-force grid search over coefficients.

    For exponent=="0"   we search a 3-D grid over (a, b, c).
    For exponent=="1/3" we search a 1-D grid over c.
    For exponent=="1/2" we search a 2-D grid over (c, d).

    Returns the best coefficients and loss.
    """
    sv, mask = _load_all_singular_values(grad_dir, prefix)

    sv, mask = sv.to(device), mask.to(device)
    target = _true_power_batch(sv, exponent)

    if exponent == "0":
        # 3-D grid over (a, b, c)
        a_vals = np.linspace(-1.0, 5.0, grid_size)
        b_vals = np.linspace(-5.0, 1.0, grid_size)
        c_vals = np.linspace(-1.0, 1.0, grid_size)
        best_a, best_b, best_c, best_loss = 0.0, 0.0, 0.0, float("inf")
        total = grid_size ** 3
        print(f"Grid search over (a, b, c) ({grid_size}³ = {total} points)")
        for i, a in enumerate(a_vals):
            for b in b_vals:
                for c in c_vals:
                    approx = _approx_power_batch(sv, c, 0.0, steps, exponent, a=a, b=b)
                    loss = _masked_mse(approx, target, mask).item()
                    if loss < best_loss:
                        best_a, best_b, best_c, best_loss = a, b, c, loss
            if (i + 1) % max(1, grid_size // 20) == 0:
                print(f"  progress: {i+1}/{grid_size}  best so far: a={best_a:.6f}, b={best_b:.6f}, c={best_c:.6f}, loss={best_loss:.8f}")
        print(f"\n--- Grid search result (exponent={exponent}) ---")
        print(f"  best a = {best_a:.6f}")
        print(f"  best b = {best_b:.6f}")
        print(f"  best c = {best_c:.6f}")
        print(f"  loss   = {best_loss:.8f}")
        return best_a, best_b, best_c, best_loss

    # c_vals = np.linspace(-1.0, 1.0, grid_size)
    if exponent in ("1/15", "13/15"):
        # Both share a denominator of 15: Newton step ~ 1/15 ≈ 0.067
        c_vals = np.linspace(0.0, 0.2, grid_size)
    elif exponent == "1/7":
        # Newton step ~ 1/7 ≈ 0.143; safe range (0, 2/7 ≈ 0.286)
        c_vals = np.linspace(0.0, 0.4, grid_size)
    else:
        c_vals = np.linspace(-1.0, 2.0, grid_size)

    if exponent in ["1/3", "1/5", "1/7", "3/5", "1/15", "13/15"]:
        # 1-D grid over c
        best_c, best_loss = 0.0, float("inf")
        print(f"Grid search over c   ({grid_size} points)")
        for i, c in enumerate(c_vals):
            approx = _approx_power_batch(sv, c, 0.0, steps, exponent)
            loss = _masked_mse(approx, target, mask).item()
            if loss < best_loss:
                best_c, best_loss = c, loss
            if (i + 1) % max(1, grid_size // 100) == 0:
                print(f"  progress: {i+1}/{grid_size}  best so far: c={best_c:.6f}, loss={best_loss:.8f}")
        print(f"\n--- Grid search result (exponent={exponent}) ---")
        print(f"  best c = {best_c:.6f}")
        print(f"  loss   = {best_loss:.8f}")
        return best_c, None, best_loss

    elif exponent == "1/2":
        # 2-D grid over (c, d)
        c_vals = np.linspace(-0.5, 1, grid_size)
        d_vals = np.linspace(0, .6, grid_size)
        best_c, best_d, best_loss = 0.0, 0.0, float("inf")
        total = grid_size * grid_size
        print(f"Grid search over (c, d) ∈ [-1, 3]×[-3, 3]  ({grid_size}×{grid_size} = {total} points)")
        for i, c in enumerate(c_vals):
            for d in d_vals:
                approx = _approx_power_batch(sv, c, d, steps, exponent)
                loss = _masked_mse(approx, target, mask).item()
                if loss < best_loss:
                    best_c, best_d, best_loss = c, d, loss
            if (i + 1) % max(1, grid_size // 20) == 0:
                print(f"  progress: {i+1}/{grid_size}  best so far: c={best_c:.6f}, d={best_d:.6f}, loss={best_loss:.8f}")
        print(f"\n--- Grid search result (exponent={exponent}) ---")
        print(f"  best c = {best_c:.6f}")
        print(f"  best d = {best_d:.6f}")
        print(f"  loss   = {best_loss:.8f}")
        return best_c, best_d, best_loss


# ---------------------------------------------------------------------------
# Main: wire everything into HuggingFace Trainer
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimize Newton-Schulz scalar coefficients on recorded gradient spectra."
    )
    parser.add_argument("--method", type=str, default="trainer", choices=["trainer", "grid"],
                        help="Optimization method: 'trainer' (gradient-based) or 'grid' (brute-force grid search)")
    parser.add_argument("--grad_dir", type=str, default="grad_logs",
                        help="Directory containing .pt files from record_grad")
    parser.add_argument("--prefix", type=str, default="grads",
                        help="Filename prefix used by record_grad")
    parser.add_argument("--exponent", type=str, default="1/3")
    parser.add_argument("--steps", type=int, default=6,
                        help="Number of Newton-Schulz iterations")
    # Trainer-specific args
    parser.add_argument("--c_init", type=float, default=-0.2330)
    parser.add_argument("--d_init", type=float, default=0.8015)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="power_approx_output")
    # Grid-search-specific args
    parser.add_argument("--grid_size", type=int, default=1000,
                        help="Number of points per axis in the grid search")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for grid search computation (e.g. 'cpu' or 'cuda')")
    args = parser.parse_args()

    if args.method == "grid":
        # ---- Grid search (no gradients) ----
        grid_search(
            grad_dir=args.grad_dir,
            prefix=args.prefix,
            exponent=args.exponent,
            steps=args.steps,
            grid_size=args.grid_size,
            device=args.device,
        )
    else:
        # ---- Gradient-based HuggingFace Trainer ----
        dataset = GradSingularValueDataset(grad_dir=args.grad_dir, prefix=args.prefix)
        print(f"Loaded {len(dataset)} singular-value samples from '{args.grad_dir}'")

        model = PowerApproxModel(
            c_init=args.c_init,
            d_init=args.d_init,
            steps=args.steps,
            exponent=args.exponent,
        )
        print(f"Initial coefficients: c={model.c.item():.6f}, d={model.d.item():.6f}")

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            logging_steps=100,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate_fn,
        )

        trainer.train()

        print("\n--- Optimized coefficients ---")
        print(f"  c = {model.c.item():.6f}")
        print(f"  d = {model.d.item():.6f}")


if __name__ == "__main__":
    main()
