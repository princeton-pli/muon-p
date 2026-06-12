import os
import torch
from typing import Optional

def cube_root(X):
    U, S, V = torch.linalg.svd(X)
    return U@torch.diag_embed(torch.pow(S,1/3))@V


def _parse_exponent(exponent) -> float:
    """Parse an exponent that may be a number, a decimal string, or a rational
    string like '1/3', '3/5', '1/15', '13/15', '1/7', etc.
    """
    if isinstance(exponent, (int, float)):
        return float(exponent)
    s = str(exponent).strip()
    if "/" in s:
        num, denom = s.split("/", 1)
        return float(num) / float(denom)
    return float(s)


def zeropower_via_svd(G, steps=None, exponent="0", halfpower_c=None):
    orig_dtype = G.dtype
    U, S, V = G.float().svd()
    return (U @ V.T).to(orig_dtype)


def halfpower_via_svd(G, exponent="1/3", steps=None, halfpower_c=None):
    orig_dtype = G.dtype
    U, S, V = G.float().svd()
    p = _parse_exponent(exponent)
    return ((U * S.pow(p)) @ V.T).to(orig_dtype)


@torch.compile
def scalar_zero(x):
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Scalar version of the quintic Newton-Schulz iteration for zero power.
    # Matrix form: X = a*X + b*(X@X.T)@X + c*(X@X.T)@(X@X.T)@X
    # On singular values: x_new = a*x + b*x^3 + c*x^5
    return a * x + b * x**3 + c * x**5

@torch.compile
def scalar_half(x, y):
    c, d = (-3.000000,  -0.795918)  # empirical coefficients for half power
    # x, y are matrices, c, d are scalars.
    f = y - (y**4 - x**2) * (c*x+d*y)
    return f

@torch.compile
def scalar_third(x, y, c=None):
    if c is None:
        c = 1.090452 # empirical coefficient for cube root
    # x, y are matrices, c, d are scalars.
    f = y - c * (y**3 - x)
    return f


@torch.compile
def scalar_fifth(x, y, c=None):
    if c is None:
        c = 1.090452 # empirical coefficient for fifth root
    # x, y are matrices, c, d are scalars.
    f = y - c * (y**5 - x)
    return f


@torch.compile
def scalar_seventh(x, y, c=None):
    """Scalar Newton-Schulz step for 1/7 power: f = y - c * (y^7 - x).

    Fixed point y* satisfies y*^7 = x. Theoretical Newton step at y* ~ 1
    gives c = 1/7 ≈ 0.143; convergence range is c ∈ (0, 2/7 ≈ 0.286).
    """
    if c is None:
        c = 1.0 / 7
    f = y - c * (y**7 - x)
    return f


@torch.compile
def scalar_fifteenth(x, y, c=None):
    """Scalar Newton-Schulz step for 1/15 power: f = y - c * (y^15 - x).

    Fixed point y* satisfies y*^15 = x. Theoretical Newton step at y* ~ 1
    gives c = 1/15; tune empirically via grid search.
    """
    if c is None:
        c = 1.0 / 15
    f = y - c * (y**15 - x)
    return f


@torch.compile
def scalar_thirteen_fifteenth(x, y, c=None):
    """Scalar Newton-Schulz step for 13/15 power: f = y - c * (y^15 - x^13).

    Fixed point y* satisfies y*^15 = x^13, i.e. y* = x^{13/15}. Theoretical
    Newton step at y* ~ 1 gives c = 1/15.
    """
    if c is None:
        c = 1.0 / 15
    f = y - c * (y**15 - x**13)
    return f


@torch.compile
def poly_half(x, y, c=None, d=None):
    #c, d = (-0.2330, 0.8015)  # empirical coefficients for half power
    if c is None or d is None:
        c, d = (3.000000, -0.795918)  # empirical coefficients for half power
    # x, y are matrices, c, d are scalars.
    Y = y@y.mT
    f = y - (Y@Y - x@x.mT) @ (c*x+d*y)

    return f

@torch.compile
def poly_fifth(x, y, c=None):
    # x, y are matrices, c, d are scalars.
    if c is None:
        c = 0.2 
    Y = y@y.mT
    f = y - c * (Y@Y@y - x)
    return f


@torch.compile
def poly_seventh(x, y, c=None):
    """Newton-Schulz step for the 1/7 power: f = y - c * (y^7 - x).

    In matrix form, y^7 is realized as (y @ y.mT)^3 @ y, which equals
    U S^7 V^T for the SVD y = U S V^T. The fixed point therefore satisfies
    y* = x^{1/7} (in the singular-value sense).

    Default c = 1/7 (theoretical Newton step at unit singular values);
    safe range c ∈ (0, 2/7 ≈ 0.286). Tune empirically with optimize_power.py.
    """
    if c is None:
        c = 1.0 / 7
    A = y @ y.mT       # (y@y.mT)^1, acts as y^2
    A2 = A @ A         # (y@y.mT)^2, acts as y^4
    y7 = A2 @ A @ y    # (y@y.mT)^3 @ y = y^7
    f = y - c * (y7 - x)
    return f

@torch.compile
def poly_three_fifth(x, y, c=None):
    # x, y are matrices, c are scalars.
    if c is None:
        c = 0.2 
    f = y - c * (y@y.mT@y@y.mT@y - x@x.mT@x)
    return f

@torch.compile
def poly_third(x, y, c=None):
    # c = 0.040830 # empirical coefficient for cube root
    # x, y are matrices, c, d are scalars.
    if c is None:
        c = 0.66 
    f = y - c * (y@y.mT@y - x)
    return f


@torch.compile
def poly_fifteenth(x, y, c=None):
    """Newton-Schulz step for the 1/15 power: f = y - c * (y^15 - x).

    In matrix form, y^15 is realized as (y @ y.mT)^7 @ y, which equals
    U S^15 V^T for the SVD y = U S V^T. The fixed point therefore satisfies
    y* = x^{1/15} (in the singular-value sense).

    Default c = 1/15 (theoretical Newton step at unit singular values);
    tune empirically with optimize_power.py for best convergence.
    """
    if c is None:
        c = 1.0 / 15
    A = y @ y.mT          # (y@y.mT)^1, acts as y^2
    A2 = A @ A            # (y@y.mT)^2, acts as y^4
    A4 = A2 @ A2          # (y@y.mT)^4, acts as y^8
    y15 = A4 @ A2 @ A @ y # (y@y.mT)^7 @ y = y^15
    f = y - c * (y15 - x)
    return f


@torch.compile
def poly_thirteen_fifteenth(x, y, c=None):
    """Newton-Schulz step for the 13/15 power: f = y - c * (y^15 - x^13).

    Matrix forms (acting on singular values via U S^k V^T):
        y^15 = (y @ y.mT)^7 @ y
        x^13 = (x @ x.mT)^6 @ x

    Fixed point y* satisfies y*^15 = x^13, i.e. y* = x^{13/15}.
    Default c = 1/15; tune empirically with optimize_power.py.
    """
    if c is None:
        c = 1.0 / 15
    # y^15 = (y@y.mT)^7 @ y   (6 matmuls)
    A = y @ y.mT
    A2 = A @ A
    A4 = A2 @ A2
    y15 = A4 @ A2 @ A @ y
    # x^13 = (x@x.mT)^6 @ x   (5 matmuls)
    B = x @ x.mT
    B2 = B @ B
    B4 = B2 @ B2
    x13 = B4 @ B2 @ x
    f = y - c * (y15 - x13)
    return f


def halfpower_via_newtonschulz5(G, c, d, steps=10, eps=1e-7, exponent="1/2"):
    assert len(G.shape) == 2
    # a, b, c = (3.4445, -4.7750,  2.0315)
    # c, d = (-0.158407, 0.1646705)

    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    # Y = X / 2
    Y = X

    for _ in range(steps):
        if exponent == "1/2":
            Y = poly_half(X, Y)
        elif exponent == "1/3":
            Y = poly_third(X, Y, c=c)
        elif exponent == "1/5":
            Y = poly_fifth(X, Y, c=c)
        elif exponent == "1/7":
            Y = poly_seventh(X, Y, c=c)
        elif exponent == "3/5":
            Y = poly_three_fifth(X, Y, c=c)
        elif exponent == "1/15":
            Y = poly_fifteenth(X, Y, c=c)
        elif exponent == "13/15":
            Y = poly_thirteen_fifteenth(X, Y, c=c)
        else:
            raise ValueError(f"Invalid exponent: {exponent}")
        # print(Y.norm())
    if G.size(0) > G.size(1):
        Y = Y.T
    return Y.to(G.dtype)


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def effective_rank(W, logger=None):
    """Computes the effective rank of a weight matrix.
    Args:
        W: 2-D weight matrix (or any tensor that will be reshaped to 2-D).

    Returns:
        Scalar tensor with the effective rank.
    """
    if W.ndim != 2:
        # W = W.reshape(W.shape[0], -1)
        if logger is not None:
            logger.info(f"Shape not rectangular! {W.ndim}")
        return W.shape[-1]
    S = torch.linalg.svdvals(W.float())
    p = S / S.sum()
    p = p[p > 0]
    entropy = -(p * p.log()).sum()
    return entropy.exp()


def record_grad(grad, grad_list, N=10000, save_dir="grad_logs", prefix="grads"):
    """
    Append a gradient to grad_list. Once the list reaches size N,
    save it to disk as a .pt file and reset the list.

    Args:
        grad: Gradient tensor to record.
        grad_list: List that accumulates gradients.
        N: Number of gradients to accumulate before saving.
        save_dir: Directory to save gradient files.
        prefix: Filename prefix for saved files.

    Returns:
        grad_list (reset to [] after saving, otherwise the appended list).
    """
    grad = grad.float() / (grad.norm() + 1e-7) 
    U, D, V = torch.linalg.svd(grad)

    grad_list.append(D.detach().cpu().clone())
    if len(grad_list) >= N:
        os.makedirs(save_dir, exist_ok=True)
        # Find next available file index
        existing = [f for f in os.listdir(save_dir) if f.startswith(prefix) and f.endswith(".pt")]
        idx = len(existing)
        save_path = os.path.join(save_dir, f"{prefix}_{idx:06d}.pt")
        torch.save(grad_list.copy(), save_path)
        grad_list.clear()
    return grad_list