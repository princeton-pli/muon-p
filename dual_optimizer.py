"""
Shared optimizer components: Muon, DualOptimizer, MuonTrainer, parameter splitting,
MuonArguments, and curriculum callback.

Used by both train_hf.py and train_gsm8k.py.
"""

import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback

import utils

logger = logging.getLogger(__name__)


# =============================================================================
# Muon optimizer
# =============================================================================

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7, exponent="1/2", c=None, d=None):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


@torch.compile
def halfpower_via_newtonschulz5(G, steps=6, eps=1e-7, exponent="1/2", c=None, d=None):
    assert len(G.shape) == 2
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    Y = X
    for _ in range(steps):
        if exponent == "1/2":
            Y = utils.poly_half(X, Y, c=c, d=d)
        elif exponent == "1/3":
            Y = utils.poly_third(X, Y, c)
        elif exponent == "1/5":
            Y = utils.poly_fifth(X, Y, c)
        elif exponent == "1/7":
            Y = utils.poly_seventh(X, Y, c)
        elif exponent == "3/5":
            Y = utils.poly_three_fifth(X, Y, c)
        elif exponent == "1/15":
            Y = utils.poly_fifteenth(X, Y, c)
        elif exponent == "13/15":
            Y = utils.poly_thirteen_fifteenth(X, Y, c)
        else:
            # raise ValueError(f"Invalid exponent: {exponent}")
            print(f"Invalid exponent: {exponent}")
    if G.size(0) > G.size(1):
        Y = Y.T
    return Y.to(G.dtype)


def zeropower_via_svd(G, steps=None, exponent="0", c=None, d=None):
    """SVD-based orthogonalization (UV^T). `steps`, `c`, `d` are ignored —
    accepted only to match the signature expected by Muon.step."""
    return utils.zeropower_via_svd(G, steps=steps, exponent=exponent, halfpower_c=c)


def halfpower_via_svd(G, steps=None, exponent="1/3", c=None, d=None):
    """SVD-based fractional-power: U S^exponent V^T. `steps` and `d` are
    ignored; `c` (== halfpower_c) is forwarded but unused by this backend."""
    return utils.halfpower_via_svd(G, exponent=exponent, steps=steps, halfpower_c=c)


zeropower_backends = dict(
    newtonschulz5=zeropower_via_newtonschulz5,
    halfpower=halfpower_via_newtonschulz5,
    zeropower_svd=zeropower_via_svd,
    halfpower_svd=halfpower_via_svd,
)


class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-Schulz.

    Internally runs standard SGD-momentum, then performs an
    orthogonalization post-processing step on each 2-D parameter update.
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='halfpower', backend_steps=6, exponent="1/3", ns_c=None, ns_d=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps, exponent=exponent,
                        ns_c=ns_c, ns_d=ns_d)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            exponent = group['exponent']
            ns_c = group['ns_c']
            ns_d = group['ns_d']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):  # split grouped QKV parameters
                    g_list = []
                    for g1 in g.split(g.size(1)):
                        g_list.append(zeropower_backend(g1, steps=group['backend_steps'], exponent=exponent, c=ns_c, d=ns_d))
                    g = torch.cat(g_list)
                    legacy_scale = g.size(1)**0.5
                elif g.size(1) == 3 * g.size(0):  # split grouped QKV parameters
                    g_list = []
                    for g1 in g.split(g.size(0)):
                        g_list.append(zeropower_backend(g1, steps=group['backend_steps'], exponent=exponent, c=ns_c, d=ns_d))
                    g = torch.cat(g_list)
                    legacy_scale = g.size(0)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'], exponent=exponent, c=ns_c, d=ns_d)
                    legacy_scale = max(g.size(0), g.size(1))**0.5
                # Scaling rule:
                #   - newtonschulz5 / zeropower_svd produce an orthogonal output
                #     (singular values ≈ 1), so the legacy `sqrt(max(m,n))` /
                #     `sqrt(n)` correctly gives `update.square().mean() == 1`.
                #   - halfpower / halfpower_svd produce U S^p V^T whose Frobenius
                #     norm shrinks with p, so the legacy scale silently down-weights
                #     the update by a p-dependent factor. We instead measure ||g||_F
                #     directly so update RMS = 1 holds for any exponent.
                if group['backend'] in ('halfpower', 'halfpower_svd'):
                    scale = (g.size(0) * g.size(1))**0.5 / (g.norm() + 1e-7)
                else:
                    scale = legacy_scale
                p.data.add_(g, alpha=-lr * scale)
        return loss


# =============================================================================
# DualOptimizer
# =============================================================================

class DualOptimizer(torch.optim.Optimizer):
    """Wrapper that steps two optimizers as one, compatible with HF Trainer.

    Properly initializes all base-class attributes so that PyTorch's
    profile_hook_step wrapper, LR schedulers, and accelerate's
    AcceleratedOptimizer all work correctly in distributed (DDP) settings.
    """

    def __init__(self, opt1, opt2):
        self.opt1 = opt1
        self.opt2 = opt2

        # Manually initialize base Optimizer attributes (we can't call
        # super().__init__() because it would create its own param_groups
        # that conflict with the sub-optimizers').
        torch._C._log_api_usage_once("python.optimizer")
        self.defaults = {}
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()
        self._warned_capturable_if_run_uncaptured = True

        # Combined param_groups for the LR scheduler and accelerate
        self.param_groups = opt1.param_groups + opt2.param_groups

        # Patch the step method for profiling hooks (must come after
        # param_groups is set, as some hooks inspect it).
        self._patch_step_function()

    @property
    def state(self):
        combined = defaultdict(dict)
        combined.update(self.opt1.state)
        combined.update(self.opt2.state)
        return combined

    @state.setter
    def state(self, value):
        if not value:
            return
        opt1_params = {p for group in self.opt1.param_groups for p in group['params']}
        opt2_params = {p for group in self.opt2.param_groups for p in group['params']}
        for k, v in value.items():
            if k in opt1_params:
                self.opt1.state[k] = v
            elif k in opt2_params:
                self.opt2.state[k] = v

    def zero_grad(self, set_to_none=True):
        self.opt1.zero_grad(set_to_none=set_to_none)
        self.opt2.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss1 = self.opt1.step(closure=closure)
        self.opt2.step()
        return loss1

    def state_dict(self):
        return {
            'opt1': self.opt1.state_dict(),
            'opt2': self.opt2.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.opt1.load_state_dict(state_dict['opt1'])
        self.opt2.load_state_dict(state_dict['opt2'])
        # Re-sync param_groups after loading
        self.param_groups = self.opt1.param_groups + self.opt2.param_groups

    def add_param_group(self, param_group):
        self.opt1.add_param_group(param_group)
        self.param_groups = self.opt1.param_groups + self.opt2.param_groups


# =============================================================================
# Parameter separation logic
# =============================================================================

# These name patterns identify embedding / unembedding layers to exclude from Muon
EMBEDDING_KEYWORDS = {"embed", "wte", "wpe", "embed_tokens", "embed_positions"}
LM_HEAD_KEYWORDS = {"lm_head", "cls", "score"}


def _is_embedding_or_lm_head(name: str) -> bool:
    """Check if a parameter name belongs to an embedding or LM head layer."""
    name_lower = name.lower()
    parts = name_lower.split(".")
    for kw in EMBEDDING_KEYWORDS | LM_HEAD_KEYWORDS:
        if kw in parts or name_lower.endswith(kw + ".weight"):
            return True
    return False


def split_params_for_muon(model: nn.Module):
    """Split model parameters into AdamW and Muon groups.

    Muon group: 2-D weight matrices that are NOT embeddings or LM head.
    AdamW group: everything else (embeddings, lm_head, biases, LayerNorm, scalars, 1-D).

    Returns:
        (adamw_params, muon_params): two lists of parameters
    """
    adamw_params = []
    muon_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not _is_embedding_or_lm_head(name):
            muon_params.append(param)
            logger.info(f"  [Muon]  {name}  shape={tuple(param.shape)}")
        else:
            adamw_params.append(param)
            logger.info(f"  [AdamW] {name}  shape={tuple(param.shape)}")

    logger.info(f"Muon params: {len(muon_params)}, AdamW params: {len(adamw_params)}")
    return adamw_params, muon_params


# =============================================================================
# Curriculum callback
# =============================================================================

class SwitchBackendCallback(TrainerCallback):
    """Switch Muon backend at a specified training step for curriculum learning."""

    def __init__(self, trainer, switch_step=None, target_backend="halfpower", target_exponent="1/3"):
        self.trainer = trainer
        self.switch_step = switch_step
        self.target_backend = target_backend
        self.target_exponent = target_exponent
        self._switched = False

    def on_step_end(self, args, state, control, **kwargs):
        # Resolve switch_step lazily if not set at init time
        switch_step = self.switch_step
        if switch_step is None or switch_step == 0:
            switch_step = max(1, args.max_steps - 500)

        if self._switched or state.global_step != switch_step:
            return

        optimizer = self.trainer.optimizer
        # accelerate wraps the optimizer in AcceleratedOptimizer; unwrap it
        if hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        if optimizer is None or not isinstance(optimizer, DualOptimizer):
            logger.warning(
                f"[SwitchBackendCallback] Step {state.global_step}: "
                f"expected DualOptimizer but got {type(optimizer).__name__} — skipping"
            )
            return
        muon_opt = optimizer.opt2
        for group in muon_opt.param_groups:
            group["backend"] = self.target_backend
            group["exponent"] = self.target_exponent
        self._switched = True
        logger.info(
            f"[SwitchBackendCallback] Step {state.global_step}: "
            f"switched Muon to {self.target_backend} (exponent={self.target_exponent})"
        )


# =============================================================================
# MuonArguments dataclass
# =============================================================================

@dataclass
class MuonArguments:
    muon_backend: str = field(
        default="halfpower",
        metadata={
            "help": (
                "Backend for the matrix-update arm: 'newtonschulz5' (standard "
                "Muon orthogonalization), 'halfpower' (fractional-power update, "
                "see --muon_exponent), 'zeropower_svd' (exact SVD-based "
                "orthogonalization), or 'halfpower_svd' (exact SVD-based "
                "fractional power). Set to 'adamw' to disable Muon and use "
                "AdamW for all parameters."
            )
        },
    )
    muon_exponent: str = field(
        default="1/3",
        metadata={"help": "Exponent for halfpower backend: '1/2', '1/3', '1/5', '1/7', '3/5', '1/15', or '13/15'"},
    )
    muon_lr_factor: float = field(
        default=0.3,
        metadata={"help": "Muon learning rate = muon_lr_factor * learning_rate"},
    )
    muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Momentum for Muon optimizer"},
    )
    muon_backend_steps: int = field(
        default=6,
        metadata={"help": "Number of Newton-Schulz iteration steps"},
    )
    halfpower_c: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient c for poly_third in halfpower backend (default: 1.090452)"},
    )
    halfpower_d: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient d for poly_half (exponent=1/2) in halfpower backend (default: -0.795918)"},
    )
    muon_curriculum: bool = field(
        default=False,
        metadata={"help": "Enable curriculum: switch Muon backend at a specified step"},
    )
    muon_curriculum_switch_step: Optional[int] = field(
        default=0,
        metadata={"help": "Step at which to switch backend (default: max_steps - 500)"},
    )
    muon_curriculum_target_backend: str = field(
        default="halfpower",
        metadata={"help": "Backend to switch to during curriculum"},
    )
    muon_curriculum_target_exponent: str = field(
        default="1/3",
        metadata={"help": "Exponent to switch to during curriculum (e.g. '1/2', '1/3', '1/5', '1/7', '3/5', '1/15', '13/15')"},
    )
    project_name: str = field(
        default="muon-training",
        metadata={"help": "Wandb project name"},
    )
    log_effective_rank: bool = field(
        default=False,
        metadata={"help": "Log the mean effective rank of Muon-optimized weights and their gradients to wandb"},
    )


# =============================================================================
# MuonTrainer
# =============================================================================

class MuonTrainer(Trainer):
    """HuggingFace Trainer with dual AdamW + Muon optimizer support."""

    def __init__(self, muon_args: MuonArguments, **kwargs):
        self.muon_args = muon_args
        super().__init__(**kwargs)

    def create_optimizer(self):
        """Create optimizer(s).

        Dispatch table for ``muon_args.muon_backend``:
            "adamw" -> single AdamW across every parameter (no orthogonalization)
            other   -> AdamW (embeddings / lm_head / 1-D params) + Muon
                       (newtonschulz5 / halfpower / *_svd for 2-D weights),
                       wrapped together in a DualOptimizer.
        """
        if self.optimizer is not None:
            return self.optimizer

        lr = self.args.learning_rate

        if self.muon_args.muon_backend == "adamw":
            logger.info("Using AdamW for ALL parameters (muon_backend=adamw)")
            all_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                all_params, lr=lr, betas=(0.9, 0.95),
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer

        adamw_params, muon_params = split_params_for_muon(self.model)

        muon_lr = self.muon_args.muon_lr_factor * lr

        adamw_optimizer = torch.optim.AdamW(
            adamw_params, lr=lr, betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )

        secondary = Muon(
            muon_params, lr=muon_lr,
            momentum=self.muon_args.muon_momentum,
            backend=self.muon_args.muon_backend,
            backend_steps=self.muon_args.muon_backend_steps,
            exponent=self.muon_args.muon_exponent,
            ns_c=self.muon_args.halfpower_c,
            ns_d=self.muon_args.halfpower_d,
        )

        self.optimizer = DualOptimizer(adamw_optimizer, secondary)
        return self.optimizer
