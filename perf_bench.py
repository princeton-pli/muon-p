"""
Performance benchmark utilities for Muon-family backends.

Drop-in HuggingFace Trainer callback that measures, for a configurable
warmup + measurement window during training:

  - Per-step optimizer wall-clock time (CUDA-event timed), split by arm
    when a ``DualOptimizer`` is in use:
        * ``opt1_adamw``  - the AdamW arm (embeddings / lm_head / 1-D params)
        * ``opt2_matrix`` - the matrix arm (Muon)
        * ``opt_outer``   - the wrapper itself (sanity check; ~= sum(arms))
  - Full training-step wall time (host ``perf_counter``, CUDA-synced) for
    steps/sec / throughput estimation.
  - Peak GPU memory delta over the measurement window
    (``torch.cuda.max_memory_allocated`` - baseline at start of window).
  - Optimizer state size in bytes (sum across both arms).
  - An analytical Newton-Schulz-5 FLOP estimate for the matrix-arm parameter
    shapes -- useful as an apples-to-apples comparison point between
    ``newtonschulz5`` and ``halfpower`` (both run an NS5-shaped quintic
    iteration on the same shapes); informational for other backends since
    their inner kernels differ.

Designed to be opt-in and side-effect-free:

  * Off by default (``perf_benchmark=False``); when off the callback never
    runs, never patches anything, and adds no overhead.
  * When on, it monkey-patches ``opt1.step`` / ``opt2.step`` (and the outer
    ``DualOptimizer.step`` for sanity), drains CUDA events at end-of-step,
    and *restores* the un-patched steps via ``on_train_end``.
  * Compatible with ``accelerate``-wrapped optimizers (we unwrap ``.optimizer``
    if present).

Usage (typically from ``train_hf.py``)::

    from perf_bench import PerfBenchmarkArguments, PerfBenchmarkCallback

    parser = HfArgumentParser((..., PerfBenchmarkArguments, CustomTrainingArguments))
    *_, perf_args, training_args = parser.parse_args_into_dataclasses()
    ...
    if perf_args.perf_benchmark:
        trainer.add_callback(PerfBenchmarkCallback(perf_args))

CLI knobs (all default-off / sane):
    --perf_benchmark                 enable
    --perf_warmup_steps 3            steps to skip before recording
    --perf_measure_steps 10          steps to record
    --perf_output  /path/to.json     write JSON summary
    --perf_log_to_wandb True         mirror to wandb.run.summary
    --perf_stop_after_measure False  end training after the measurement window
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


# =============================================================================
# CLI arguments
# =============================================================================

@dataclass
class PerfBenchmarkArguments:
    """Argparse-friendly knobs for ``PerfBenchmarkCallback``.

    All flags default to a no-op configuration so adding this dataclass to
    an existing ``HfArgumentParser`` doesn't change behavior.
    """

    perf_benchmark: bool = field(
        default=False,
        metadata={"help": "Enable optimizer-step performance benchmark via PerfBenchmarkCallback."},
    )
    perf_warmup_steps: int = field(
        default=3,
        metadata={
            "help": (
                "Steps to skip at the start of training before recording. Lets "
                "torch.compile JIT, CUDA caches, and accelerate prep settle."
            )
        },
    )
    perf_measure_steps: int = field(
        default=10,
        metadata={"help": "Steps to record after warmup."},
    )
    perf_output: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to write the JSON summary. If unset, defaults to "
                "<training_args.output_dir>/perf_benchmark.json when output_dir "
                "is set, otherwise no file is written."
            )
        },
    )
    perf_log_to_wandb: bool = field(
        default=True,
        metadata={"help": "Mirror summary scalars to wandb.run.summary if W&B is active."},
    )
    perf_stop_after_measure: bool = field(
        default=False,
        metadata={
            "help": (
                "Set control.should_training_stop=True after measurement completes. "
                "Useful for quick benchmark-only runs without burning a full schedule."
            )
        },
    )


# =============================================================================
# Standalone helpers
# =============================================================================

def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Sum the bytes of all tensors in ``optimizer.state``.

    Walks the live ``optimizer.state`` dict (not ``state_dict()`` which
    duplicates the tensors on CPU during serialization), so this reflects
    the actual GPU footprint of optimizer buffers.
    """
    total = 0
    state = getattr(optimizer, "state", {}) or {}
    for s in state.values():
        if isinstance(s, dict):
            for v in s.values():
                if torch.is_tensor(v):
                    total += v.element_size() * v.numel()
    return total


def estimate_ns5_flops_per_iter(rows: int, cols: int) -> int:
    """Analytical FLOP estimate for one Newton-Schulz quintic iteration.

    The iteration on the (smaller-side-first) m x n matrix X is::

        A  = X @ X.T        # m^2 n   FMAs
        AA = A @ A          # m^3     FMAs
        B  = b * A + c * AA # 2 m^2   muls + adds
        BX = B @ X          # m^2 n   FMAs
        X  = a * X + B X    # 2 m n   muls + adds

    Counting one FMA as 2 FLOPs (standard convention), one iteration is
    roughly::

        2 * (m^2 n + m^3 + m^2 n) + 2 m^2 + 2 m n
      = 2 m^3 + 4 m^2 n + 2 m^2 + 2 m n

    NS5 always works on the smaller side, so we set m = min(rows, cols),
    n = max(rows, cols).
    """
    m = min(rows, cols)
    n = max(rows, cols)
    return 2 * m * m * m + 4 * m * m * n + 2 * m * m + 2 * m * n


def estimate_ns5_flops_total(shapes, steps: int) -> int:
    """Sum NS5 FLOPs across all >=2-D parameter shapes for a single optimizer step."""
    return steps * sum(
        estimate_ns5_flops_per_iter(s[0], s[1])
        for s in shapes if len(s) >= 2 and s[0] and s[1]
    )


# =============================================================================
# Callback
# =============================================================================

class PerfBenchmarkCallback(TrainerCallback):
    """Measure optimizer-step wall-clock + memory for a few steps of training.

    The callback hooks ``optimizer.step`` (and the inner arms of a
    ``DualOptimizer`` when applicable) with CUDA events, drains them at
    end-of-step, and emits a one-shot summary at the end of the measurement
    window. It restores the un-patched ``step`` methods on ``on_train_end``.

    See module docstring for the full set of CLI knobs.
    """

    def __init__(self, cfg: PerfBenchmarkArguments):
        self.cfg = cfg

        self._opt_patched: bool = False
        self._is_dual: bool = False
        self._opt2_class: Optional[str] = None
        self._patches: List[Tuple[Any, str, Any]] = []  # (obj, attr_name, original)

        # Per-step CUDA-event timings (ms) keyed by name.
        self._timings_ms: Dict[str, List[float]] = defaultdict(list)
        # Pending events, drained at end-of-step.
        self._pending_events: List[Tuple[str, "torch.cuda.Event", "torch.cuda.Event"]] = []

        # Whole-step wall-clock (host timer with cuda sync).
        self._step_wall_start: Optional[float] = None
        self._step_wall_durations_s: List[float] = []

        # Memory.
        self._mem_baseline_alloc: Optional[int] = None
        self._mem_peak_alloc: Optional[int] = None

        # Optimizer state.
        self._optimizer_state_bytes_after: Optional[int] = None

        # Step bookkeeping. We track our own monotone counter rather than
        # ``state.global_step`` because HF Trainer increments ``global_step``
        # *between* ``on_step_begin`` and ``on_step_end``, which would put the
        # two halves of a single iteration into different relative-step
        # windows. ``_cached_rel`` is the rel-step of the iteration currently
        # in flight (set in ``on_step_begin``, consumed in ``on_step_end``).
        self._begin_count: int = 0
        self._cached_rel: Optional[int] = None
        self._measured_count: int = 0
        self._summary_emitted: bool = False

        # Backend / param info (filled in at first patch).
        self._matrix_param_shapes: List[Tuple[int, ...]] = []
        self._n_matrix_params: int = 0
        self._n_adamw_params: int = 0
        self._n_total_trainable_params: int = 0
        self._matrix_backend: Optional[str] = None

    # ------------------------------------------------------------------
    # Window predicates
    # ------------------------------------------------------------------

    def _is_in_warmup(self, rel_step: int) -> bool:
        return rel_step < self.cfg.perf_warmup_steps

    def _is_measuring(self, rel_step: int) -> bool:
        start = self.cfg.perf_warmup_steps
        return start <= rel_step < start + self.cfg.perf_measure_steps

    def _is_done(self, rel_step: int) -> bool:
        return rel_step >= self.cfg.perf_warmup_steps + self.cfg.perf_measure_steps

    # ------------------------------------------------------------------
    # Optimizer patching
    # ------------------------------------------------------------------

    def _wrap_step(self, opt: torch.optim.Optimizer, name: str) -> None:
        """Replace ``opt.step`` with a CUDA-event-timed wrapper.

        Recording happens unconditionally inside the wrapper; the callback
        decides whether to record by toggling ``self._record_events``. We
        keep the inner code path branch-free at step time so the overhead
        is just two ``Event.record()`` calls when active.
        """
        orig_step = opt.step
        cb = self

        def wrapped(*args, **kwargs):
            if cb._record_events:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                ret = orig_step(*args, **kwargs)
                end_evt.record()
                cb._pending_events.append((name, start_evt, end_evt))
                return ret
            return orig_step(*args, **kwargs)

        opt.step = wrapped
        self._patches.append((opt, "step", orig_step))

    def _restore_steps(self) -> None:
        for opt, attr, orig in self._patches:
            try:
                setattr(opt, attr, orig)
            except Exception:
                pass
        self._patches.clear()

    def _patch_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: Optional[torch.nn.Module],
    ) -> None:
        # Unwrap accelerate's AcceleratedOptimizer if present.
        opt = optimizer
        if hasattr(opt, "optimizer"):
            opt = opt.optimizer

        # Lazy import to avoid pulling dual_optimizer at import time.
        from dual_optimizer import DualOptimizer

        if isinstance(opt, DualOptimizer):
            self._is_dual = True
            self._opt2_class = type(opt.opt2).__name__
            self._wrap_step(opt.opt1, "opt1_adamw")
            self._wrap_step(opt.opt2, "opt2_matrix")
            self._wrap_step(opt, "opt_outer")  # sanity: ~= sum(arms)
            for grp in opt.opt2.param_groups:
                for p in grp["params"]:
                    if p.ndim >= 2:
                        self._matrix_param_shapes.append(tuple(p.shape))
            self._n_matrix_params = sum(
                p.numel() for grp in opt.opt2.param_groups for p in grp["params"]
            )
            self._n_adamw_params = sum(
                p.numel() for grp in opt.opt1.param_groups for p in grp["params"]
            )
            # Try to read the configured backend name from the matrix arm's
            # first param-group (Muon stores it; ROOT/PolarGrad/AdaMuon don't).
            grp0 = (opt.opt2.param_groups or [{}])[0]
            self._matrix_backend = grp0.get("backend") or self._opt2_class
        else:
            self._is_dual = False
            self._opt2_class = type(opt).__name__
            self._wrap_step(opt, "opt_only")
            self._n_adamw_params = sum(
                p.numel() for grp in opt.param_groups for p in grp["params"]
            )
            self._matrix_backend = self._opt2_class

        if model is not None:
            self._n_total_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

        logger.info(
            "[perf_bench] patched optimizer for benchmark: "
            f"is_dual={self._is_dual}, matrix_backend={self._matrix_backend}, "
            f"matrix_params={self._n_matrix_params:,}, adamw_params={self._n_adamw_params:,}"
        )

    @property
    def _record_events(self) -> bool:
        """Whether the wrapped step() should record CUDA events right now.

        Read at every wrapped ``optimizer.step`` call, so this property must
        be cheap and consistent within a single training iteration.
        """
        if self._cached_rel is None or self._summary_emitted:
            return False
        return self._is_measuring(self._cached_rel)

    # ------------------------------------------------------------------
    # Trainer callback hooks
    # ------------------------------------------------------------------

    def on_step_begin(self, args, state, control, optimizer=None, model=None, **kwargs):
        if not self.cfg.perf_benchmark:
            return

        # Patch lazily on the very first step we see (after accelerate.prepare).
        if not self._opt_patched and optimizer is not None:
            self._patch_optimizer(optimizer, model)
            self._opt_patched = True

        # Use a self-incrementing counter; see comment on ``_begin_count``.
        rel_step = self._begin_count
        self._begin_count += 1
        self._cached_rel = rel_step

        if self._is_done(rel_step) and not self._summary_emitted:
            self._capture_optimizer_state(optimizer)
            self._emit_summary(args)
            if self.cfg.perf_stop_after_measure:
                control.should_training_stop = True
            return

        if self._is_in_warmup(rel_step):
            return

        # Measuring window:
        if self._is_measuring(rel_step):
            # On the first measured step, anchor memory + cuda timer.
            if rel_step == self.cfg.perf_warmup_steps and self._mem_baseline_alloc is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    self._mem_baseline_alloc = torch.cuda.memory_allocated()
            # Anchor step wall time after a sync so fwd/bwd start cleanly.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._step_wall_start = time.perf_counter()

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        if not self.cfg.perf_benchmark or self._cached_rel is None:
            return

        rel_step = self._cached_rel
        # Consume the cached rel-step so a stray on_step_end without a
        # matching on_step_begin (shouldn't happen, but be defensive)
        # doesn't accidentally re-record.
        self._cached_rel = None

        if not self._is_measuring(rel_step):
            return

        # Drain CUDA events emitted during this step's optimizer.step calls.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for name, s_evt, e_evt in self._pending_events:
            try:
                self._timings_ms[name].append(s_evt.elapsed_time(e_evt))
            except Exception as exc:
                logger.warning(f"[perf_bench] dropped event {name}: {exc}")
        self._pending_events.clear()

        if self._step_wall_start is not None:
            self._step_wall_durations_s.append(time.perf_counter() - self._step_wall_start)
            self._step_wall_start = None

        if torch.cuda.is_available():
            self._mem_peak_alloc = torch.cuda.max_memory_allocated()

        self._measured_count += 1

        if self._measured_count >= self.cfg.perf_measure_steps and not self._summary_emitted:
            self._capture_optimizer_state(optimizer)
            self._emit_summary(args)
            if self.cfg.perf_stop_after_measure:
                control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if not self.cfg.perf_benchmark:
            return
        if not self._summary_emitted:
            self._emit_summary(args)
        # Always restore — even if we never measured (e.g., training stopped
        # short of warmup_steps + measure_steps).
        self._restore_steps()

    # ------------------------------------------------------------------
    # Summary collection / emission
    # ------------------------------------------------------------------

    def _capture_optimizer_state(self, optimizer) -> None:
        """Sum optimizer-state bytes across both arms (or single optimizer).

        Called after the measurement window so all per-param buffers have
        been allocated (e.g. AdaMuon's ``v_buffer`` is created lazily on
        the first step).
        """
        if optimizer is None:
            return
        opt = optimizer
        if hasattr(opt, "optimizer"):
            opt = opt.optimizer
        try:
            from dual_optimizer import DualOptimizer
            if isinstance(opt, DualOptimizer):
                self._optimizer_state_bytes_after = (
                    optimizer_state_bytes(opt.opt1) + optimizer_state_bytes(opt.opt2)
                )
            else:
                self._optimizer_state_bytes_after = optimizer_state_bytes(opt)
        except Exception as exc:
            logger.warning(f"[perf_bench] could not measure optimizer state size: {exc}")

    def _emit_summary(self, args) -> None:
        self._summary_emitted = True

        def _stats(xs: List[float]) -> Dict[str, float]:
            if not xs:
                return {"n": 0}
            return {
                "n": len(xs),
                "mean": float(statistics.fmean(xs)),
                "median": float(statistics.median(xs)),
                "stdev": float(statistics.stdev(xs)) if len(xs) >= 2 else 0.0,
                "min": float(min(xs)),
                "max": float(max(xs)),
            }

        timings_ms = {name: _stats(vals) for name, vals in self._timings_ms.items()}
        wall_ms = _stats([t * 1000.0 for t in self._step_wall_durations_s])

        steps_per_s: Optional[float] = None
        if self._step_wall_durations_s:
            mean_s = statistics.fmean(self._step_wall_durations_s)
            steps_per_s = (1.0 / mean_s) if mean_s > 0 else None

        peak_mb = (self._mem_peak_alloc / (1024 ** 2)) if self._mem_peak_alloc else None
        baseline_mb = (
            self._mem_baseline_alloc / (1024 ** 2) if self._mem_baseline_alloc else None
        )
        delta_mb = (peak_mb - baseline_mb) if (peak_mb is not None and baseline_mb is not None) else None
        opt_state_mb = (
            self._optimizer_state_bytes_after / (1024 ** 2)
            if self._optimizer_state_bytes_after is not None else None
        )

        ns5_flops_per_step = estimate_ns5_flops_total(self._matrix_param_shapes, steps=5)

        summary: Dict[str, Any] = {
            "matrix_backend": self._matrix_backend,
            "is_dual_optimizer": self._is_dual,
            "outer_optimizer_class": self._opt2_class,
            "n_total_trainable_params": self._n_total_trainable_params,
            "n_matrix_params": self._n_matrix_params,
            "n_adamw_params": self._n_adamw_params,
            "n_matrix_param_tensors": len(self._matrix_param_shapes),
            "warmup_steps": self.cfg.perf_warmup_steps,
            "measure_steps": self.cfg.perf_measure_steps,
            "step_wall_ms": wall_ms,
            "steps_per_sec": steps_per_s,
            "optimizer_step_ms": timings_ms,
            "peak_memory_mb": peak_mb,
            "baseline_memory_mb": baseline_mb,
            "memory_delta_mb": delta_mb,
            "optimizer_state_mb": opt_state_mb,
            "ns5_flops_per_step_estimate": ns5_flops_per_step,
        }

        self._print_summary(summary)
        self._save_summary(summary, args)
        self._log_summary_to_wandb(summary)

    # ------------------------------------------------------------------
    # Output sinks
    # ------------------------------------------------------------------

    def _print_summary(self, s: Dict[str, Any]) -> None:
        wall = s["step_wall_ms"]
        lines = [
            "",
            "=" * 78,
            "  Performance Benchmark Summary",
            "=" * 78,
            f"  matrix backend       : {s['matrix_backend']}",
            f"  outer optimizer class: {s['outer_optimizer_class']} (dual={s['is_dual_optimizer']})",
            f"  trainable params     : {s['n_total_trainable_params']:,}",
            f"    matrix arm params  : {s['n_matrix_params']:,} ({s['n_matrix_param_tensors']} tensors)",
            f"    adamw arm params   : {s['n_adamw_params']:,}",
            "",
            f"  warmup / measure     : {s['warmup_steps']} / {s['measure_steps']} steps",
        ]
        if wall.get("n", 0) > 0:
            lines.append(
                f"  step wall time (ms)  : "
                f"mean={wall['mean']:.2f}  median={wall['median']:.2f}  "
                f"std={wall['stdev']:.2f}  [{wall['min']:.2f}, {wall['max']:.2f}]"
            )
        if s["steps_per_sec"] is not None:
            lines.append(f"  steps/sec            : {s['steps_per_sec']:.3f}")
        for name, st in s["optimizer_step_ms"].items():
            if st.get("n", 0) == 0:
                continue
            lines.append(
                f"  {name:<20s} : "
                f"mean={st['mean']:.3f}ms  median={st['median']:.3f}ms  "
                f"std={st['stdev']:.3f}ms  [{st['min']:.3f}, {st['max']:.3f}]"
            )
        if s["peak_memory_mb"] is not None:
            lines.append(f"  peak memory          : {s['peak_memory_mb']:.1f} MB")
        if s["memory_delta_mb"] is not None:
            lines.append(f"  memory delta         : {s['memory_delta_mb']:.1f} MB (over baseline)")
        if s["optimizer_state_mb"] is not None:
            lines.append(f"  optimizer state      : {s['optimizer_state_mb']:.1f} MB")
        # The NS5 FLOP estimate uses 5 iterations and assumes the matrix arm
        # runs an NS5-style quintic on the smaller side. It is most directly
        # comparable for 'newtonschulz5' vs 'halfpower' (which both run the
        # same iteration shape per parameter). For ROOT / PolarGrad /
        # AdaMuon it is informational -- those backends differ in inner
        # kernel cost (extra matmuls, QR/Cholesky, sign(), etc.).
        lines.append(
            f"  NS5 FLOPs/step est.  : {s['ns5_flops_per_step_estimate']:.3e} "
            f"(for newtonschulz5/halfpower comparison; informational otherwise)"
        )
        lines.append("=" * 78)
        logger.info("\n".join(lines))

    def _save_summary(self, summary: Dict[str, Any], args) -> None:
        out_path = self.cfg.perf_output
        if out_path is None and getattr(args, "output_dir", None):
            out_path = os.path.join(args.output_dir, "perf_benchmark.json")
        if out_path is None:
            return
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"[perf_bench] wrote benchmark summary to {out_path}")
        except Exception as exc:
            logger.warning(f"[perf_bench] failed to write summary to {out_path}: {exc}")

    def _log_summary_to_wandb(self, summary: Dict[str, Any]) -> None:
        if not self.cfg.perf_log_to_wandb:
            return
        try:
            import wandb  # noqa: F401
            if wandb.run is None:
                return
        except Exception:
            return
        try:
            for k, v in summary.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, dict):
                            for k3, v3 in v2.items():
                                wandb.run.summary[f"perf/{k}/{k2}/{k3}"] = v3
                        else:
                            wandb.run.summary[f"perf/{k}/{k2}"] = v2
                else:
                    wandb.run.summary[f"perf/{k}"] = v
        except Exception as exc:
            logger.warning(f"[perf_bench] failed to log to wandb: {exc}")
