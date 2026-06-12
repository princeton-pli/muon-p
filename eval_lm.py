"""
Evaluate a HuggingFace model checkpoint using lm-evaluation-harness.
Optionally compare two checkpoints side-by-side.

Usage:
    # Single model evaluation
    python eval_lm.py --model_path outputs/SmolLM2-135M_halfpower_exp1-3_lr5e-4_fineweb_curr_c0.66
    python eval_lm.py --model_path outputs/SmolLM2-135M_halfpower_exp1-3_lr5e-4_fineweb_curr_c0.66 --tasks hellaswag,arc_easy,piqa,winogrande

    # Compare two checkpoints
    python eval_lm.py --model_path outputs/checkpoint_A --compare_path outputs/checkpoint_B
    python eval_lm.py --model_path outputs/checkpoint_A --compare_path HuggingFaceTB/SmolLM2-135M --tasks hellaswag,arc_easy
"""

import argparse
import json
import os

import lm_eval


DEFAULT_TASKS = [
    # "hellaswag",
    # "arc_easy",
    "piqa",
    "winogrande",
    "lambada_openai",
    "boolq",
    # "triviaqa", # too slow!
    ## "truthfulqa", # too slow!
    # "mathqa",
    # "humaneval",
    # "gsm8k",
    #"commonsense_qa",
    "openbookqa",
    # "mmlu",
]


def evaluate_model(model_path, tasks, batch_size, num_fewshot, device):
    """Run lm-eval-harness on a single model and return results."""
    print(f"\nEvaluating: {model_path}")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        device=device,
    )
    return results


def evaluate_single_task(model_path, task, batch_size, num_fewshot, device):
    """Run lm-eval-harness on a single model for a single task."""
    return lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=[task],
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        device=device,
    )


def extract_metrics(results):
    """Extract (task, metric) -> value dict from lm-eval results, filtering to primary metrics."""
    metrics = {}
    for task_name, task_results in results["results"].items():
        for metric, value in task_results.items():
            if metric.endswith(",none") and not metric.endswith("_stderr,none"):
                if isinstance(value, float):
                    clean_metric = metric.replace(",none", "")
                    metrics[(task_name, clean_metric)] = value
    return metrics


def print_single_results(results, model_path):
    """Print results table for a single model."""
    metrics = extract_metrics(results)
    print("\n" + "=" * 70)
    print(f"  {model_path}")
    print("=" * 70)
    print(f"{'Task':<25} {'Metric':<20} {'Value':>10}")
    print("-" * 70)
    for (task, metric), value in sorted(metrics.items()):
        print(f"{task:<25} {metric:<20} {value:>10.4f}")
    print("=" * 70)


def print_task_comparison(results_a, results_b, path_a, path_b, task_name, task_idx, total_tasks):
    """Print comparison for a single task. Returns list of diffs for this task."""
    metrics_a = extract_metrics(results_a)
    metrics_b = extract_metrics(results_b)
    all_keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

    label_a = os.path.basename(path_a.rstrip("/"))
    label_b = os.path.basename(path_b.rstrip("/"))
    max_label = 18
    short_a = label_a[:max_label] if len(label_a) > max_label else label_a
    short_b = label_b[:max_label] if len(label_b) > max_label else label_b

    header_width = 95
    print(f"\n--- [{task_idx}/{total_tasks}] {task_name} " + "-" * max(0, header_width - len(task_name) - 12))
    print(f"{'Task':<25} {'Metric':<15} {short_a:>18} {short_b:>18} {'Diff (B-A)':>12}")

    diffs = []
    for (task, metric) in all_keys:
        val_a = metrics_a.get((task, metric))
        val_b = metrics_b.get((task, metric))
        str_a = f"{val_a:>18.4f}" if val_a is not None else f"{'N/A':>18}"
        str_b = f"{val_b:>18.4f}" if val_b is not None else f"{'N/A':>18}"
        if val_a is not None and val_b is not None:
            diff = val_b - val_a
            diffs.append(diff)
            sign = "+" if diff > 0 else ""
            print(f"{task:<25} {metric:<15} {str_a} {str_b} {sign}{diff:>11.4f}")
        else:
            print(f"{task:<25} {metric:<15} {str_a} {str_b} {'N/A':>12}")
    return diffs


def print_comparison_summary(all_diffs, path_a, path_b):
    """Print final summary across all tasks."""
    header_width = 95
    print("\n" + "=" * header_width)
    print("  OVERALL COMPARISON")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")
    print("=" * header_width)
    if all_diffs:
        avg_diff = sum(all_diffs) / len(all_diffs)
        sign = "+" if avg_diff > 0 else ""
        print(f"  Average diff (B-A) across {len(all_diffs)} metrics: {sign}{avg_diff:.4f}")
    print("=" * header_width)


def save_results(results, model_path, task_list, batch_size, num_fewshot, output_dir, suffix=""):
    """Save evaluation results to JSON."""
    fname = f"lm_eval_results{suffix}.json"
    results_file = os.path.join(output_dir, fname)
    serializable = {
        "results": results["results"],
        "config": {
            "model_path": model_path,
            "tasks": task_list,
            "batch_size": batch_size,
            "num_fewshot": num_fewshot,
        },
    }
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Results saved to {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate HF model with lm-eval-harness")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved HF model checkpoint directory (or HF model name)")
    parser.add_argument("--compare_path", type=str, default=None,
                        help="Second model to compare against (optional)")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS),
                        help="Comma-separated list of tasks to evaluate")
    parser.add_argument("--batch_size", type=str, default="auto",
                        help="Batch size for evaluation (default: auto)")
    parser.add_argument("--num_fewshot", type=int, default=None,
                        help="Number of few-shot examples (default: task-specific)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: <model_path>/eval_results)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Log results to wandb project")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.model_path, "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model A: {args.model_path}")
    if args.compare_path:
        print(f"Model B: {args.compare_path}")
    print(f"Tasks:   {task_list}")
    print(f"Output:  {output_dir}")

    if args.compare_path:
        # Compare mode: evaluate each task on both models, print comparison after each
        all_diffs = []
        merged_a = {"results": {}}
        merged_b = {"results": {}}

        for i, task in enumerate(task_list, 1):
            print(f"\n{'='*95}")
            print(f"  [{i}/{len(task_list)}] Evaluating task: {task}")
            print(f"{'='*95}")

            res_a = evaluate_single_task(args.model_path, task, args.batch_size, args.num_fewshot, args.device)
            res_b = evaluate_single_task(args.compare_path, task, args.batch_size, args.num_fewshot, args.device)

            merged_a["results"].update(res_a["results"])
            merged_b["results"].update(res_b["results"])

            diffs = print_task_comparison(res_a, res_b, args.model_path, args.compare_path, task, i, len(task_list))
            all_diffs.extend(diffs)

        print_comparison_summary(all_diffs, args.model_path, args.compare_path)

        # Save full results
        save_results(merged_a, args.model_path, task_list, args.batch_size, args.num_fewshot, output_dir, suffix="_A")
        save_results(merged_b, args.compare_path, task_list, args.batch_size, args.num_fewshot, output_dir, suffix="_B")

        # Save comparison
        comparison_file = os.path.join(output_dir, "lm_eval_comparison.json")
        metrics_a = extract_metrics(merged_a)
        metrics_b = extract_metrics(merged_b)
        comparison = {}
        for (task, metric) in sorted(set(metrics_a.keys()) | set(metrics_b.keys())):
            val_a = metrics_a.get((task, metric))
            val_b = metrics_b.get((task, metric))
            comparison[f"{task}/{metric}"] = {
                "model_a": val_a,
                "model_b": val_b,
                "diff": (val_b - val_a) if val_a is not None and val_b is not None else None,
            }
        with open(comparison_file, "w") as f:
            json.dump({"model_a": args.model_path, "model_b": args.compare_path, "metrics": comparison}, f, indent=2)
        print(f"Comparison saved to {comparison_file}")

        results_a = merged_a  # for wandb logging below
    else:
        # Single model: evaluate all tasks at once
        results_a = evaluate_model(args.model_path, task_list, args.batch_size, args.num_fewshot, args.device)
        print_single_results(results_a, args.model_path)
        save_results(results_a, args.model_path, task_list, args.batch_size, args.num_fewshot, output_dir)

    # Optionally log to wandb
    if args.wandb_project:
        try:
            import wandb
            run_name = args.wandb_run_name or os.path.basename(args.model_path.rstrip("/"))
            wandb.init(project=args.wandb_project, name=f"eval_{run_name}")
            flat_metrics = {}
            for task_name, task_results in results_a["results"].items():
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        flat_metrics[f"eval/{task_name}/{metric.replace(',none', '')}"] = value
            wandb.log(flat_metrics)
            wandb.finish()
            print(f"Results logged to wandb project: {args.wandb_project}")
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")


if __name__ == "__main__":
    main()
