#!/usr/bin/env python3
"""
Debug/eval script for running ViDoRe(v2) benchmark.
Examples:
  - List tasks:                     python benchmark/vidorev2_debug.py --list
  - Evaluate single task w/ models: python benchmark/vidorev2_debug.py \
      --task-name Vidore2ESGReportsHLRetrieval \
      --models vidore/colqwen2.5-v0.2,vidore/colpaligemma-Qwen2-V0.1 \
      --output ./vidore_results
      === ViDoRe(v2) tasks & datasets ===
- Vidore2ESGReportsRetrieval
  dataset.path: vidore/esg_reports_v2
  dataset.revision: 0542c0d03da0ec1c8cbc517c8d78e7e95c75d3d3
  hf_subsets: ['french', 'spanish', 'english', 'german']
  eval_splits: ['test']
- Vidore2EconomicsReportsRetrieval
  dataset.path: vidore/economics_reports_v2
  dataset.revision: b3e3a04b07fbbaffe79be49dabf92f691fbca252
  hf_subsets: ['french', 'spanish', 'english', 'german']
  eval_splits: ['test']
- Vidore2BioMedicalLecturesRetrieval
  dataset.path: vidore/biomedical_lectures_v2
  dataset.revision: a29202f0da409034d651614d87cd8938d254e2ea
  hf_subsets: ['french', 'spanish', 'english', 'german']
  eval_splits: ['test']
- Vidore2ESGReportsHLRetrieval
  dataset.path: vidore/esg_reports_human_labeled_v2
  dataset.revision: 6d467dedb09a75144ede1421747e47cf036857dd
  hf_subsets: ['default']
  eval_splits: ['test']
"""

import mteb
import os
import argparse
import json

# Optional: Set specific GPU (uncomment if needed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def sanitize_model_id(model_id: str) -> str:
    return model_id.replace('/', '__').replace(':', '_')


def list_vidore_tasks():
    print("Loading ViDoRe(v2) benchmark (for listing tasks/subsets)...")
    benchmark = mteb.get_benchmark("ViDoRe(v2)")
    print("\n=== ViDoRe(v2) tasks & datasets ===")
    for task in benchmark:
        md = task.metadata
        ds = md.dataset
        print(f"- {md.name}")
        print(f"  dataset.path: {ds.get('path')}")
        print(f"  dataset.revision: {ds.get('revision')}")
        print(f"  hf_subsets: {md.hf_subsets}")
        print(f"  eval_splits: {md.eval_splits}")
    print("=== End of list ===\n")


def main(args: argparse.Namespace):
    if args.list:
        list_vidore_tasks()
        return {}

    # Load the ViDoRe(v2) benchmark (for evaluation)
    print("Loading ViDoRe(v2) benchmark for evaluation...")
    benchmark = mteb.get_benchmark("ViDoRe(v2)")

    # Filter to the requested single task (by name or dataset.path)
    target_name = args.task_name
    target_path = args.dataset_path
    selected_tasks = []
    for task in benchmark:
        md = task.metadata
        ds = md.dataset
        if (target_name and md.name == target_name) or (target_path and ds.get('path') == target_path):
            selected_tasks.append(task)

    if not selected_tasks:
        print("No matching task found. Use --list to see available tasks.")
        return {}

    md0 = selected_tasks[0].metadata
    print("\nSelected task:")
    print(f"- {md0.name}")
    print(f"  dataset.path: {md0.dataset.get('path')}")
    print(f"  hf_subsets: {md0.hf_subsets}")
    print(f"  eval_splits: {md0.eval_splits}")

    # Prepare output root
    output_root = args.output if args.output else "vidore_results"
    os.makedirs(output_root, exist_ok=True)

    # Prepare models
    model_ids = args.models
    if not model_ids:
        model_ids = ["vidore/colqwen2.5-v0.2"]

    all_results = {}
    for model_id in model_ids:
        print(f"\nLoading model: {model_id}")
        model = mteb.get_model(model_id)
        print(f"Model loaded: {model}")

        out_dir = os.path.join(output_root, sanitize_model_id(model_id))
        os.makedirs(out_dir, exist_ok=True)

        print("Starting evaluation...")
        results = mteb.evaluate(
            model=model,
            tasks=selected_tasks,
            prediction_folder=out_dir,
        )
        print("Evaluation complete!")
        print(f"Results for {model_id}: {results}")

        # Save a compact JSON summary too
        try:
            summary = results.model_dump() if hasattr(results, "model_dump") else (
                results.dict() if hasattr(results, "dict") else None
            )
            if summary is None:
                # Fallback: stringify
                summary = {"repr": repr(results)}
            with open(os.path.join(out_dir, "results_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: could not save results_summary.json: {e}")

        all_results[model_id] = results

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViDoRe(v2) evaluation on a single task with one or more models.")
    parser.add_argument("--list", action="store_true", help="List all ViDoRe(v2) tasks and exit.")
    parser.add_argument("--task-name", type=str, default="Vidore2ESGReportsHLRetrieval", help="Task name to run (exact match).")
    parser.add_argument("--dataset-path", type=str, default="vidore/esg_reports_human_labeled_v2", help="Alternative selector: HF dataset path to match.")
    parser.add_argument("--models", type=lambda s: [x.strip() for x in s.split(',') if x.strip()], default=None, help="Comma-separated list of model IDs (e.g. 'vidore/colqwen2.5-v0.2,another/model').")
    parser.add_argument("--output", type=str, default="vidore_results", help="Output directory root where per-model results will be written.")

    cli_args = parser.parse_args()
    results = main(cli_args)
