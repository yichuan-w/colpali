#!/usr/bin/env python3
"""
Debug script for running ViDoRe(v2) benchmark with colqwen2.5-v0.2
You can run this with a debugger or directly with: python benchmark/vidorev2_debug.py
"""

import mteb
import os
import json

# Optional: Set specific GPU (uncomment if needed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def sanitize_model_id(model_id: str) -> str:
    return model_id.replace('/', '__').replace(':', '_')

def main():
    # Load the ViDoRe(v2) benchmark
    print("Loading ViDoRe(v2) benchmark...")
    benchmark = mteb.get_benchmark("ViDoRe(v2)")
    print(f"Benchmark loaded: {benchmark}")

    # Load the model
    print("\nLoading model: vidore/colqwen2.5-v0.2")
    model = mteb.get_model("vidore/colqwen2.5-v0.2")
    print(f"Model loaded: {model}")

    # Run evaluation - you can set a breakpoint here or inside evaluate
    print("\nStarting evaluation...")
    # Prepare output dir (predictions + summary) similar to vidorev2_debug.py
    output_root = "vidore_results-new"
    os.makedirs(output_root, exist_ok=True)
    out_dir = os.path.join(output_root, sanitize_model_id("vidore/colqwen2.5-v0.2"))
    os.makedirs(out_dir, exist_ok=True)
    results = mteb.evaluate(
        model=model,
        tasks=benchmark,
        prediction_folder=out_dir,
    )
    print("\nEvaluation complete!")
    print(f"Results: {results}")
    # Save a compact JSON summary
    try:
        summary = results.model_dump() if hasattr(results, "model_dump") else (
            results.dict() if hasattr(results, "dict") else None
        )
        if summary is None:
            summary = {"repr": repr(results)}
        summary_path = os.path.join(out_dir, "results_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved results summary to: {summary_path}")
    except Exception as e:
        print(f"Warning: could not save results_summary.json: {e}")
    return results

if __name__ == "__main__":
    main()