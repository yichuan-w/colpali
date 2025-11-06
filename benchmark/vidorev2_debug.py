#!/usr/bin/env python3
"""
Debug script for running ViDoRe(v2) benchmark with colqwen2.5-v0.2
You can run this with a debugger or directly with: python benchmark/vidorev2_debug.py
"""

import mteb
import os

# Optional: Set specific GPU (uncomment if needed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    results = mteb.evaluate(
        model=model,
        tasks=benchmark,
        # Optional: Add output folder
        # output_folder="my_results"
    )

    print("\nEvaluation complete!")
    print(f"Results: {results}")

    return results


if __name__ == "__main__":
    # You can set breakpoints anywhere in main() for debugging
    results = main()
