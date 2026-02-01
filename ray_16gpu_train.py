#!/usr/bin/env python3
"""
Launch 16-GPU distributed training using Ray to coordinate both nodes.
"""
import ray
import time

ray.init(address="auto")

print("Starting 16-GPU distributed training...")
print(f"Available resources: {ray.cluster_resources()}")

MASTER_ADDR = "172.31.2.101"
MASTER_PORT = "29500"

@ray.remote(num_gpus=8)
def run_training(node_rank: int, colpali_path: str):
    """Run training on a node."""
    import os
    import subprocess
    import sys

    os.chdir(colpali_path)

    # Activate venv
    python = os.path.join(colpali_path, ".venv/bin/python")
    torchrun = os.path.join(colpali_path, ".venv/bin/torchrun")

    # Set environment
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["USE_LOCAL_DATASET"] = "0"
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_SOCKET_IFNAME"] = "enp135s0"
    env["MASTER_ADDR"] = MASTER_ADDR
    env["MASTER_PORT"] = MASTER_PORT

    cmd = [
        torchrun,
        "--nnodes=2",
        "--nproc_per_node=8",
        f"--node_rank={node_rank}",
        f"--master_addr={MASTER_ADDR}",
        f"--master_port={MASTER_PORT}",
        "scripts/configs/qwen2/train_colqwen25_model.py",
        "--output-dir", "./models/colqwen25-16gpu-5epoch",
        "--peft",
        "--batch-size", "16",
        "--lr", "5e-5",
        "--optimizer", "paged_adamw_8bit",
        "--num-epochs", "5",
    ]

    print(f"Node {node_rank}: Starting training at {colpali_path}")
    print(f"Command: {' '.join(cmd)}")

    # Run training
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output
    for line in process.stdout:
        print(f"[Node {node_rank}] {line}", end="")

    process.wait()
    return {"node_rank": node_rank, "returncode": process.returncode}

# Launch on both nodes
# Master node (172.31.2.101): /home/andy/colpali
# Worker node (172.31.1.143): /home/andyl/colpali

print("\nLaunching training on both nodes...")

futures = [
    run_training.options(resources={"node:172.31.2.101": 0.01}).remote(0, "/home/andy/colpali"),
    run_training.options(resources={"node:172.31.1.143": 0.01}).remote(1, "/home/andyl/colpali"),
]

# Wait for completion
results = ray.get(futures)
print(f"\nTraining completed: {results}")
