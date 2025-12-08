import argparse
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set

from scripts.configs.qwen2.wandb_utils import prepare_wandb_config, setup_wandb_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True, help="where to write model + script copy")
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--tau", type=float, default=0.02, help="temperature for loss function")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "pairwise"], help="loss function to use")
    p.add_argument("--peft", action="store_true", help="use PEFT for training")
    p.add_argument("--batch-size", type=int, default=64, help="per device train batch size (default: 64)")
    p.add_argument("--eval-batch-size", type=int, default=16, help="per device eval batch size (default: 16)")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1, help="gradient accumulation steps (default: 1)")
    p.add_argument("--num-epochs", type=int, default=5, help="number of training epochs (default: 5)")
    p.add_argument("--dataloader-num-workers", type=int, default=8, help="number of dataloader worker processes (default: 8, use 0 for debug)")
    p.add_argument("--warmup-steps", type=int, default=None, help="number of warmup steps (default: 2.5%% of total steps, calculated from dataset size, batch size, gradient accumulation, and epochs)")
    p.add_argument("--optimizer", type=str, default=None, help="optimizer to use (default: adamw_torch_fused, options: adamw_torch, adamw_torch_fused, paged_adamw_8bit)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.loss == "ce":
        loss_func = ColbertLoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
    elif args.loss == "pairwise":
        loss_func = ColbertPairwiseCELoss(
            normalize_scores=False,
        )
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    # Check if flash attention is available
    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
    if attn_implementation:
        print("✅ Using Flash Attention 2")
    else:
        print("⚠️  Flash Attention 2 not available, using default attention implementation")
        print("   Install flash-attn for better performance: pip install flash-attn")

    # Check if using local or remote dataset
    use_local_dataset = os.environ.get("USE_LOCAL_DATASET", "1") == "1"
    eval_dataset_path = "./data_dir/colpali_train_set" if use_local_dataset else "vidore/colpali_train_set"
    if not use_local_dataset:
        print("📥 Using remote dataset from HuggingFace Hub")

    # Load training dataset to calculate total steps for warmup calculation
    train_dataset = load_train_set()
    
    # Calculate warmup steps: 2.5% of total steps if not specified
    if args.warmup_steps is None:
        # Calculate total training steps
        # Note: This is approximate for multi-GPU, but HuggingFace Trainer will handle the actual calculation
        num_gpus = int(os.environ.get("WORLD_SIZE", 1))
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps * num_gpus
        dataset_size = len(train_dataset)
        steps_per_epoch = (dataset_size + effective_batch_size - 1) // effective_batch_size  # ceiling division
        total_steps = steps_per_epoch * args.num_epochs
        warmup_steps = max(1, int(total_steps * 0.025))  # 2.5% of total steps, at least 1
        print(f"📊 Calculated warmup steps: {warmup_steps} (2.5% of {total_steps} total steps)")
    else:
        warmup_steps = args.warmup_steps

    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=ColQwen2_5_Processor.from_pretrained(
            pretrained_model_name_or_path="./models/base_models/colqwen2.5-base",
            max_num_visual_tokens=768,
        ),
        model=ColQwen2_5.from_pretrained(
            pretrained_model_name_or_path="./models/base_models/colqwen2.5-base",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation=attn_implementation,
        ),
        train_dataset=train_dataset,
        eval_dataset=ColPaliEngineDataset(
            load_dataset(eval_dataset_path, split="test"), pos_target_column_name="image"
        ),
        run_eval=True,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=None,
            overwrite_output_dir=True,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=args.eval_batch_size,
            eval_strategy="steps",
            dataloader_num_workers=args.dataloader_num_workers,
            save_steps=500,
            logging_steps=10,
            eval_steps=100,
            warmup_steps=warmup_steps,
            learning_rate=args.lr,
            lr_scheduler_type="linear",  # Linear decay as per document requirements
            optim=args.optimizer if args.optimizer else "adamw_torch_fused",  # Default optimizer
            save_total_limit=1,
            report_to="wandb",  # Enable wandb logging
            run_name=f"colqwen25_lora_{args.output_dir.split('/')[-1]}",  # Optional: set run name
        ),
        peft_config=LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
        )
        if args.peft
        else None,
    )

    # make sure output_dir exists and copy script for provenance
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    # Prepare LoRA config dict if using PEFT
    lora_config = None
    if args.peft and config.peft_config:
        lora_config = {
            "r": config.peft_config.r,
            "alpha": config.peft_config.lora_alpha,
            "dropout": config.peft_config.lora_dropout,
            "init_weights": config.peft_config.init_lora_weights,
            "task_type": config.peft_config.task_type,
            "bias": config.peft_config.bias,
            "target_modules": config.peft_config.target_modules,
        }

    # Setup wandb logging with custom hyperparameters
    custom_config = prepare_wandb_config(
        args=args,
        warmup_steps=warmup_steps,
        eval_dataset_path=eval_dataset_path,
        use_local_dataset=use_local_dataset,
        attn_implementation=attn_implementation,
        model_pretrained_path="./models/base_models/colqwen2.5-base",
        max_num_visual_tokens=768,
        lora_config=lora_config,
    )
    setup_wandb_logging(config, custom_config)

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()
