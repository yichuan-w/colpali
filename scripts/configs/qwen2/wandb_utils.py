"""Utilities for logging training hyperparameters to wandb."""

import wandb
from transformers import TrainerCallback


def create_wandb_config_callback(custom_config: dict) -> TrainerCallback:
    """
    Create a TrainerCallback to log custom hyperparameters to wandb.
    
    HuggingFace Trainer only logs TrainingArguments automatically, so we need
    to manually log custom hyperparameters using a callback.
    
    Args:
        custom_config: Dictionary of custom hyperparameters to log
        
    Returns:
        TrainerCallback instance that will log config on train begin
    """
    class CustomWandbConfigCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            # Update wandb config after Trainer has initialized wandb
            if wandb.run is not None:
                wandb.config.update(custom_config)
    
    return CustomWandbConfigCallback()


def prepare_wandb_config(
    args,
    warmup_steps: int,
    eval_dataset_path: str,
    use_local_dataset: bool,
    attn_implementation: str = None,
    model_pretrained_path: str = "./models/base_models/colqwen2.5-base",
    max_num_visual_tokens: int = 768,
    lora_config: dict = None,
) -> dict:
    """
    Prepare custom hyperparameters dictionary for wandb logging.
    
    Args:
        args: Parsed command line arguments
        warmup_steps: Calculated warmup steps
        eval_dataset_path: Path to evaluation dataset
        use_local_dataset: Whether using local dataset
        attn_implementation: Attention implementation (flash_attention_2 or None)
        model_pretrained_path: Path to pretrained model
        max_num_visual_tokens: Maximum number of visual tokens
        lora_config: Optional LoRA configuration dict with keys: r, alpha, dropout, 
                    init_weights, task_type, bias, target_modules
        
    Returns:
        Dictionary of hyperparameters ready for wandb.config.update()
    """
    custom_config = {
        # Loss and training config
        "loss_type": args.loss,
        "tau": args.tau,
        "use_peft": args.peft,
        "trainer": args.trainer,
        "warmup_steps": warmup_steps,
        
        # Dataset config
        "dataset_path": eval_dataset_path,
        "use_local_dataset": use_local_dataset,
        
        # Model config
        "model_pretrained_path": model_pretrained_path,
        "torch_dtype": "bfloat16",
        "use_cache": False,
        "attn_implementation": attn_implementation or "default",
        
        # Processor config
        "max_num_visual_tokens": max_num_visual_tokens,
        
        # Output config
        "output_dir": args.output_dir,
    }
    
    # Add LoRA config if using PEFT
    if args.peft:
        if lora_config:
            # Use provided LoRA config
            custom_config.update({
                "lora_r": lora_config.get("r", 32),
                "lora_alpha": lora_config.get("alpha", 32),
                "lora_dropout": lora_config.get("dropout", 0.1),
                "lora_init_weights": lora_config.get("init_weights", "gaussian"),
                "lora_task_type": lora_config.get("task_type", "FEATURE_EXTRACTION"),
                "lora_bias": lora_config.get("bias", "none"),
                "lora_target_modules": lora_config.get("target_modules", ""),
            })
        else:
            # Use default LoRA config
            custom_config.update({
                "lora_r": 32,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_init_weights": "gaussian",
                "lora_task_type": "FEATURE_EXTRACTION",
                "lora_bias": "none",
                "lora_target_modules": "(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
            })
    
    return custom_config


def setup_wandb_logging(config, custom_config: dict):
    """
    Setup wandb logging with custom hyperparameters.
    
    Adds a callback to TrainingArguments to log custom config after wandb is initialized.
    
    Args:
        config: ColModelTrainingConfig instance
        custom_config: Dictionary of custom hyperparameters to log
    """
    if config.tr_args.report_to == "wandb":
        # Create callback to log custom config
        callback = create_wandb_config_callback(custom_config)
        
        # Add callback to TrainingArguments
        if not hasattr(config.tr_args, "callbacks") or config.tr_args.callbacks is None:
            config.tr_args.callbacks = []
        config.tr_args.callbacks.append(callback)


