## install
```bash
# Clone the repository with submodules
git submodule update --init --recursive

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install mteb from submodule (editable mode)
uv pip install -e ./mteb

uv pip install -e ./LEANN

# Install colpali-engine
uv pip install .
```

## eval
```
mteb run -b "ViDoRe(v2)" -m "vidore/colqwen2.5-v0.2"
```
## for train 

```
uv pip install flash-attn --no-build-isolation

USE_LOCAL_DATASET=0 python scripts/configs/qwen2/train_colqwen25_model.py \
    --output-dir ./models/my_colqwen25_lora \
    --peft \
    --batch-size 8 \
    --eval-batch-size 4 \
    --gradient-accumulation-steps 8
```

if we use multi GPU
```
MODEL_NAME="colqwen25-training-e2e-Dec7"
export WANDB_PROJECT="$MODEL_NAME"
USE_LOCAL_DATASET=0 accelerate launch --multi-gpu scripts/configs/qwen2/train_colqwen25_model.py \
    --output-dir ./models/$MODEL_NAME \
    --peft \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    --num-epochs 5
```

## Training with Document-Aligned Parameters (Recommended)

If accuracy is not sufficient, try aligning with the parameters from the paper/document:

```bash
MODEL_NAME="colqwen25-training-aligned"
export WANDB_PROJECT="$MODEL_NAME"
USE_LOCAL_DATASET=0 accelerate launch --multi-gpu scripts/configs/qwen2/train_colqwen25_model.py \
    --output-dir ./models/$MODEL_NAME \
    --peft \
    --batch-size 32 \
    --lr 5e-5 \
    --optimizer paged_adamw_8bit \
    --warmup-steps 12 \
    --num-epochs 1
```

```
MODEL_NAME="colqwen25-training-aligned-5epoch"
export WANDB_PROJECT="$MODEL_NAME"
USE_LOCAL_DATASET=0 accelerate launch --multi-gpu scripts/configs/qwen2/train_colqwen25_model.py \
    --output-dir ./models/$MODEL_NAME \
    --peft \
    --batch-size 32 \
    --lr 5e-5 \
    --optimizer paged_adamw_8bit \
    --warmup-steps 62 \
    --num-epochs 5
```

Parameter explanations:
- `--batch-size 32`: Batch size from the document
- `--lr 5e-5`: Learning rate from the document (4x lower than default 2e-4)
- `--optimizer paged_adamw_8bit`: Optimizer from the document (memory-efficient)
- `--warmup-steps 12`: Document requires 2.5% warmup (1 epoch ≈ 498 steps, 2.5% = 12 steps)
- `--num-epochs 1`: Train for 1 epoch as in the document

If training for multiple epochs, warmup steps should be increased accordingly:
- 5 epochs: `--warmup-steps 62` (2.5% of ~2489 steps)