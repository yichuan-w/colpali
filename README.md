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
USE_LOCAL_DATASET=0 accelerate launch --multi-gpu scripts/configs/qwen2/train_colqwen25_model.py \
    --output-dir ./models/my_colqwen25_lora \
    --peft \
    --batch-size 8 \
    --gradient-accumulation-steps 8
```