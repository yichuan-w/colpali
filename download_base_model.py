#!/usr/bin/env python3
"""Download base model for training"""

import os
from pathlib import Path
from transformers import AutoModel, AutoProcessor

def download_base_model():
    """Download vidore/colqwen2.5-base model to local path
    
    This is the untrained base version specifically designed for ColQwen2.5
    to guarantee deterministic projection layer initialization.
    See: https://huggingface.co/vidore/colqwen2.5-base
    """
    
    # Model info - MUST use vidore/colqwen2.5-base, not Qwen/Qwen2.5-VL-3B-Instruct
    hf_model_id = "vidore/colqwen2.5-base"
    local_path = "./models/base_models/colqwen2.5-base"
    
    print(f"📥 Downloading ColQwen2.5 base model: {hf_model_id}")
    print(f"📁 Target path: {local_path}")
    print("ℹ️  This is the untrained base version for deterministic LoRA initialization")
    
    # Create directory
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    try:
        print("\n🔄 Downloading model and processor...")
        # Download both model and processor together
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        
        print("  - Downloading model...")
        model = ColQwen2_5.from_pretrained(
            hf_model_id,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.save_pretrained(local_path)
        print("  ✅ Model downloaded successfully!")
        
        print("  - Downloading processor...")
        processor = ColQwen2_5_Processor.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
        )
        processor.save_pretrained(local_path)
        print("  ✅ Processor downloaded successfully!")
        
        print(f"\n🎉 Base model ready at: {local_path}")
        print("\nYou can now run training:")
        print("python scripts/configs/qwen2/train_colqwen25_model.py \\")
        print("    --output-dir ./models/my_colqwen25_lora \\")
        print("    --peft")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Internet connection")
        print("   2. HuggingFace access (login if needed: huggingface-cli login)")
        print("   3. Sufficient disk space (~6GB)")
        raise

if __name__ == "__main__":
    download_base_model()

