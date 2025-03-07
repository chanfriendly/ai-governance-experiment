# New file: utils/check_model_compatibility.py
import os
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_model_compatibility():
    """Check what models can run on this system"""
    print(f"Platform: {platform.system()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    small_models = ["gpt2", "distilgpt2", "microsoft/phi-2"]
    
    for model_id in small_models:
        try:
            print(f"Trying to load {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            print(f"✓ Successfully loaded {model_id}")
            
            # Test generation
            input_text = "Hello, I am a"
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=10)
            result = tokenizer.decode(outputs[0])
            print(f"  Sample output: {result}")
        except Exception as e:
            print(f"✗ Failed to load {model_id}: {e}")
    
if __name__ == "__main__":
    check_model_compatibility()