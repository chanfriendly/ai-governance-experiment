#!/usr/bin/env python3
"""
Script to download models compatible with the Windows version of AI Governance Experiment.
"""

import os
import argparse
import logging
import requests
import yaml
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("download_windows_model")

# Define model options
MODELS = {
    "phi2": {
        "name": "Microsoft Phi-2 (small, 2.7B parameters)",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi-2.Q4_K_M.gguf",
        "size_mb": 1500,  # Approximate size in MB
        "chat_template": "<s>[INST] {{ system_message }}\n\n{{ user_message }} [/INST]"
    },
    "mistral": {
        "name": "Mistral 7B Instruct (medium, 7B parameters)",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_mb": 4500,  # Approximate size in MB
        "chat_template": "<s>[INST] {{ system_message }}\n\n{{ user_message }} [/INST]"
    },
    "gpt2": {
        "name": "GPT-2 (small, 124M parameters)",
        "url": "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin",
        "filename": "gpt2.bin",
        "size_mb": 500,   # Approximate size in MB
        "chat_template": "{{ system_message }}\n\nUser: {{ user_message }}\n\nAssistant:",
        "is_hf": True     # This is a HuggingFace model, not GGUF
    }
}

def download_file(url, filename, size_mb=None):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        filename: Local filename to save to
        size_mb: Approximate size in MB for progress bar estimation
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Download with progress bar
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0)) or (size_mb * 1024 * 1024 if size_mb else None)
        
        with open(filename, 'wb') as f, tqdm(
            desc=os.path.basename(filename),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
        logger.info(f"Downloaded {filename} successfully!")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def update_config(model_info, is_hf=False):
    """
    Update the config file to use the downloaded model.
    
    Args:
        model_info: Dictionary with model information
        is_hf: Whether this is a HuggingFace model (not GGUF)
    """
    try:
        config_path = os.path.join("configs", "agent_test_config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Update model path
        model_path = os.path.join("models", model_info["filename"])
        config['model']['model_name'] = model_path
        
        # Update chat template if available
        if "chat_template" in model_info:
            config['model']['chat_template'] = model_info["chat_template"]
            
        # Add HF flag if needed
        if is_hf:
            config['model']['is_hf'] = True
            
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"Updated {config_path} with model: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return False

def download_hf_model(model_name):
    """
    Download a model from HuggingFace directly using the transformers library.
    
    Args:
        model_name: Model name on HuggingFace
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Downloading {model_name} using transformers...")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Save them locally
        model_path = os.path.join("models", model_name.replace("/", "_"))
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        logger.info(f"Downloaded and saved {model_name} to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error downloading HuggingFace model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download models for Windows AI Governance Experiment")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=list(MODELS.keys()) + ["list"], 
        default="list",
        help="Model to download (or 'list' to see options)"
    )
    parser.add_argument(
        "--huggingface", 
        type=str, 
        help="Download a specific HuggingFace model by name (e.g., 'microsoft/phi-2')"
    )
    
    args = parser.parse_args()
    
    # List available models
    if args.model == "list":
        logger.info("Available models:")
        for key, info in MODELS.items():
            logger.info(f"  - {key}: {info['name']} ({info['size_mb']} MB)")
        return
    
    # Download a specific HuggingFace model
    if args.huggingface:
        model_path = download_hf_model(args.huggingface)
        if model_path:
            # Update config
            model_info = {
                "filename": model_path,
                "chat_template": "{{ system_message }}\n\nUser: {{ user_message }}\n\nAssistant:"
            }
            update_config(model_info, is_hf=True)
        return
    
    # Download a pre-defined model
    model_info = MODELS[args.model]
    logger.info(f"Downloading {model_info['name']}...")
    
    # Download the file
    filename = os.path.join("models", model_info["filename"])
    if model_info.get("is_hf", False):
        success = download_hf_model(args.model)
    else:
        success = download_file(model_info["url"], filename, model_info["size_mb"])
    
    # Update config if download was successful
    if success:
        update_config(model_info, is_hf=model_info.get("is_hf", False))
        logger.info("Download complete and config updated!")
        logger.info(f"Model saved to: {filename}")
        logger.info("You can now run the experiment with:")
        logger.info("python src/test_agent_oumi.py --config configs/agent_test_config.yaml --agent effective_altruism --scenario trolley")
    else:
        logger.error("Download failed. Please try again or choose a different model.")

if __name__ == "__main__":
    main()