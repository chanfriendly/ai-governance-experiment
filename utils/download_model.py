#!/usr/bin/env python3
"""
Script to download GGUF models from Hugging Face for the AI governance experiment.
This script uses the huggingface_hub library which handles authentication and file paths correctly.
"""

import os
import argparse
from huggingface_hub import hf_hub_download
import shutil

# Dictionary of recommended models with their repo_ids and filenames
RECOMMENDED_MODELS = {
    "deepseek": {
        "repo_id": "TheBloke/deepseek-llm-7b-chat-GGUF",
        "filename": "deepseek-llm-7b-chat.q4_k_m.gguf" 
    },
    "mistral": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.q4_k_m.gguf"
    },
    "tiny": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.q4_k_m.gguf"
    },
    "phi2": {
        "repo_id": "TheBloke/phi-2-GGUF",
        "filename": "phi-2.q4_k_m.gguf"
    }
}

def download_gguf_model(model_key, output_dir="models", force_download=False):
    """
    Download a GGUF model using the Hugging Face Hub library
    
    Args:
        model_key: Key from RECOMMENDED_MODELS or a custom repo_id
        output_dir: Directory to save the model
        force_download: Whether to redownload if the model already exists
    
    Returns:
        Path to the downloaded model
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if model_key in RECOMMENDED_MODELS:
        # Use a recommended model
        repo_id = RECOMMENDED_MODELS[model_key]["repo_id"]
        filename = RECOMMENDED_MODELS[model_key]["filename"]
        print(f"Using recommended model: {repo_id}/{filename}")
    else:
        # Assume model_key is a repo_id
        repo_id = model_key
        filename = None  # Will search for a default GGUF file
        print(f"Searching for GGUF files in repository: {repo_id}")
    
    # Check if model already exists in output_dir
    local_path = os.path.join(output_dir, filename) if filename else None
    if not force_download and local_path and os.path.exists(local_path):
        print(f"Model already exists at {local_path}")
        return local_path
    
    try:
        # Download the model
        print(f"Downloading model from {repo_id}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        # If path is not in the direct output directory, copy it there
        if os.path.dirname(path) != os.path.abspath(output_dir):
            new_path = os.path.join(output_dir, os.path.basename(path))
            shutil.copy2(path, new_path)
            path = new_path
            
        print(f"Model downloaded successfully to: {path}")
        return path
    
    except Exception as e:
        print(f"Error downloading model: {e}")
        
        # If no specific filename was provided, try to list available files
        if filename is None:
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id)
                gguf_files = [f for f in files if f.endswith('.gguf')]
                
                if gguf_files:
                    print("\nAvailable GGUF files in this repository:")
                    for f in gguf_files:
                        print(f"  {f}")
                    print("\nTry again with a specific filename.")
            except Exception:
                pass
        
        return None

def list_recommended_models():
    """Print the list of recommended models"""
    print("Recommended models:")
    for key, info in RECOMMENDED_MODELS.items():
        print(f"  {key}: {info['repo_id']}/{info['filename']}")

def main():
    parser = argparse.ArgumentParser(description="Download GGUF models from Hugging Face")
    parser.add_argument("--model", type=str, help=f"Model key or Hugging Face repo_id")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force redownload if model exists")
    parser.add_argument("--list", action="store_true", help="List recommended models")
    
    args = parser.parse_args()
    
    if args.list:
        list_recommended_models()
        return
    
    if not args.model:
        parser.error("Please specify a model key or repository ID with --model")
    
    model_path = download_gguf_model(args.model, args.output_dir, args.force)
    
    if model_path:
        print("\nTo use this model, update your config.yaml file with:")
        print(f"model:\n  model_name: \"{model_path}\"")
        
        # Provide chat template info based on the model
        if "mistral" in model_path.lower():
            print('  chat_template: "<s>[INST] {{ system_message }}\\n\\n{{ user_message }} [/INST]"')
        elif "deepseek" in model_path.lower():
            print('  chat_template: "### System:\\n{{ system_message }}\\n\\n### User:\\n{{ user_message }}\\n\\n### Assistant:\\n"')
        elif "tiny" in model_path.lower() or "llama" in model_path.lower():
            print('  chat_template: "<s>[INST] {{ system_message }}\\n\\n{{ user_message }} [/INST]"')
        elif "phi" in model_path.lower():
            print('  chat_template: "{{ system_message }}\\n\\nHuman: {{ user_message }}\\n\\nAssistant:"')

if __name__ == "__main__":
    main()