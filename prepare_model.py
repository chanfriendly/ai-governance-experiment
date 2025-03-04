#!/usr/bin/env python3
"""
Script to download, quantize, and prepare models for the AI governance experiment.
This script will:
1. Download a model from Hugging Face
2. Optionally quantize it using Unsloth
3. Convert it to GGUF format for use with llama.cpp
"""

import os
import argparse
import subprocess
import sys
from huggingface_hub import snapshot_download

# List of recommended models for the governance experiment
RECOMMENDED_MODELS = {
    "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek-instruct": "deepseek-ai/deepseek-llm-7b-chat",
    "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

def download_model(model_name, output_dir="models"):
    """Download a model from Hugging Face"""
    print(f"Downloading model: {model_name}")
    
    # Check if model_name is one of our recommended shortcuts
    if model_name in RECOMMENDED_MODELS:
        model_name = RECOMMENDED_MODELS[model_name]
        print(f"Using recommended model: {model_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the model
    model_path = snapshot_download(
        repo_id=model_name,
        local_dir=os.path.join(output_dir, model_name.split("/")[-1]),
        ignore_patterns=["*.bin", "*.pt", "*.safetensors"] if args.gguf_only else None
    )
    
    print(f"Model downloaded to: {model_path}")
    return model_path

def download_gguf(model_name, output_dir="models", quantization="Q4_K_M"):
    """Download a pre-converted GGUF model"""
    # Check if model_name is one of our recommended shortcuts
    if model_name in RECOMMENDED_MODELS:
        base_model = RECOMMENDED_MODELS[model_name]
        model_name = base_model.split("/")[-1]
    
    # Construct TheBloke's version name (common naming pattern)
    bloke_name = f"TheBloke/{model_name}-GGUF"
    
    # Try to find the model
    try:
        print(f"Looking for pre-converted GGUF from: {bloke_name}")
        
        # Download the GGUF file
        gguf_path = os.path.join(output_dir, f"{model_name}.{quantization}.gguf")
        
        # For Mistral which has a known good GGUF
        if "Mistral-7B-Instruct-v0.2" in model_name:
            url = f"https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.{quantization.lower()}.gguf"
            print(f"Downloading from: {url}")
            os.makedirs(output_dir, exist_ok=True)
            subprocess.run(["wget", url, "-O", gguf_path], check=True)
            return gguf_path
        
        # For TinyLlama
        if "TinyLlama-1.1B-Chat" in model_name:
            url = f"https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.{quantization.lower()}.gguf"
            print(f"Downloading from: {url}")
            os.makedirs(output_dir, exist_ok=True)
            subprocess.run(["wget", url, "-O", gguf_path], check=True)
            return gguf_path
        
        # Fall back to conversion
        print(f"Could not find pre-converted GGUF model. Will download and convert.")
        return None
        
    except Exception as e:
        print(f"Error downloading GGUF: {e}")
        print("Will download and convert the model instead.")
        return None

def convert_to_gguf(model_path, output_path, q_type="Q4_K_M"):
    """Convert a Hugging Face model to GGUF format"""
    # Ensure llama.cpp is available
    if not os.path.exists("llama.cpp"):
        print("Cloning llama.cpp repository...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run the conversion script
    convert_script = os.path.join("llama.cpp", "convert.py")
    print(f"Converting {model_path} to GGUF format...")
    
    try:
        subprocess.run([
            sys.executable, convert_script,
            model_path,
            "--outfile", output_path
        ], check=True)
        
        print(f"Model converted to GGUF and saved as {output_path}")
        
        # Quantize the GGUF model
        if q_type:
            quantize_script = os.path.join("llama.cpp", "quantize")
            
            # Check if quantize executable exists, build if needed
            if not os.path.exists(quantize_script):
                print("Building llama.cpp...")
                subprocess.run(["make"], cwd="llama.cpp", check=True)
            
            q_output = f"{os.path.splitext(output_path)[0]}.{q_type}.gguf"
            print(f"Quantizing to {q_type}...")
            
            subprocess.run([
                quantize_script, 
                output_path, 
                q_output, 
                q_type
            ], check=True)
            
            print(f"Model quantized and saved as {q_output}")
            return q_output
        
        return output_path
    except Exception as e:
        print(f"Error converting model: {e}")
        return None

def main(args):
    """Main function to prepare a model"""
    # First check if a pre-converted GGUF is available
    if args.gguf_only:
        gguf_path = download_gguf(args.model, args.output_dir, args.quantization)
        if gguf_path:
            print(f"Successfully downloaded pre-converted GGUF model: {gguf_path}")
            print(f"\nTo use this model, update your config.yaml file with:")
            print(f"model:\n  model_name: \"{gguf_path}\"")
            return gguf_path
    
    # If not, download and convert
    model_path = download_model(args.model, args.output_dir)
    
    # Convert to GGUF
    model_basename = os.path.basename(model_path)
    gguf_path = os.path.join(args.output_dir, f"{model_basename}.gguf")
    
    gguf_path = convert_to_gguf(model_path, gguf_path, args.quantization)
    
    if gguf_path:
        print(f"\nSuccessfully prepared model: {gguf_path}")
        print(f"\nTo use this model, update your config.yaml file with:")
        print(f"model:\n  model_name: \"{gguf_path}\"")
        return gguf_path
    else:
        print("Failed to prepare model.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a model for the AI governance experiment")
    parser.add_argument("--model", type=str, required=True, 
                        help=f"Model name or shortcut ({', '.join(RECOMMENDED_MODELS.keys())})")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--quantization", type=str, default="Q4_K_M", 
                        choices=["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
                        help="Quantization type")
    parser.add_argument("--gguf-only", action="store_true", 
                        help="Only download pre-converted GGUF if available (faster)")
    
    args = parser.parse_args()
    main(args)