import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch

def setup_model(model_name, quantize=True, bits=4):
    """
    Download and optionally quantize a model.
    
    Args:
        model_name: Name or path of the model to download
        quantize: Whether to quantize the model
        bits: Number of bits for quantization (4 or 8)
    
    Returns:
        Path to the quantized model or original model
    """
    print(f"Setting up model: {model_name}")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"models/{model_name.split('/')[-1]}_tokenizer")
    
    if not quantize:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model_path = f"models/{model_name.split('/')[-1]}"
        model.save_pretrained(model_path)
        return model_path
    
    # Quantize the model
    model_path = f"models/{model_name.split('/')[-1]}_{bits}bit"
    
    # Check if the model is already quantized
    if os.path.exists(model_path):
        print(f"Quantized model already exists at {model_path}")
        return model_path
    
    print(f"Quantizing model to {bits} bits...")
    
    # Different approach based on quantization method
    if bits == 4:
        # Use bitsandbytes for 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto"
        )
    else:
        # Use GPTQ for 8-bit or other quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto"
        )
        quantizer = GPTQQuantizer(
            bits=bits,
            dataset="c4",
            model_seqlen=2048,
        )
        model = quantizer.quantize_model(model, tokenizer)
    
    # Save the quantized model
    model.save_pretrained(model_path)
    print(f"Model quantized and saved to {model_path}")
    
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and quantize a model")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--no-quantize", action="store_true", help="Don't quantize the model")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits")
    
    args = parser.parse_args()
    setup_model(args.model, not args.no_quantize, args.bits)
