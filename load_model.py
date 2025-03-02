from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes
print(bitsandbytes.__version__)

# Define the model name
model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the quantization configuration
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Load the model with the updated config
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

print("Model and tokenizer loaded successfully.")