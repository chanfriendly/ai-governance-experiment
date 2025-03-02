# patch_oumi.py

import oumi  # Import the original oumi
import logging
from ollama_bridge import get_ollama_model  # Import our bridge

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Save the original get_model function
original_get_model = oumi.models.get_model

# Create a list of model names that should be handled by Ollama
OLLAMA_MODELS = [
    "deepseek-r1:8b",
    "deepseek-r1:1.5b",
    "granite3.2:8b",
    "llama3.2:latest"
]

def patched_get_model(model_name, **kwargs):
    """Patched version of get_model that routes certain models to Ollama.
    
    This function checks if the requested model is in our list of Ollama models.
    If it is, we use our bridge to connect to Ollama. If not, we use the
    original Oumi function.
    
    Args:
        model_name: Name of the model to get
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        A model instance that provides the .infer() method
    """
    # Check if it's an Ollama model
    if model_name.startswith("ollama:"):
        # Remove the "ollama:" prefix
        ollama_model_name = model_name[7:]
        logger.info(f"Routing model request to Ollama: {ollama_model_name}")
        return get_ollama_model(ollama_model_name, **kwargs)
    
    # Check if it's in our list of known Ollama models
    elif model_name in OLLAMA_MODELS:
        logger.info(f"Recognized {model_name} as an Ollama model")
        return get_ollama_model(model_name, **kwargs)
    
    # Otherwise, use the original function
    else:
        logger.info(f"Using original Oumi get_model for {model_name}")
        return original_get_model(model_name, **kwargs)

# Replace the original function with our patched version
oumi.models.get_model = patched_get_model

logger.info("Successfully patched Oumi to work with Ollama models")