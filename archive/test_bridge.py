# test_bridge.py

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# First, patch Oumi
logger.info("Patching Oumi to work with Ollama...")
import patch_oumi

# Now import Oumi (it's already patched)
import oumi

def test_model(model_name):
    """Test a model with a simple prompt."""
    logger.info(f"Testing model: {model_name}")
    
    try:
        # Get the model using the patched function
        model = oumi.models.get_model(model_name, temperature=0.7, max_tokens=100)
        
        # Test with a simple prompt
        prompt = "Hello, can you hear me? Please give a short response."
        logger.info(f"Sending prompt: {prompt}")
        
        response = model.infer(prompt)
        
        logger.info(f"Response: {response}")
        print(f"\nTest results for {model_name}:")
        print("-" * 50)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 50)
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\nTest failed for {model_name}: {e}")
        return False

if __name__ == "__main__":
    # Test with your Ollama model
    test_model("deepseek-r1:8b")