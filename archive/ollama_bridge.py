# ollama_bridge.py

import requests
import logging

# Set up logging to help with debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaModel:
    """A class that mimics the Oumi model interface but connects to Ollama."""
    
    def __init__(self, model_name, temperature=0.7, max_tokens=512):
        """Initialize the Ollama model connector.
        
        Args:
            model_name: Name of the model in Ollama (e.g., "deepseek-r1:8b")
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # The base URL for Ollama API
        self.api_url = "http://localhost:11434/api/generate"
        
        logger.info(f"Initialized Ollama connector for model: {model_name}")
        
        # Verify that we can connect to Ollama
        self._check_connection()
    
    def _check_connection(self):
        """Check if we can connect to the Ollama API."""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models = [model["name"] for model in response.json()["models"]]
            
            if self.model_name not in models:
                logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {models}")
            else:
                logger.info(f"Successfully connected to Ollama. Model {self.model_name} is available.")
                
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.error("Make sure Ollama is running and accessible at localhost:11434")
            raise ConnectionError(f"Could not connect to Ollama: {e}")
    
    def infer(self, prompt):
        """Generate a response from the model.
        
        Args:
            prompt: The input prompt for the model
            
        Returns:
            The generated text response
        """
        logger.info(f"Sending prompt to Ollama model {self.model_name}")
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.text}")
                raise Exception(f"Ollama API returned status code {response.status_code}")
            
            result = response.json()["response"]
            logger.info("Successfully received response from Ollama")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise Exception(f"Failed to generate response: {e}")

def get_ollama_model(model_name, temperature=0.7, max_tokens=512):
    """Create an OllamaModel instance.
    
    This function mimics the interface of oumi.models.get_model.
    
    Args:
        model_name: Name of the model in Ollama
        temperature: Controls randomness in generation
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        An OllamaModel instance
    """
    return OllamaModel(model_name, temperature, max_tokens)