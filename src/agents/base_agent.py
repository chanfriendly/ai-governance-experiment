from typing import Dict, List, Any, Optional
import os
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Agent:
    def __init__(
        self,
        model_path: str,
        system_prompt: str,
        name: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512
    ):
        """
        Initialize an agent with a specialized system prompt.
        
        Args:
            model_path: Path to the model
            system_prompt: System prompt that defines the agent's specialization
            name: Name of the agent
            temperature: Sampling temperature for generation
            max_new_tokens: Maximum number of tokens to generate
        """
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, "tokenizer") 
            if os.path.exists(os.path.join(model_path, "tokenizer")) 
            else model_path
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        logging.info(f"Agent {name} initialized with model {model_path}")
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response based on the conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
        
        Returns:
            Generated response text
        """
        # Format the messages according to the model's expected format
        formatted_prompt = self._format_messages(messages)
        
        # Tokenize the prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate a response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for the model.
        This will need to be adjusted based on the specific model's expected format.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            Formatted prompt string
        """
        # Add the system prompt as the first message if not present
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        # Basic formatting - adjust based on model requirements
        formatted = ""
        
        for msg in messages:
            if msg["role"] == "system":
                formatted += f"<|system|>\n{msg['content']}\n"
            elif msg["role"] == "user":
                formatted += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"<|assistant|>\n{msg['content']}\n"
        
        # Add the final assistant prompt
        if messages[-1]["role"] != "assistant":
            formatted += "<|assistant|>\n"
        
        return formatted

    def save_config(self, path: str) -> None:
        """Save agent configuration."""
        config = {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens
        }
        
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{self.name}_config.json"), "w") as f:
            json.dump(config, f, indent=2)
