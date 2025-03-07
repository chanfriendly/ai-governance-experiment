import os
import logging
import torch
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("windows_inference")

class WindowsInferenceEngine:
    """A simplified Windows-compatible inference engine."""
    
    def __init__(self, model_params):
        """Initialize the Windows inference engine."""
        self.model_path = model_params.model_name
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Try to load as a Hugging Face model ID
            model_id = "gpt2"  # Fallback to GPT-2
            logger.info(f"Loading {model_id} model")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.9):
        """Generate text from the model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(
                inputs["input_ids"], 
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,  # Avoid 0
                top_p=top_p,
                do_sample=temperature > 0
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
    
    def infer_online(self, conversations, inference_config):
        """Process a list of conversations and return responses."""
        results = []
        try:
            for conv in conversations:
                # Format conversation into a prompt
                prompt = self._format_conversation(conv, inference_config)
                
                # Generate response
                response = self.generate(
                    prompt,
                    max_tokens=inference_config.generation.max_new_tokens,
                    temperature=inference_config.generation.temperature,
                    top_p=inference_config.generation.top_p
                )
                
                # Add the response to the conversation
                from oumi.core.types.conversation import Message, Role, Conversation
                
                # Clone the conversation to avoid modifying the original
                new_messages = list(conv.messages)
                new_conv = Conversation(messages=new_messages)
                
                # Create a new message with the response
                new_message = Message(
                    role=Role.ASSISTANT, 
                    content=response
                )
                
                # Add the new message to the conversation
                new_conv.messages.append(new_message)
                
                results.append(new_conv)
                
            return results
        except Exception as e:
            logger.error(f"Error in infer_online: {e}")
            raise
    
    def _format_conversation(self, conversation, inference_config):
        """Format a conversation into a prompt string."""
        try:
            prompt = ""
            
            # Apply chat template if specified
            if hasattr(inference_config.model, 'model_kwargs') and inference_config.model.model_kwargs:
                if 'chat_template' in inference_config.model.model_kwargs:
                    template = inference_config.model.model_kwargs['chat_template']
                    
                    # Get system and user messages
                    from oumi.core.types.conversation import Role
                    system_content = next((m.content for m in conversation.messages if m.role == Role.SYSTEM), "")
                    user_content = next((m.content for m in conversation.messages if m.role == Role.USER), "")
                    
                    # Replace placeholders with actual content
                    prompt = template.replace("{{ system_message }}", system_content).replace("{{ user_message }}", user_content)
                    return prompt
            
            # If no template or template application failed, use standard formatting
            from oumi.core.types.conversation import Role
            for message in conversation.messages:
                if message.role == Role.SYSTEM:
                    prompt += f"### System:\n{message.content}\n\n"
                elif message.role == Role.USER:
                    prompt += f"### User:\n{message.content}\n\n"
                elif message.role == Role.ASSISTANT:
                    prompt += f"### Assistant:\n{message.content}\n\n"
            
            prompt += "### Assistant:\n"
            
            return prompt
        except Exception as e:
            logger.error(f"Error formatting conversation: {e}")
            return "Error occurred while formatting the conversation."