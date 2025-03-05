from typing import Dict, List, Any, Optional
import os
import json
import logging
import re

class Agent:
    def __init__(
        self,
        model_engine,  # This will be your LLAMACPP engine
        system_prompt: str,
        name: str,
        temperature: float = 0.3,
        max_new_tokens: int = 1536,
        reasoning_window: int = 300  # Control the thinking/reasoning process
    ):
        """
        Initialize an agent with a specialized system prompt.
        
        Args:
            model_engine: The inference engine to use for generation
            system_prompt: System prompt that defines the agent's specialization
            name: Name of the agent
            temperature: Sampling temperature for generation
            max_new_tokens: Maximum number of tokens to generate
            reasoning_window: Approximate token limit for reasoning section
        """
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model_engine = model_engine
        self.reasoning_window = reasoning_window
        
        logging.info(f"Agent {name} initialized")
    
    def respond(self, scenario: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, str]:
        """
        Generate a response to the given scenario.
        
        Args:
            scenario: Text describing the scenario to respond to
            conversation_history: Optional list of previous messages
            
        Returns:
            Dictionary containing reasoning, decision, and full response
        """
        # Format the input with system prompt, conversation history, and scenario
        formatted_input = self._format_input(scenario, conversation_history)
        
        # Generate response using the model engine
        response = self.model_engine.generate(
            formatted_input,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        
        # Extract reasoning and decision from response
        reasoning, decision = self._parse_response(response)
        
        return {
            "reasoning": reasoning,
            "decision": decision,
            "full_response": response
        }
    
    def _format_input(self, scenario: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Format the input for the model engine.
        
        Args:
            scenario: The scenario text
            conversation_history: Optional previous conversation
            
        Returns:
            Formatted input string
        """
        # Start with the system prompt
        formatted = f"<|system|>\n{self.system_prompt}\n\n"
        
        # Add conversation history if provided
        if conversation_history:
            for message in conversation_history:
                role = message.get("role", "user")
                content = message.get("content", "")
                formatted += f"<|{role}|>\n{content}\n\n"
        
        # Add the current scenario as a user message
        formatted += f"<|user|>\n{scenario}\n\n<|assistant|>\n"
        
        return formatted
    
    def _parse_response(self, response: str) -> tuple[str, str]:
        """
        Parse the model response to extract reasoning and decision.
        
        Args:
            response: The full response text from the model
            
        Returns:
            Tuple of (reasoning, decision)
        """
        # Extract reasoning if enclosed in <think></think> tags
        reasoning_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # If there's no explicit reasoning, check if there's a natural division
        if not reasoning and len(response.split('\n\n')) > 1:
            parts = response.split('\n\n')
            # Assume first part might be reasoning
            reasoning = parts[0]
            # Rest is the decision
            decision = '\n\n'.join(parts[1:])
        else:
            # Remove the reasoning section from the response if it exists
            decision = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            # If decision is empty, use the whole response
            if not decision:
                decision = response
        
        return reasoning, decision
    
    def save_config(self, path: str) -> None:
        """Save agent configuration."""
        config = {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "reasoning_window": self.reasoning_window
        }
       

    # Add this method to your existing Agent class in base_agent.py

    def respond_to_message(self, message, conversation_history=None):
        """
        Process a message from another agent and generate a response.
        
        This is a simple wrapper around the existing respond method,
        adapting it for message-based communication.
        
        Args:
            message: Message object to respond to
            conversation_history: Optional list of previous messages
            
        Returns:
            Dictionary with response content and metadata
        """
        # Format the input by combining the message content with any context
        scenario = message.content
        
        # If there's a conversation history, format it for the agent
        formatted_history = None
        if conversation_history:
            formatted_history = []
            for msg in conversation_history:
                if isinstance(msg, dict):  # Support both dict and Message objects
                    formatted_history.append(msg)
                else:
                    formatted_history.append({
                        "role": msg.sender,
                        "content": msg.content
                    })
        
        # Use the existing respond method
        return self.respond(scenario, formatted_history)

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{self.name}_config.json"), "w") as f:
            json.dump(config, f, indent=2)