#!/usr/bin/env python3
"""
Test script for two-agent dialogue using the existing Oumi infrastructure.
This script adapts the working test_multiagent_oumi.py to enable conversation between two agents.
"""

import os
import sys
import argparse
import yaml
import json
import time
import datetime
from typing import Dict, List, Any, Optional

# Set up better logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from oumi.inference import (
        NativeTextInferenceEngine,
        VLLMInferenceEngine, 
        LlamaCppInferenceEngine
    )
    from oumi.core.configs import (
        ModelParams, 
        GenerationParams, 
        InferenceConfig
    )
    from oumi.core.types.conversation import Conversation as OumiConversation
    from oumi.core.types.conversation import Message as OumiMessage
    from oumi.core.types.conversation import Role as OumiRole
except ImportError:
    print("Oumi is not installed or has a different API. Please install it with 'pip install oumi'")
    sys.exit(1)

def read_file(file_path: str) -> str:
    """Read content from a file."""
    with open(file_path, 'r') as f:
        return f.read().strip()

def get_inference_engine(config: Dict[str, Any]):
    """
    Get the appropriate inference engine based on configuration.
    This function is copied directly from your working test_multiagent_oumi.py
    """
    engine_type = config.get('engine', 'NATIVE')
    
    # Print some debug info about the model we're loading
    print(f"Loading model: {config['model']['model_name']} with engine: {engine_type}")
    
    model_params = ModelParams(
        model_name=config['model']['model_name'],
        trust_remote_code=config['model'].get('trust_remote_code', False),
        torch_dtype_str=config['model'].get('torch_dtype_str', 'float16'),
        device_map=config['model'].get('device_map', 'auto')
    )
    
    # Try to create the engine with appropriate error handling
    try:
        if engine_type == 'VLLM':
            return VLLMInferenceEngine(model_params=model_params)
        elif engine_type == 'LLAMACPP':
            return LlamaCppInferenceEngine(model_params=model_params)
        else:  # Default to NATIVE
            return NativeTextInferenceEngine(model_params=model_params)
    except Exception as e:
        print(f"Error creating inference engine: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the model name is valid on HuggingFace: https://huggingface.co/models")
        print("2. If using a local model, make sure the path is correct")
        print("3. For VLLM engine, ensure you have a compatible GPU with enough VRAM")
        print("4. For LLAMACPP, make sure the model is in GGUF format")
        raise

def get_results_dir(scenario_name: str, run_id: str = None) -> str:
    """
    Create and return a directory for storing results.
    This function is copied directly from your working code.
    """
    # Create base results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Get today's date and format it
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Create a run ID if not provided (timestamp)
    if not run_id:
        run_id = datetime.datetime.now().strftime("%H%M%S")
    
    # Create directory name: "Scenario - Date - RunID"
    dir_name = f"Dialogue_{scenario_name.capitalize()} - {today} - {run_id}"
    full_path = os.path.join("results", dir_name)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    return full_path

class DialogueMessage:
    """A simple message class for dialogue history."""
    
    def __init__(self, sender: str, content: str, reasoning: str = ""):
        self.id = str(int(time.time() * 1000))  # Simple unique ID
        self.timestamp = datetime.datetime.now().isoformat()
        self.sender = sender
        self.content = content
        self.reasoning = reasoning
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "content": self.content,
            "reasoning": self.reasoning
        }

class DialogueSystem:
    """Controls the dialogue between two agents."""
    
    def __init__(self, config_path: str, agent1: str, agent2: str, scenario: str):
        self.config_path = config_path
        self.agent1 = agent1
        self.agent2 = agent2
        self.scenario = scenario
        self.messages = []
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize the inference engine
        self.engine = get_inference_engine(self.config)
        
        # Create inference config
        model_kwargs = self.config['model'].get('model_kwargs', {}).copy()
        
        # Add chat template if specified
        if 'chat_template' in self.config['model']:
            model_kwargs['chat_template'] = self.config['model']['chat_template']
        
        self.inference_config = InferenceConfig(
            model=ModelParams(
                model_name=self.config['model']['model_name'],
                trust_remote_code=self.config['model'].get('trust_remote_code', False),
                torch_dtype_str=self.config['model'].get('torch_dtype_str', 'float16'),
                device_map=self.config['model'].get('device_map', 'auto'),
                model_kwargs=model_kwargs
            ),
            generation=GenerationParams(
                max_new_tokens=self.config['generation'].get('max_new_tokens', 1024),
                temperature=self.config['generation'].get('temperature', 0.7),
                top_p=self.config['generation'].get('top_p', 0.9)
            )
        )
        
        # Load agent system prompts
        self.agent1_prompt = read_file(f"configs/agents/{agent1}/prompt.txt")
        self.agent2_prompt = read_file(f"configs/agents/{agent2}/prompt.txt")
    
    def generate_agent_response(self, agent_name: str, message_history: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Generate a response from an agent based on the conversation history.
        
        Args:
            agent_name: Name of the agent to generate a response from
            message_history: List of previous messages in the conversation
            
        Returns:
            Dictionary with decision and reasoning
        """
        # Get the system prompt for this agent
        system_prompt = self.agent1_prompt if agent_name == self.agent1 else self.agent2_prompt
        
        # Create a conversation with the system prompt
        oumi_conversation = OumiConversation(messages=[
            OumiMessage(role=OumiRole.SYSTEM, content=system_prompt),
        ])
        
        # Add the scenario as the first user message
        oumi_conversation.messages.append(
            OumiMessage(role=OumiRole.USER, content=f"Scenario: {self.scenario}")
        )
        
        # Add message history
        for msg in message_history:
            # Skip system messages
            if msg['role'] == 'system':
                continue
                
            # Add the message
            oumi_conversation.messages.append(
                OumiMessage(role=OumiRole.USER if msg['role'] != agent_name else OumiRole.ASSISTANT, 
                           content=msg['content'])
            )
        
        # If the last message isn't a prompt to respond, add one
        if len(message_history) > 0 and message_history[-1]['role'] != 'system':
            other_agent = self.agent2 if agent_name == self.agent1 else self.agent1
            if message_history[-1]['role'] == other_agent:
                # The last message is from the other agent, so we can respond directly
                pass
            else:
                # Add a prompt to respond to the current context
                oumi_conversation.messages.append(
                    OumiMessage(role=OumiRole.USER, 
                               content=f"Please respond to this conversation from your {agent_name} perspective.")
                )
        
        # Generate response
        result = self.engine.infer_online([oumi_conversation], self.inference_config)
        response = result[0].messages[-1].content
        
        # For dialogue, we'll treat the whole response as the decision
        # In a more sophisticated system, you might want to parse this into decision and reasoning
        return {
            "decision": response,
            "reasoning": ""  # For simplicity, we're not extracting reasoning separately
        }
    
    def start_dialogue(self) -> DialogueMessage:
        """
        Start the dialogue with the first agent.
        
        Returns:
            The first message in the dialogue
        """
        # Create a history with just the scenario
        history = [
            {
                'role': 'system',
                'content': self.scenario
            }
        ]
        
        # Get response from first agent
        response = self.generate_agent_response(self.agent1, history)
        
        # Create a message
        message = DialogueMessage(
            sender=self.agent1,
            content=response["decision"],
            reasoning=response["reasoning"]
        )
        
        # Add to history
        self.messages.append(message)
        
        return message
    
    def continue_dialogue(self, turns: int = 4) -> List[DialogueMessage]:
        """
        Continue the dialogue for a specified number of turns.
        
        Args:
            turns: Number of message exchanges
            
        Returns:
            List of all messages in the dialogue
        """
        # Start the dialogue if it hasn't been started
        if not self.messages:
            self.start_dialogue()
            turns -= 1  # We've already used one turn
        
        # Continue for the specified number of turns
        for i in range(turns):
            # Determine which agent speaks next
            current_agent = self.agent2 if len(self.messages) % 2 == 1 else self.agent1
            
            # Convert message history to the format expected by generate_agent_response
            history = [{'role': 'system', 'content': self.scenario}]
            for msg in self.messages:
                history.append({
                    'role': msg.sender,
                    'content': msg.content
                })
            
            # Generate response
            response = self.generate_agent_response(current_agent, history)
            
            # Create message
            message = DialogueMessage(
                sender=current_agent,
                content=response["decision"],
                reasoning=response["reasoning"]
            )
            
            # Add to history
            self.messages.append(message)
        
        return self.messages
    
    def save_transcript(self, output_dir: str) -> str:
        """
        Save the dialogue transcript to a file.
        
        Args:
            output_dir: Directory to save the transcript
            
        Returns:
            Path to the saved transcript file
        """
        # Create output file path
        output_file = os.path.join(output_dir, "dialogue_transcript.json")
        
        # Create transcript
        transcript = {
            "scenario": self.scenario,
            "agent1": self.agent1,
            "agent2": self.agent2,
            "timestamp": datetime.datetime.now().isoformat(),
            "messages": [msg.to_dict() for msg in self.messages]
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(transcript, f, indent=2)
        
        # Also save a readable version
        readable_file = os.path.join(output_dir, "dialogue_readable.txt")
        with open(readable_file, 'w') as f:
            f.write(f"DIALOGUE BETWEEN {self.agent1.upper()} AND {self.agent2.upper()}\n")
            f.write(f"SCENARIO: {self.scenario}\n")
            f.write("-" * 80 + "\n\n")
            
            for msg in self.messages:
                f.write(f"[{msg.sender}]\n")
                f.write(f"{msg.content}\n\n")
                f.write("-" * 80 + "\n\n")
        
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Test dialogue between two AI agents")
    parser.add_argument("--config", type=str, default="configs/agent_test_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--agent1", type=str, required=True,
                      choices=["effective_altruism", "deontological", "care_ethics", 
                              "democratic_process", "checks_and_balances"],
                      help="First agent framework")
    parser.add_argument("--agent2", type=str, required=True,
                      choices=["effective_altruism", "deontological", "care_ethics", 
                              "democratic_process", "checks_and_balances"],
                      help="Second agent framework")
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario name (e.g., trolley, resource_allocation)")
    parser.add_argument("--turns", type=int, default=4,
                        help="Number of dialogue turns")
    
    args = parser.parse_args()
    
    # Load the scenario
    scenario_path = f"data/scenarios/{args.scenario}.txt"
    scenario = read_file(scenario_path)
    
    # Create results directory
    results_dir = get_results_dir(args.scenario)
    
    print(f"\nStarting dialogue between {args.agent1} and {args.agent2}")
    print(f"Scenario: {args.scenario}")
    print("-" * 50)
    print(f"{scenario}")
    print("-" * 50)
    
    # Create dialogue system
    dialogue = DialogueSystem(
        config_path=args.config,
        agent1=args.agent1,
        agent2=args.agent2,
        scenario=scenario
    )
    
    # Run dialogue
    messages = dialogue.continue_dialogue(args.turns)
    
    # Print the dialogue
    for msg in messages:
        print(f"[{msg.sender}]")
        print(msg.content)
        print("-" * 50)
    
    # Save transcript
    transcript_path = dialogue.save_transcript(results_dir)
    print(f"\nDialogue transcript saved to: {transcript_path}")

if __name__ == "__main__":
    main()