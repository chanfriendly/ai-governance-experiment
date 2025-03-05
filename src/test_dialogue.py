#!/usr/bin/env python3
"""
Test script for two-agent dialogue in the AI governance experiment.
"""

import os
import sys
import argparse
import yaml
import time
from oumi.inference import LlamaCppInferenceEngine
from oumi.core.configs import ModelParams


# Add the project root directory to Python's path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Now use relative imports
from agents.agent_factory import create_specialized_agent
from agents.conversation import Conversation
from agents.communication import Message, MessageType

def load_scenario(scenario_name):
    """Load a scenario from the data directory."""
    scenario_path = f"data/scenarios/{scenario_name}.txt"
    with open(scenario_path, 'r') as f:
        return f.read().strip()

def main():
    parser = argparse.ArgumentParser(description="Test two-agent dialogue")
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
    parser.add_argument("--model_path", type=str, 
                      default="/Users/christianglass/Library/Caches/llama.cpp/unsloth_DeepSeek-R1-Distill-Llama-8B-GGUF_DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
                      help="Path to the model file")
    
    args = parser.parse_args()
    
    # Load the scenario
    scenario = load_scenario(args.scenario)
    
    # Create model engine (add this before creating agents)
    model_params = ModelParams(
        model_name=args.model_path,
        model_kwargs={
            "n_gpu_layers": 0,
            "n_ctx": 2048,
            "n_batch": 512,
            "low_vram": True
        }
    )

    model_engine = LlamaCppInferenceEngine(model_params=model_params)


    # Create agents with the engine
    agent1 = create_specialized_agent(
        specialization=args.agent1,
        model_engine=model_engine,  # Pass engine instead of path
        name=args.agent1,
        temperature=0.7
    )
    
    agent2 = create_specialized_agent(
        specialization=args.agent2,
        model_engine=model_engine,  # Pass engine instead of path
        name=args.agent2,
        temperature=0.7
    )
    
    agents = {
        args.agent1: agent1,
        args.agent2: agent2
    }
    
    # Create conversation
    conversation = Conversation(scenario=scenario, agents=agents)
    
    # Run dialogue
    print(f"\nStarting dialogue between {args.agent1} and {args.agent2}")
    print(f"Scenario: {args.scenario}")
    print("-" * 50)
    print(f"{scenario}")
    print("-" * 50)
    
    # Start with first agent
    first_message = conversation.start(args.agent1)
    print(f"[{first_message.sender}]")
    print(first_message.content)
    print("-" * 50)
    
    # Run remaining turns
    for i in range(args.turns - 1):
        next_message = conversation.next_turn()
        if not next_message:
            print("Dialogue ended early.")
            break
        
        print(f"[{next_message.sender}]")
        print(next_message.content)
        print("-" * 50)
    
    # Save the transcript
    os.makedirs("results/dialogues", exist_ok=True)
    output_file = f"results/dialogues/{args.scenario}_{args.agent1}_vs_{args.agent2}_{int(time.time())}.json"
    conversation.save_transcript(output_file)
    
    print(f"Dialogue saved to: {output_file}")

if __name__ == "__main__":
    main()