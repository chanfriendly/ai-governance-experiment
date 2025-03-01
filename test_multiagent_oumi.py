#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import json
from typing import Dict, List, Any

try:
    import oumi
except ImportError:
    print("Oumi is not installed. Please install it with 'pip install oumi'")
    sys.exit(1)

def read_file(file_path: str) -> str:
    """Read content from a file."""
    with open(file_path, 'r') as f:
        return f.read().strip()

def test_agent(model, agent_name: str, scenario: str) -> Dict[str, Any]:
    """Test a single agent with a scenario using Oumi."""
    # Read the system prompt
    prompt_path = f"configs/agents/{agent_name}/prompt.txt"
    system_prompt = read_file(prompt_path)
    
    # Format the prompt for the model
    formatted_prompt = f"{system_prompt}\n\n{scenario}"
    
    # Generate response
    response = model.infer(formatted_prompt)
    
    return {
        "agent": agent_name,
        "response": response
    }

def test_multiple_agents(config_path: str, agents: List[str], scenario_name: str) -> Dict[str, Any]:
    """Test multiple agents with the same scenario."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Read the scenario
    scenario_path = f"data/scenarios/{scenario_name}.txt"
    scenario = read_file(scenario_path)
    
    # Initialize the model through Oumi
    model = oumi.models.get_model(
        config['model']['name'],
        temperature=config['inference']['temperature'],
        max_tokens=config['inference']['max_tokens']
    )
    
    # Print test information
    print(f"\nTesting {len(agents)} AGENTS on scenario: {scenario_name}")
    print("-" * 50)
    print(f"Model: {config['model']['name']}")
    print(f"Agents: {', '.join(agents)}")
    print("-" * 50)
    print(f"Scenario Text: {scenario}")
    print("-" * 50)
    
    # Test each agent
    results = []
    for agent_name in agents:
        print(f"\nTesting {agent_name.upper()} AGENT...")
        result = test_agent(model, agent_name, scenario)
        results.append(result)
        
        # Print the response
        print(f"RESPONSE FROM {agent_name.upper()}:")
        print("-" * 50)
        print(result["response"])
        print("-" * 50)
    
    return {
        "scenario": scenario_name,
        "scenario_text": scenario,
        "model": config['model']['name'],
        "agents": agents,
        "results": results
    }

def main():
    """Main function for the multi-agent test script."""
    parser = argparse.ArgumentParser(description="Test multiple agents using Oumi")
    parser.add_argument("--config", type=str, default="configs/agent_test_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--agents", type=str, nargs="+", 
                        choices=["effective_altruism", "deontological", "care_ethics", 
                                "democratic_process", "checks_and_balances"],
                        default=["effective_altruism", "deontological", "care_ethics", 
                                "democratic_process", "checks_and_balances"],
                        help="Agents to test")
    parser.add_argument("--scenario", type=str, required=True,
                        choices=["trolley", "resource_allocation", "content_moderation"],
                        help="Scenario to test")
    parser.add_argument("--output", type=str, 
                        help="Path to save output to a file")
    
    args = parser.parse_args()
    
    # Test the agents
    results = test_multiple_agents(args.config, args.agents, args.scenario)
    
    # Save output if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
