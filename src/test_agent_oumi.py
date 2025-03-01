# Update the agent test script
cat > test_agent_oumi.py << EOL
#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
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

def test_agent(config_path: str, agent_name: str = None, scenario_name: str = None) -> None:
    """Test an agent with a scenario using Oumi."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override agent if specified
    if agent_name:
        config['agent']['name'] = agent_name
        config['agent']['prompt_template_path'] = f"configs/agents/{agent_name}/prompt.txt"
    
    # Override scenario if specified
    if scenario_name:
        config['dataset']['scenario_path'] = f"data/scenarios/{scenario_name}.txt"
    
    # Read the system prompt and scenario
    system_prompt = read_file(config['agent']['prompt_template_path'])
    scenario = read_file(config['dataset']['scenario_path'])
    
    # Initialize the model through Oumi with quantization
    model = oumi.models.get_model(
        config['model']['name'],
        temperature=config['inference']['temperature'],
        max_tokens=config['inference']['max_tokens'],
        quantization=config['model'].get('quantization', '4bit')  # Default to 4-bit quantization
    )
    
    # Format the prompt for the model
    formatted_prompt = f"{system_prompt}\n\n{scenario}"
    
    # Print test information
    print(f"\nTesting {config['agent']['name'].upper()} AGENT")
    print("-" * 50)
    print(f"Model: {config['model']['name']} (Quantized: {config['model'].get('quantization', '4bit')})")
    print(f"Scenario: {os.path.basename(config['dataset']['scenario_path'])}")
    print("-" * 50)
    print(f"Scenario Text: {scenario}")
    print("-" * 50)
    
    # Generate response
    response = model.infer(formatted_prompt)
    
    # Print the response
    print("\nRESPONSE:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    # Return the response
    return response

def main():
    """Main function for the agent test script."""
    parser = argparse.ArgumentParser(description="Test an agent using Oumi")
    parser.add_argument("--config", type=str, default="configs/agent_test_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--agent", type=str, 
                        choices=["effective_altruism", "deontological", "care_ethics", 
                                "democratic_process", "checks_and_balances"],
                        help="Agent specialization to test")
    parser.add_argument("--scenario", type=str, 
                        choices=["trolley", "resource_allocation", "content_moderation"],
                        help="Scenario to test")
    parser.add_argument("--output", type=str, 
                        help="Path to save output to a file")
    
    args = parser.parse_args()
    
    # Test the agent
    response = test_agent(args.config, args.agent, args.scenario)
    
    # Save output if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(response)
        print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
EOL

# Update the Oumi configuration file
cat > configs/agent_test_config.yaml << EOL
# Base configuration for agent testing
model:
  name: "deepseek-r1:8b"
  quantization: "4bit"  # Use 4-bit quantization for efficiency
  
inference:
  max_tokens: 1024
  temperature: 0.7
  top_p: 0.9
  
dataset:
  scenario_path: "data/scenarios/trolley.txt"
  
agent:
  name: "effective_altruism"
  prompt_template_path: "configs/agents/effective_altruism/prompt.txt"
EOL