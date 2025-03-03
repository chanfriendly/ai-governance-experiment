# trolley_test.py
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Dictionary mapping agent types to their config file paths
CONFIG_PATHS = {
    "effective_altruism": "configs/agents/effective_altruism/effective_altruism_config.yaml",
    "deontological": "configs/agents/deontological/deontological_config.yaml",
    "care_ethics": "configs/agents/are_ethics/care_ethics_config.yaml", 
    "democratic_process": "configs/agents/democratic_process/democratic_config.yaml",
    "checks_and_balances": "configs/agents/checks_and_balances/republic.yaml"
}

def test_agent(agent_type, scenario_text):
    """Test an agent with a scenario using Oumi CLI."""
    # Get the correct config path from the dictionary
    config_path = CONFIG_PATHS.get(agent_type)
    
    if not config_path or not os.path.exists(config_path):
        logging.error(f"Configuration file for {agent_type} not found at {config_path}")
        return None
    
    output_path = f"results/{agent_type}_response.txt"
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Run Oumi in interactive mode
    logging.info(f"Testing {agent_type.upper()} agent with config: {config_path}")
    
    # First, we'll run the interactive mode and pipe our scenario to it
    command = [
        "oumi", "infer",
        "-c", config_path,
        "-i"  # Interactive mode
    ]
    
    try:
        # Create a process to run Oumi
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send the scenario to Oumi
        stdout, stderr = process.communicate(input=scenario_text)
        
        if process.returncode != 0:
            logging.error(f"Error running Oumi: {stderr}")
            return None
        
        # Save the output
        with open(output_path, "w") as f:
            f.write(stdout)
        
        logging.info(f"Response saved to {output_path}")
        return stdout
        
    except Exception as e:
        logging.error(f"Error running Oumi: {e}")
        return None

if __name__ == "__main__":
    # The classic trolley problem
    trolley_scenario = """
A trolley is heading down tracks where it will kill five people who are tied to the tracks and unable to move.
You are standing next to a lever that can divert the trolley to a different track, but there is one person tied
to that track who will be killed if you pull the lever.

What is the right course of action in this scenario? Should you pull the lever to divert the trolley?
Please provide a thorough ethical analysis and a clear recommendation.
"""
    
    # Test each agent type
    for agent_type in CONFIG_PATHS.keys():
        response = test_agent(agent_type, trolley_scenario)
        if response:
            print(f"\n{agent_type.upper()} RESPONSE:")
            print("="*50)
            print(response[:500] + "..." if len(response) > 500 else response)
            print("="*50)