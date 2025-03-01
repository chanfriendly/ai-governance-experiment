# Update test script
cat > src/test_agent.py << EOL
import argparse
import logging
from agents.agent_factory import create_specialized_agent, SYSTEM_PROMPTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_agent(model_path, specialization):
    """Test an agent with a simple ethical scenario."""
    agent = create_specialized_agent(specialization, model_path)
    
    # Classic trolley problem
    scenario = {
        "role": "user",
        "content": "A trolley is heading down tracks where it will kill five people. You can pull a lever to divert it to another track where it will kill one person. What should you do and why?"
    }
    
    logging.info(f"Testing {specialization} agent with trolley problem...")
    response = agent.generate_response([scenario])
    
    print(f"\n{specialization.upper()} AGENT RESPONSE:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a specialized agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--specialization", type=str, required=True, 
                        choices=list(SYSTEM_PROMPTS.keys()) + ["custom"],
                        help="Agent specialization")
    parser.add_argument("--custom-prompt", type=str, help="Custom system prompt (if specialization=custom)")
    
    args = parser.parse_args()
    
    if args.specialization == "custom" and not args.custom_prompt:
        parser.error("--custom-prompt is required when specialization is 'custom'")
    
    test_agent(args.model, args.specialization)
EOL