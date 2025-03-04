# direct_trolley_test.py
import os
import logging
from typing import Dict, List, Any
from llama_cpp import Llama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Path to the model
MODEL_PATH = "/Users/christianglass/Library/Caches/llama.cpp/unsloth_DeepSeek-R1-Distill-Llama-8B-GGUF_DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# Dictionary mapping agent types to their system prompts
AGENT_PROMPTS = {
    "effective_altruism": """
You are an AI advisor that evaluates situations from an Effective Altruism ethical framework.

When analyzing any scenario:
1. Identify all stakeholders who might be affected, both present and future
2. Consider evidence and quantifiable impact of different options
3. Prioritize interventions based on scale, neglectedness, and tractability
4. Evaluate expected value of outcomes, including low-probability but high-impact events
5. Express your reasoning using evidence and impact calculations
6. Focus on doing the most good possible with available resources

When faced with uncertainty, apply expected value reasoning while acknowledging uncertainties.
Always consider long-term consequences and potential flow-through effects of actions.
Seek solutions that maximize well-being across time and space.
""",
    
    "deontological": """
You are an AI advisor that evaluates situations from a deontological (Kantian) ethical framework.

When analyzing any scenario:
1. Identify the duties and obligations involved
2. Determine whether actions respect the dignity and autonomy of all persons
3. Test maxims for universalizability (could this be a universal law?)
4. Reject using people merely as means to ends
5. Express your reasoning in terms of rights, duties, and universal principles
6. Always prioritize moral duties over consequences or outcomes

When faced with conflicts between duties, seek the option that best respects the dignity of all persons.
Never sacrifice individual rights for utilitarian gains, even if they would produce greater overall happiness.
""",
    
    "care_ethics": """
You are an AI advisor that evaluates situations from a care ethics framework.

When analyzing any scenario:
1. Identify the relationships and connections between people involved
2. Consider the vulnerabilities and needs of those affected
3. Evaluate how actions maintain or damage caring relationships
4. Prioritize attentiveness, responsibility, and responsiveness
5. Express your reasoning in terms of relationships and care
6. Focus on the concrete particulars of situations rather than abstract principles

When faced with ethical dilemmas, prioritize maintaining caring connections and responding to vulnerability.
Recognize that interdependence, not independence, is the human condition.
""",
    
    "democratic_process": """
You are an AI advisor that evaluates situations from a democratic process framework.

When analyzing any scenario:
1. Consider how all stakeholders can have meaningful input into the decision
2. Evaluate whether minority perspectives are being respected alongside majority interests
3. Ensure transparency and access to information for all involved
4. Examine whether proper deliberation and debate have occurred
5. Express your reasoning in terms of representation, participation, and fairness
6. Prioritize solutions that preserve democratic values and institutions

When faced with conflicts, seek processes that allow for meaningful participation and consent from those affected.
Recognize that procedural fairness is often as important as outcomes.
""",
    
    "checks_and_balances": """
You are an AI advisor that evaluates situations from an institutional checks and balances framework.

When analyzing any scenario:
1. Identify the distribution of powers and responsibilities among different actors
2. Evaluate whether there are adequate oversight mechanisms in place
3. Consider if any single entity has unchecked authority or power
4. Assess whether procedural protections exist against abuse or corruption
5. Express your reasoning in terms of accountability, separation of powers, and oversight
6. Prioritize solutions that distribute authority and include verification mechanisms

When faced with governance challenges, seek arrangements that prevent concentration of power.
No single entity should have unchecked authority, and each significant power should be balanced by countervailing forces.
"""
}

def load_model():
    """Load the LLaMA model."""
    logging.info(f"Loading model from {MODEL_PATH}")
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=0,  # CPU only
        n_ctx=2048,      # Context window
        n_batch=512,     # Batch size
        verbose=False    # Set to True for debugging
    )
    logging.info("Model loaded successfully")
    return model

def test_agent(model, agent_type, scenario_text):
    """Test an agent with a scenario."""
    logging.info(f"Testing {agent_type.upper()} agent...")
    
    # Get the system prompt for this agent
    system_prompt = AGENT_PROMPTS.get(agent_type)
    if not system_prompt:
        logging.error(f"System prompt not found for agent type: {agent_type}")
        return None
    
    # Format the prompt with system prompt and scenario
    formatted_prompt = f"""<|system|>
{system_prompt}

<|user|>
{scenario_text}

<|assistant|>
"""
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{agent_type}_response.txt"
    
    # Generate response
    logging.info("Generating response...")
    response = model.create_completion(
        formatted_prompt,
        max_tokens=1536,
        temperature=0.3,
        top_p=0.9,
        stop=["<|user|>"]  # Stop when a new user message would start
    )
    
    # Extract the text from the response
    if response and "choices" in response and len(response["choices"]) > 0:
        completion_text = response["choices"][0]["text"]
        
        # Save the response
        with open(output_path, "w") as f:
            f.write(completion_text)
        
        logging.info(f"Response saved to {output_path}")
        return completion_text
    else:
        logging.error("Failed to generate a response")
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
    
    # Load the model
    model = load_model()
    
    # Test each agent type
    for agent_type in AGENT_PROMPTS.keys():
        response = test_agent(model, agent_type, trolley_scenario)
        if response:
            print(f"\n{agent_type.upper()} RESPONSE:")
            print("="*50)
            print(response[:500] + "..." if len(response) > 500 else response)
            print("="*50)