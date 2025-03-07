def get_model_path(config: dict[str, any]) -> str:
    """
    Resolve the model path using various methods:
    1. Check if an environment variable is specified and use that
    2. Use the direct path specified in the config
    3. Use a relative path from the current directory
    
    Returns:
        Resolved path to the model file
    """
    model_path = config['model']['model_name']
    
    # Check if an environment variable is specified
    if 'model_path_env' in config['model'] and config['model']['model_path_env'] in os.environ:
        env_path = os.environ[config['model']['model_path_env']]
        if os.path.isfile(env_path):
            return env_path
        elif os.path.isdir(env_path):
            # If the env var points to a directory, join with the model name
            model_name = os.path.basename(model_path)
            return os.path.join(env_path, model_name)
    
    # If the path is absolute and exists, use it directly
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    
    # Check if the model exists in the relative path
    if os.path.exists(model_path):
        return os.path.abspath(model_path)
    
    # Try to find the model in common locations
    common_locations = [
        "models/",
        "./models/",
        "../models/",
        os.path.expanduser("~/models/"),
        os.path.expanduser("~/.cache/models/"),
        os.path.expanduser("~/.cache/llama.cpp/")
    ]
    
    model_name = os.path.basename(model_path)
    for location in common_locations:
        check_path = os.path.join(location, model_name)
        if os.path.exists(check_path):
            return os.path.abspath(check_path)
    
    # If we can't find the model, warn but return the original path
    # (the error will be handled later when trying to load the model)
    print(f"Warning: Could not find model at {model_path} or in common locations.")
    print(f"Please place your model file at {os.path.abspath(model_path)} or set the {config['model'].get('model_path_env', 'MODEL_PATH')} environment variable.")
    
    return model_path


def get_results_dir(scenario_name: str, run_id: str = None) -> str:
    """
    Create and return a directory for storing results.
    
    Args:
        scenario_name: Name of the scenario being tested
        run_id: Optional run identifier (defaults to timestamp)
        
    Returns:
        Path to the results directory
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
    dir_name = f"{scenario_name.capitalize()} - {today} - {run_id}"
    full_path = os.path.join("results", dir_name)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    return full_path#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import json
from typing import Dict, List, Any
import datetime
import time

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
    from oumi.core.types.conversation import Conversation, Message, Role
except ImportError:
    print("Oumi is not installed or has a different API. Please install it with 'pip install oumi'")
    sys.exit(1)

# Set up better logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def read_file(file_path: str) -> str:
    """Read content from a file."""
    with open(file_path, 'r') as f:
        return f.read().strip()

def get_inference_engine(config: Dict[str, Any]):
    """
    Get the appropriate inference engine based on configuration.
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

def test_agent(engine, inference_config: InferenceConfig, agent_name: str, scenario: str, results_dir: str = None) -> Dict[str, Any]:
    """Test a single agent with a scenario using Oumi."""
    # Read the system prompt
    prompt_path = f"configs/agents/{agent_name}/prompt.txt"
    system_prompt = read_file(prompt_path)
    
    # Create conversation with system prompt and scenario
    if 'chat_template' in inference_config.model.model_kwargs:
        # Apply custom template processing if needed
        template = inference_config.model.model_kwargs['chat_template']
        # Replace placeholders with actual content
        formatted_prompt = template.replace("{{ system_message }}", system_prompt).replace("{{ user_message }}", scenario)
        # Use a direct user message with pre-formatted content
        conversation = Conversation(messages=[
            Message(role=Role.USER, content=formatted_prompt)
        ])
    else:
        # Standard conversation structure
        conversation = Conversation(messages=[
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=scenario)
        ])
    
    # Generate response
    result = engine.infer_online([conversation], inference_config)
    response = result[0].messages[-1].content
    
    # Save the response to a file if results_dir is provided
    if results_dir:
        output_file = os.path.join(results_dir, f"{agent_name}.txt")
        with open(output_file, 'w') as f:
            f.write(f"# {agent_name.upper()} AGENT RESPONSE\n\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"# GENERATION PARAMETERS\n")
            f.write(f"Temperature: {inference_config.generation.temperature}\n")
            f.write(f"Max tokens: {inference_config.generation.max_new_tokens}\n")
            f.write(f"Top-p: {inference_config.generation.top_p}\n\n")
            f.write(f"SYSTEM PROMPT:\n{system_prompt}\n\n")
            f.write(f"SCENARIO:\n{scenario}\n\n")
            f.write(f"RESPONSE:\n{response}\n")
        print(f"Response for {agent_name} saved to: {output_file}")
    
    return {
        "agent": agent_name,
        "response": response
    }

def test_multiple_agents(config_path: str, agents: List[str], scenario_name: str, run_id: str = None) -> Dict[str, Any]:
    """Test multiple agents with the same scenario."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Read the scenario
    scenario_path = f"data/scenarios/{scenario_name}.txt"
    scenario = read_file(scenario_path)
    
    # Create results directory
    results_dir = get_results_dir(scenario_name, run_id)
    print(f"Saving results to: {results_dir}")
    
    # Initialize the inference engine
    engine = get_inference_engine(config)
    
    # Create inference config
    model_kwargs = config['model'].get('model_kwargs', {}).copy()
    
    # Add chat template if specified
    if 'chat_template' in config['model']:
        model_kwargs['chat_template'] = config['model']['chat_template']
    
    inference_config = InferenceConfig(
        model=ModelParams(
            model_name=config['model']['model_name'],
            trust_remote_code=config['model'].get('trust_remote_code', False),
            torch_dtype_str=config['model'].get('torch_dtype_str', 'float16'),
            device_map=config['model'].get('device_map', 'auto'),
            model_kwargs=model_kwargs
        ),
        generation=GenerationParams(
            max_new_tokens=config['generation'].get('max_new_tokens', 1024),
            temperature=config['generation'].get('temperature', 0.7),
            top_p=config['generation'].get('top_p', 0.9)
        )
    )
    
    # Print test information
    print(f"\nTesting {len(agents)} AGENTS on scenario: {scenario_name}")
    print("-" * 50)
    print(f"Model: {config['model']['model_name']}")
    print(f"Engine: {config.get('engine', 'NATIVE')}")
    print(f"Agents: {', '.join(agents)}")
    print("-" * 50)
    print(f"Scenario Text: {scenario}")
    print("-" * 50)
    
    # Save scenario and test configuration
    with open(os.path.join(results_dir, "scenario.txt"), 'w') as f:
        f.write(f"# SCENARIO: {scenario_name}\n\n")
        f.write(f"{scenario}\n\n")
        f.write(f"# TEST CONFIGURATION\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {config['model']['model_name']}\n")
        f.write(f"Engine: {config.get('engine', 'NATIVE')}\n")
        f.write(f"Agents: {', '.join(agents)}\n\n")
        
        # Include generation parameters
        f.write(f"# GENERATION PARAMETERS\n")
        f.write(f"Temperature: {inference_config.generation.temperature}\n")
        f.write(f"Max tokens: {inference_config.generation.max_new_tokens}\n")
        f.write(f"Top-p: {inference_config.generation.top_p}\n")
        
        # Include additional model parameters if available
        if model_kwargs:
            f.write(f"\n# MODEL PARAMETERS\n")
            for key, value in model_kwargs.items():
                if key != 'chat_template':  # Skip the template as it can be verbose
                    f.write(f"{key}: {value}\n")
            
            # Save chat template separately if it exists
            if 'chat_template' in model_kwargs:
                f.write(f"\n# CHAT TEMPLATE\n")
                f.write(f"{model_kwargs['chat_template']}\n")
    
    # Test each agent
    results = []
    for agent_name in agents:
        print(f"\nTesting {agent_name.upper()} AGENT...")
        result = test_agent(engine, inference_config, agent_name, scenario, results_dir)
        results.append(result)
        
        # Print the response
        print(f"RESPONSE FROM {agent_name.upper()}:")
        print("-" * 50)
        print(result["response"])
        print("-" * 50)
    
    # Save the complete results as JSON
    complete_results = {
        "scenario": scenario_name,
        "scenario_text": scenario,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": config['model']['model_name'],
        "engine": config.get('engine', 'NATIVE'),
        "agents": agents,
        "generation_parameters": {
            "temperature": inference_config.generation.temperature,
            "max_tokens": inference_config.generation.max_new_tokens,
            "top_p": inference_config.generation.top_p
        },
        "model_parameters": {
            k: v for k, v in model_kwargs.items() if k != 'chat_template'
        },
        "results": results
    }
    
    with open(os.path.join(results_dir, "results.json"), 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    return complete_results

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
                        choices=["trolley", "resource_allocation", "content_moderation", "power_transfer", "trickle_down", 
                                 "trolley_value", "prisoner", "shapley", "genocide", "affirmative"],
                        help="Scenario to test")
    parser.add_argument("--engine", type=str,
                        choices=["NATIVE", "VLLM", "LLAMACPP"],
                        help="Inference engine to use")
    parser.add_argument("--run-id", type=str,
                        help="Optional run identifier for organizing results")
    parser.add_argument("--output", type=str, 
                        help="Path to save output to a file (legacy)")
    
    args = parser.parse_args()
    
    # Update engine in config if specified
    if args.engine:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['engine'] = args.engine
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    
    # Test the agents
    results = test_multiple_agents(args.config, args.agents, args.scenario, args.run_id)
    
    # Legacy output support (if still needed)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()