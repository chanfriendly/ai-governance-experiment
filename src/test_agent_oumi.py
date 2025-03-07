#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from typing import Dict, Any
import datetime
import time
import platform  # Add this import
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_agent_oumi")

# Import libraries based on platform
try:
    from oumi.core.configs import (
        ModelParams, 
        GenerationParams, 
        InferenceConfig
    )
    from oumi.core.types.conversation import Conversation, Message, Role
    
    # In test_agent_oumi.py, try this import approach:
    if platform.system() == "Windows":
        # Try to import from src.windows_inference directly
        try:
            from src.windows_inference import WindowsInferenceEngine
            logger.info("Using Windows inference engine")
        except ImportError:
            # Try relative import
            try:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from windows_inference import WindowsInferenceEngine
                logger.info("Using Windows inference engine from current directory")
            except ImportError as e:
                logger.error(f"Could not import WindowsInferenceEngine. Error: {e}")
                logger.error("Make sure windows_inference.py is in the src directory.")
                raise
    else:
        # For non-Windows platforms, use the regular Oumi engines
        try:
            from oumi.inference import (
                NativeTextInferenceEngine,
                VLLMInferenceEngine, 
                LlamaCppInferenceEngine
            )
            logger.info(f"Using standard Oumi inference engines on {platform.system()}")
        except ImportError:
            logger.error("Oumi inference engines not available")
            raise
except ImportError as e:
    logger.error(f"Required libraries not installed: {e}")
    logger.error("Please install the required libraries with: pip install oumi ctransformers")
    sys.exit(1)

def read_file(file_path: str) -> str:
    """Read content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return f"Error reading file: {str(e)}"

def get_model_path(config: Dict[str, Any]) -> str:
    """
    Resolve the model path using various methods:
    1. Check if an environment variable is specified and use that
    2. Use the direct path specified in the config
    3. Use a relative path from the current directory
    
    Returns:
        Resolved path to the model file
    """
    model_path = config['model']['model_name']
    
    # Handle Windows backslashes vs Unix forward slashes
    if platform.system() == "Windows":
        model_path = model_path.replace("/", "\\")
    else:
        model_path = model_path.replace("\\", "/")
    
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
        "models",
        os.path.join(".", "models"),
        os.path.join("..", "models"),
        os.path.join(os.path.expanduser("~"), "models")
    ]
    
    # Add platform-specific locations
    if platform.system() == "Windows":
        common_locations.extend([
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "models"),
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "llama.cpp")
        ])
    else:
        common_locations.extend([
            os.path.join(os.path.expanduser("~"), ".cache", "models"),
            os.path.join(os.path.expanduser("~"), ".cache", "llama.cpp")
        ])
    
    model_name = os.path.basename(model_path)
    for location in common_locations:
        check_path = os.path.join(location, model_name)
        if os.path.exists(check_path):
            return os.path.abspath(check_path)
    
    # If we can't find the model, warn but return the original path
    # (the error will be handled later when trying to load the model)
    logger.warning(f"Could not find model at {model_path} or in common locations.")
    logger.warning(f"Please place your model file at {os.path.abspath(model_path)} or set the {config['model'].get('model_path_env', 'MODEL_PATH')} environment variable.")

    return model_path

def get_inference_engine(config: Dict[str, Any]):
    """
    Get the appropriate inference engine based on configuration and platform.
    """
    engine_type = config.get('engine', 'NATIVE')
    
    # Resolve the model path
    model_path = get_model_path(config)
    logger.info(f"Using model: {model_path}")
    
    # Create the ModelParams object
    # For LLAMACPP engine, don't pass certain parameters directly
    if platform.system() != "Windows" and engine_type == 'LLAMACPP':
        # Create base model params
        model_params = ModelParams(
            model_name=model_path,
            trust_remote_code=config['model'].get('trust_remote_code', False),
        )
        
        # Carefully add model_kwargs to avoid duplication
        if 'model_kwargs' in config['model']:
            model_params.model_kwargs = {}
            for k, v in config['model']['model_kwargs'].items():
                # Skip n_ctx for LLAMACPP as it may be passed elsewhere
                if k != 'n_ctx' or engine_type != 'LLAMACPP':
                    model_params.model_kwargs[k] = v
    else:
        # For other engines, pass all parameters
        model_params = ModelParams(
            model_name=model_path,
            trust_remote_code=config['model'].get('trust_remote_code', False),
            torch_dtype_str=config['model'].get('torch_dtype_str', 'float16'),
            device_map=config['model'].get('device_map', 'auto')
        )
        
        # Add model_kwargs if present
        if 'model_kwargs' in config['model']:
            model_params.model_kwargs = config['model']['model_kwargs']
    
    # Try to create the engine with appropriate error handling
    try:
        # Use Windows-specific engine on Windows
        if platform.system() == "Windows":
            logger.info("Creating Windows inference engine")
            return WindowsInferenceEngine()
        else:
            # Use appropriate Oumi engine based on configuration
            logger.info(f"Creating {engine_type} inference engine")
            if engine_type == 'VLLM':
                return VLLMInferenceEngine(model_params=model_params)
            elif engine_type == 'LLAMACPP':
                return LlamaCppInferenceEngine(model_params=model_params)
            else:  # Default to NATIVE
                return NativeTextInferenceEngine(model_params=model_params)
    except Exception as e:
        logger.error(f"Error creating inference engine: {e}")
        logger.error("\nTroubleshooting tips:")
        logger.error("1. Check if the model name is valid and exists")
        logger.error("2. If using a local model, make sure the path is correct")
        logger.error("3. For Windows, make sure PyTorch and transformers are installed")
        logger.error("4. For GGUF models, make sure they're compatible with your engine")
        raise

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
    
    return full_path

def test_agent(config_path: str, agent_name: str = None, scenario_name: str = None, run_id: str = None) -> str:
    """Test an agent with a scenario using Oumi."""
    # Load configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise
    
    # Override agent if specified
    if agent_name:
        config['agent']['name'] = agent_name
        config['agent']['prompt_template_path'] = os.path.join("configs", "agents", agent_name, "prompt.txt")
    
    # Override scenario if specified
    if scenario_name:
        config['scenario']['path'] = os.path.join("data", "scenarios", f"{scenario_name}.txt")
    else:
        # Extract scenario name from path if not specified
        scenario_name = os.path.splitext(os.path.basename(config['scenario']['path']))[0]
    
    # Read the system prompt and scenario
    system_prompt = read_file(config['agent']['prompt_template_path'])
    scenario = read_file(config['scenario']['path'])
    
    # Initialize the inference engine
    engine = get_inference_engine(config)
    
    # Create inference config - important: set the chat template if needed
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
    
    # Create conversation with system prompt and scenario
    if 'chat_template' in config['model'] and config['model']['chat_template']:
        # Apply custom template processing if needed
        template = config['model']['chat_template']
        # Replace placeholders with actual content
        formatted_prompt = template.replace("{{ system_message }}", system_prompt).replace("{{ user_message }}", scenario)
        # Use a direct user message with pre-formatted content
        conversation = Conversation(messages=[
            Message(role=Role.USER, content=formatted_prompt)
        ])
        logger.info("Using custom chat template")
    else:
        # Standard conversation structure
        conversation = Conversation(messages=[
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=scenario)
        ])
    
    # Print test information
    logger.info(f"\nTesting {config['agent']['name'].upper()} AGENT")
    logger.info("-" * 50)
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"Engine: {config.get('engine', 'NATIVE')} {'(Windows)' if platform.system() == 'Windows' else ''}")
    logger.info(f"Scenario: {os.path.basename(config['scenario']['path'])}")
    logger.info("-" * 50)
    logger.info(f"Scenario Text: {scenario}")
    logger.info("-" * 50)
    
    # Generate response
    result = engine.infer_online([conversation], inference_config)
    response = result[0].messages[-1].content
    
    # Print the response
    logger.info("\nRESPONSE:")
    logger.info("-" * 50)
    logger.info(response)
    logger.info("-" * 50)
    
    # Create results directory and save response
    results_dir = get_results_dir(scenario_name, run_id)
    output_file = os.path.join(results_dir, f"{config['agent']['name']}.txt")
    
    # Save the response to a file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {config['agent']['name'].upper()} AGENT RESPONSE\n")
            f.write(f"Model: {config['model']['model_name']}\n")
            f.write(f"Engine: {config.get('engine', 'NATIVE')} {'(Windows)' if platform.system() == 'Windows' else ''}\n")
            f.write(f"Scenario: {scenario_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"SCENARIO TEXT:\n{scenario}\n")
            f.write("-" * 50 + "\n\n")
            f.write(f"RESPONSE:\n{response}\n")
        logger.info(f"Response saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving response to {output_file}: {e}")
    
    # Return the response
    return response

def main():
    """Main function for the agent test script."""
    parser = argparse.ArgumentParser(description="Test an agent using Oumi")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "agent_test_config.yaml"), 
                        help="Path to configuration file")
    parser.add_argument("--agent", type=str, 
                        choices=["effective_altruism", "deontological", "care_ethics", 
                                "democratic_process", "checks_and_balances"],
                        help="Agent specialization to test")
    parser.add_argument("--scenario", type=str, 
                        choices=["trolley", "resource_allocation", "content_moderation", "power_transfer", 
                                "trickle_down", "trolley_value", "prisoner", "shapley", "genocide", "affirmative"],
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
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            config['engine'] = args.engine
            with open(args.config, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
        except Exception as e:
            logger.error(f"Error updating engine in config: {e}")
    
    # Test the agent
    try:
        response = test_agent(args.config, args.agent, args.scenario, args.run_id)
        
        # Legacy output support (if still needed)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.info(f"Output saved to {args.output}")
    except Exception as e:
        logger.error(f"Error running test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()