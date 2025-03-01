# AI Governance Experiment

This project implements a multi-agent AI governance system based on diverse philosophical frameworks. The goal is to create a balanced system of AI agents that inform each other, keep each other in check, and collaboratively arrive at ethical decisions.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m virtualenv venv
   source venv/bin/activate  # On macOS/Linux
   # or venv\Scripts\activate on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and quantize a model:
   ```bash
   python src/setup_model.py --model "DeepseekAI/deepseek-r1-distill-8b" --bits 4
   # or
   python src/setup_model.py --model "meta-llama/Llama-3-8B-hf" --bits 4
   # or
   python src/setup_model.py --model "mistralai/Mistral-7B-v0.1" --bits 4
   ```

4. Test an agent:
   ```bash
   python src/test_agent.py --model "models/deepseek-r1-distill-8b_4bit" --specialization utilitarian
   ```

## Project Structure

- `src/agents/`: Agent implementations
- `src/governance/`: Governance mechanisms
- `src/evaluation/`: Evaluation frameworks
- `src/utils/`: Utility functions
- `data/`: Test scenarios and data
- `models/`: Downloaded and quantized models
- `experiments/`: Experiment results
- `logs/`: Log files

## Phase 1: Agent Development and Specialization

Current phase focuses on creating and validating individual specialized agents based on:
- Utilitarian ethics
- Deontological (Kantian) ethics
- Virtue ethics
- Care ethics

Each agent is provided with a specialized system prompt that guides its reasoning approach.
