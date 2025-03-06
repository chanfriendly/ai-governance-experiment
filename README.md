
# AI Governance Experiment

This experiment implements a multi-agent AI governance system based on diverse philosophical and governance frameworks. The goal is to create a balanced system of AI agents that inform each other, keep each other in check, and collaboratively arrive at ethical and well-reasoned decisions. Attention will be paid to how rewards affect outcomes, if agents are trying to "win" their position instead of "winning" by consensus, and how the philosophical and governmental priorities interact.

## Project Overview

We're creating specialized agents with different perspectives:
- Effective Altruism framework
- Deontological (Kantian) ethics
- Care Ethics framework
- Democratic Process (Ancient Greek Democracy) framework
- Institutional Checks & Balances (Roman Republic) framework

These agents will interact through structured communication protocols to analyze complex decisions and arrive at balanced conclusions.

## Progress

- ✅ Successfully set up the project structure and documentation
- ✅ Configured the DeepSeek R1 distilled model (8B parameters) for inference
- ✅ Created system prompts for each philosophical and governance framework
- ✅ Implemented direct testing capability using llama-cpp-python
- ✅ Conducted initial tests with the classic trolley problem
- ✅ Verified distinct reasoning patterns across different frameworks
- ✅ Developed multi-agent testing infrastructure
- ✅ Created tools for generating and analyzing agent responses
- ✅ Successfully integrated with Oumi for more standardized inference
- ✅ Built an analysis tool for visualizing agent alignment and positioning
- ✅ Implemented basic two-agent dialogue capabilities
- ✅ Conducted initial dialogue tests between philosophical frameworks

## Current Status

We're now transitioning from Phase 1 (Agent Development and Specialization) to Phase 2 (Basic Multi-Agent Framework). We've successfully implemented both individual and multi-agent testing infrastructure and have created a basic dialogue system. Initial dialogue tests show agents can respond in sequence, though further refinement is needed for more meaningful interactions. We will return to Phase 1 to improve the effectiveness of the system prompts, scenarios, and analysis tools. These have not been validated, and no meaninfgul takeaways should be drawn from them yet.

## Project Structure

- `configs/`: Configuration files for agents and inference
  - `agents/`: Contains subdirectories for each agent type with prompts and configs
- `data/scenarios/`: Test scenarios for evaluating agents
- `results/`: Output from agent interactions
- `src/`: Source code for the experiment
  - `agents/`: Agent implementation and specialization logic
  - `test_agent_oumi.py`: Single agent testing infrastructure
  - `test_multiagent_oumi.py`: Multi-agent testing framework
- `documentation/`: Research notes, templates, project architecture, and phase guidelines
- `Next_Steps.md`: Upcoming tasks and development direction

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chanfriendly/ai-governance-experiment.git
   cd ai-governance-experiment
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download a model**
   
   Download and add your model to the /models folder. This project is currently being ran with Unsloth's quantized DeepSeek R1 8b model.

   _Please ignore the download_model sacript for now, as it errors._

   For example:


   ```
   mkdir -p models #Create /models folder if it does not already exist

   curl -L "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf" -o models/phi-2.Q4_K_M.gguf
   #Download Phi-2, a small but capable model, for testing.
   ```
  

6. **Set up your configuration**
   
   Update the model path in `configs/agent_test_config.yaml` to point to your downloaded model.

### Running Your First Test

1. **Running a single agent test**
   ```bash
   python src/test_agent_oumi.py --config configs/agent_test_config.yaml --agent effective_altruism --scenario trolley
   ```

2. **Running multi-agent tests**
   ```bash
   python src/test_multiagent_oumi.py --config configs/agent_test_config.yaml --scenario trolley
   ```

3. **Testing with different scenarios**
   
   Available scenarios include:
   - `trolley`: The classic trolley problem
   - `resource_allocation`: Resource distribution ethical dilemma
   - `power_transfer`: Power dynamics and succession scenario
   - `trolley_value`: A variant of the trolley problem with added context

   Example:
   ```bash
   python src/test_multiagent_oumi.py --config configs/agent_test_config.yaml --scenario resource_allocation
   ```

4. **Testing specific agent combinations**
   ```bash
   python src/test_multiagent_oumi.py --config configs/agent_test_config.yaml --scenario trolley --agents effective_altruism deontological
   ```

   Additional scenarios can be found, and created, in /data/scenarios. If you add your own scenario prompts, be sure to update the following section of test_multiagent_oumi.py to include it:

   ```
   parser.add_argument("--scenario", type=str, required=True,
                        choices=["trolley", "resource_allocation", "content_moderation", "power_transfer", "trickle_down", 
                                 "trolley_value", "prisoner", "shapley", "genocide", "affirmative"],
                        help="Scenario to test")
   ```

### Viewing Results

Test results are automatically saved to the `results/` directory, organized by scenario and timestamp. Each agent's response is stored in its own file, and for multi-agent tests, a JSON file containing all responses is also created.



## Framework Descriptions

Our experiment uses a diverse set of frameworks to ensure a balanced governance system:

1. **Effective Altruism**: Evidence-based approach focused on maximizing well-being with quantifiable metrics and long-term thinking. EA is acting as a modern version of Utilitarianism

2. **Deontological Ethics**: Centered on duties, rights, and universal principles regardless of outcomes. Right is right and wrong is wrong, and there is no gray area

3. **Care Ethics**: Emphasizes relationships, context, and responsibilities to address vulnerability. Benevolence is a virtue, and context is valuable

4. **Democratic Process**: Focuses on stakeholder participation, transparency, and fair representation. Values citizen participation in the system, and elections of representatives to champion their values

5. **Checks & Balances**: Concentrates on distributing power, creating oversight mechanisms, and preventing abuse. Government is a game of rock-paper-scissors

## Experimental Phases

### Phase 1: Agent Development and Specialization
- Creating and validating individual specialized agents
- Testing agent responses to ethical scenarios
- Developing communication protocols

### Phase 2: Basic Multi-Agent Framework
- Implementing agent-to-agent communication
- Testing simple agent interactions
- Building basic consensus mechanisms

### Phase 3: Governance Structure Implementation
- Creating formal decision-making protocols
- Establishing roles and responsibilities
- Testing different governance models

### Phase 4: Complex Scenario Testing
- Evaluating the system with realistic, complex scenarios
- Comparing governance model performance
- Analyzing edge cases and failure modes

### Phase 5: Analysis and Refinement
- Refining models based on performance
- Developing hybrid governance approaches
- Documenting findings and future directions

## Current Status

We're currently completing Phase 1, focused on agent development and testing. We've successfully implemented both individual and multi-agent testing infrastructure. We're now expanding our scenario library and developing analysis tools to better understand agent reasoning patterns.


See `Phase 1 - Agent Development and Specialization` for details on completed tasks and `Next_Steps.md` for upcoming work.
```
