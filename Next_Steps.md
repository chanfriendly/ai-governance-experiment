# AI Governance Experiment: Next Steps

This document outlines the immediate next steps for our AI governance experiment, picking up from where we left off in setting up the initial agent frameworks.

## Immediate Next Steps (Phase 1 Completion)

1. **Refactor Code for Platform Independence**
   - Replace hardcoded paths (e.g., `/Users/[my_user]/...`) with relative paths or environment variables
   - Create a configuration file for system-specific settings
   - Add documentation on setup requirements for other researchers
   - Ensure all paths use OS-agnostic path joining (`os.path.join`)

2. **Improve Oumi Integration**
   - Fix configuration issues with Oumi's expected format
   - Update YAML files to properly reference prompts
   - Create a working version of `trolley_test.py` using Oumi
   - Document correct Oumi configuration format for future reference

3. **Analyze Initial Agent Responses**
   - Compare responses from each agent for the trolley problem
   - Identify distinct reasoning patterns for each framework
   - Document how well each agent adheres to its assigned framework
   - Note any unexpected results or interesting patterns

4. **Expand Scenario Library**
   - Create additional ethical scenarios beyond the trolley problem
   - Implement resource allocation scenarios
   - Develop content moderation test cases
   - Design scenarios with competing stakeholder interests

5. **Prepare for Phase 2**
   - Design agent-to-agent communication protocol
   - Create conversation tracking mechanism
   - Implement simple two-agent conversation pattern
   - Develop framework for measuring interaction quality

## Technical Implementation Details

### Refactoring for Platform Independence

Replace hardcoded paths:

```python
# Before
MODEL_PATH = "/Users/christianglass/Library/Caches/llama.cpp/unsloth_DeepSeek-R1-Distill-Llama-8B-GGUF_DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# After
import os
from pathlib import Path

# Either use environment variables
MODEL_PATH = os.environ.get("MODEL_PATH")

# Or use relative paths from project root
PROJECT_ROOT = Path(__file__).parent.absolute()
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "deepseek-r1-distill-llama-8b.gguf")
```


### Message Format Design
Design a message format for inter-agent communication that includes:
- Message ID
- Sender agent
- Recipient agent(s)
- Message type (proposal, critique, question, response)
- Content
- References to previous messages
- Reasoning trace

### Communication Protocol
Implement basic functions for:
- Sending messages between agents
- Updating conversation history
- Formatting agent prompts with conversation context
- Extracting structured information from responses

### Oumi Configuration Format
Correct Oumi configuration format (based on their documentation):
``` yaml
model:
  model_name: "path/to/model.gguf"  # Use relative path
  model_kwargs:
    n_gpu_layers: 0
    n_ctx: 2048
    n_batch: 512
    low_vram: true
  # Reference prompt file instead of embedding in YAML
  # system_prompt_file: "path/to/prompt.txt"  # If this syntax is supported
  
generation:
  max_new_tokens: 1536
  temperature: 0.3
  top_p: 0.9
  
engine: LLAMACPP
```

## Preparation for Phase 2

Once Phase 1 is complete, we'll move to Phase 2 where we implement basic multi-agent interactions. Key preparations include:

1. Creating a conversation controller that manages turn-taking between agents
2. Implementing simple dialogue patterns like:
   - Statement-response
   - Proposal-critique
   - Question-answer
3. Developing Conversation Analysis Tools

   - Create metrics for measuring interaction quality
   - Implement tools for detecting emergent patterns
   - Develop visualization for conversation flow
   - Create framework for measuring agreement/disagreement


## Research Questions to Consider

As we move forward, keep these research questions in mind:
1. How consistently do the agents maintain their assigned frameworks?
2. What types of reasoning patterns emerge for each agent type?
3. Are there certain scenario types where some frameworks struggle to provide clear guidance?
4. How do we measure the "distinctiveness" of each agent's perspective?

## Resources

- Oumi documentation: https://oumi.ai/docs
- Unsloth for distilled and quantized models: https://github.com/unslothai/unsloth
- DeepSeek Model documentation: https://github.com/deepseek-ai/DeepSeek-R1