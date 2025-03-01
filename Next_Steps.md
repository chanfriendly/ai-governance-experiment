# AI Governance Experiment: Next Steps

This document outlines the immediate next steps for our AI governance experiment, picking up from where we left off in setting up the initial agent frameworks.

## Immediate Next Steps (Phase 1 Completion)

1. **Test Individual Agents**
   - Run the  script for each agent type with the trolley problem
   - Document the responses and analyze how well each agent maintains its specialization
   - Iterate on prompt templates if necessary to ensure clear differentiation between agents

2. **Implement Response Analysis Tools**
   - Create a simple analysis script that can identify key phrases or reasoning patterns in agent responses
   - Define metrics for measuring how well each agent adheres to its intended framework
   - Compare responses across agents to identify similarities and differences

3. **Develop Agent Communication Protocol**
   - Design a JSON schema for message exchange between agents
   - Implement basic message handling functions
   - Create a simple conversation history tracker

4. **Prepare for Phase 2**
   - Design simple two-agent conversation patterns
   - Create test scenarios for agent interactions
   - Implement a basic conversation controller

## Technical Implementation Details

### Agent Testing
To test individual agents:
```bash
./test_agent_oumi.py --agent effective_altruism --scenario trolley
./test_agent_oumi.py --agent deontological --scenario trolley
./test_agent_oumi.py --agent care_ethics --scenario trolley
./test_agent_oumi.py --agent democratic_process --scenario trolley
./test_agent_oumi.py --agent checks_and_balances --scenario trolley
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

## Preparation for Phase 2

Once Phase 1 is complete, we'll move to Phase 2 where we implement basic multi-agent interactions. Key preparations include:

1. Creating a conversation controller that manages turn-taking between agents
2. Implementing simple dialogue patterns like:
   - Statement-response
   - Proposal-critique
   - Question-answer
3. Designing metrics to evaluate the quality of multi-agent discussions

## Research Questions to Consider

As we move forward, keep these research questions in mind:
1. How consistently do the agents maintain their assigned frameworks?
2. What types of reasoning patterns emerge for each agent type?
3. Are there certain scenario types where some frameworks struggle to provide clear guidance?
4. How do we measure the "distinctiveness" of each agent's perspective?

## Resources

- Oumi documentation: https://oumi.ai/docs
- DeepSeek Model documentation: Check the official DeepSeek repository
- Multi-agent conversation examples: Review relevant academic papers on multi-agent dialogue systems
