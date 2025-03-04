# AI Governance Experiment: Next Steps

This document outlines the immediate next steps for our AI governance experiment, picking up from where we left off in setting up the initial agent frameworks.

## Immediate Next Steps (Phase 1 Completion)

1. **Response Analysis Tool Development (HIGH PRIORITY)**
- Create tools to analyze response patterns across different agents
- Implement keyword and theme extraction from agent responses
- Develop metrics for comparing reasoning approaches
- Build visualizations to highlight differences in agent decision-making
- Create a framework for evaluating framework adherence

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

## Analysis Tool Implementation Details

### Response Pattern Analyzer

```python
class ResponseAnalyzer:
    def __init__(self):
        self.frameworks = ["effective_altruism", "deontological", "care_ethics", 
                          "democratic_process", "checks_and_balances"]
        self.framework_keywords = {
            "effective_altruism": ["utility", "outcomes", "maximize", "consequentialist", 
                                  "evidence", "calculation", "impact", "greater good"],
            "deontological": ["duty", "obligation", "rights", "universal law", "dignity", 
                            "categorical", "principle", "inherent"],
            "care_ethics": ["relationship", "care", "vulnerability", "context", 
                          "connection", "responsibility", "attentiveness"],
            "democratic_process": ["participation", "stakeholder", "transparency", 
                                  "representation", "vote", "deliberation", "consent"],
            "checks_and_balances": ["oversight", "power", "distribution", "accountability", 
                                   "authority", "procedure", "institutional"]
        }
    
    def analyze_response(self, agent_name, response_text):
        """Analyze an agent's response for framework adherence and reasoning patterns"""
        # Implementation here
        
    def compare_responses(self, responses_dict):
        """Compare responses across different agents"""
        # Implementation here
        
    def visualize_comparison(self, comparison_data):
        """Create visualization of response comparisons"""
        # Implementation here
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

### Framework Adherence Metrics

Implementing the following metrics for each agent response:
1. **Framework Keyword Density**: Percentage of framework-specific terms
2. **Cross-Framework Influence**: Detection of reasoning from other frameworks
3. **Decision Consistency**: Consistency of decisions across similar scenarios
4. **Reasoning Structure**: Analysis of how agents structure their ethical reasoning
5. **Framework Distinctiveness**: Quantitative measure of how distinct each agent's approach is

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


## Research Questions to Investigate

1. How consistently do the agents maintain their assigned frameworks across different scenario types?
2. Which frameworks provide more decisive ethical guidance versus more nuanced/contextual responses?
3. Are there scenarios where certain frameworks consistently fail to provide clear guidance?
4. How do we measure and maximize the "distinctiveness" of each agent's perspective?
5. Can we identify patterns of reasoning that are unique to each framework?
6. How does scenario framing affect agent responses within their frameworks

## Resources

- Oumi documentation: https://oumi.ai/docs
- Unsloth for distilled and quantized models: https://github.com/unslothai/unsloth
- DeepSeek Model documentation: https://github.com/deepseek-ai/DeepSeek-R1
- Natural Language Toolkit (NLTK): https://www.nltk.org/ (for text analysis)
- spaCy: https://spacy.io/ (for NLP and text processing)
- Matplotlib and Seaborn: For creating visualizations of agent responses