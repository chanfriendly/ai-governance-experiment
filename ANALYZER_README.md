# Agent Response Analyzer

This tool is designed to analyze how your AI agents with different philosophical frameworks respond to ethical scenarios. It can help you visualize patterns, measure alignment between agents, and evaluate how well each agent adheres to its framework.

## Features

- **Framework Adherence Analysis**: Measures how well each agent sticks to its designated philosophical framework
- **Alignment Measurement**: Calculates how often agents agree with each other
- **Political Compass Plotting**: Maps responses on progressive/conservative and authoritarian/distributed power axes
- **Key Concept Extraction**: Identifies the main themes and ideas in each agent's response
- **Comprehensive Visualization**: Generates charts, heatmaps, and reports to help understand agent behavior

## Installation

1. Make sure you have Python 3.8+ installed
2. Run the setup script to install dependencies:

```bash
python setup_analyzer.py
```

This will install all required packages (numpy, pandas, matplotlib, scikit-learn, nltk, spacy, etc.)

## Quick Start

To analyze all agent responses for all scenarios:

```bash
python analyze_responses.py --report
```

This will:
1. Load all result files from the `results` directory
2. Analyze agent responses across all scenarios
3. Generate visualizations and a comprehensive HTML report in the `analysis_results` directory

## Usage Examples

### Analyze a specific scenario

```bash
python analyze_responses.py --scenario trolley --report
```

### Generate separate analyses for each scenario

```bash
python analyze_responses.py --all_scenarios --output_dir scenario_analyses
```

### Only generate the report without individual plot files

```bash
python analyze_responses.py --report --no_plots
```

## Understanding the Visualizations

### Political Compass

![Political Compass Example](analysis_results/political_compass.png)

This plot shows where each agent's responses fall on two axes:
- **X-axis**: Progressive (positive) vs. Conservative (negative)
- **Y-axis**: Distributed Power (positive) vs. Authoritarian (negative)

The chart helps you see which agents tend toward which political orientations in their reasoning. Clusters indicate agents with similar approaches to governance.

### Alignment Matrix

![Alignment Matrix Example](analysis_results/alignment_matrix.png)

This heatmap shows how similar each agent's responses are to others:
- Darker colors indicate stronger alignment between agents
- The diagonal is always 1.0 (perfect alignment with self)
- Values represent cosine similarity between responses

This visualization helps identify which philosophical frameworks tend to produce similar recommendations despite different reasoning.

### Framework Adherence

![Framework Adherence Example](analysis_results/framework_adherence.png)

This bar chart shows how well each agent adheres to its assigned framework:
- Higher percentages indicate stronger adherence to framework-specific reasoning
- The score represents the percentage of framework-specific concepts used in responses

This helps you evaluate whether your prompts are successfully producing specialized agents with distinct reasoning approaches.

### Cross-Framework Influence

![Cross-Framework Influence Example](analysis_results/cross_framework_influence.png)

This heatmap shows how much each framework influences others:
- The value in each cell represents how much the column framework influences the row framework
- Higher values indicate stronger influence

This helps identify when agents are borrowing concepts or reasoning from other frameworks.

### Decision Distribution

![Decision Distribution Example](analysis_results/decision_distribution.png)

This chart shows the distribution of decisions across different agents:
- Each bar represents an agent's decision for a particular scenario
- Different colors represent different decision categories

This helps analyze whether different frameworks lead to different decisions despite considering the same scenario.

## Customizing the Analyzer

You can customize the analyzer by modifying `agent_analyzer.py`:

- **Add new frameworks**: Update the `self.frameworks` and `self.framework_keywords` dictionaries
- **Adjust political spectrum terms**: Modify the progressive/conservative and authoritarian/distributed power term lists
- **Customize visualizations**: Modify the plotting functions to change colors, styles, or layouts

## Troubleshooting

If you encounter any issues:

1. **Missing dependencies**: Run `python setup_analyzer.py` again
2. **No results found**: Check that your results are in the expected format and location
3. **Visualization errors**: Ensure you have the latest matplotlib and seaborn versions
4. **NLP errors**: Make sure NLTK and spaCy models are properly installed

## Advanced Usage

### Adding Custom Metrics

You can extend the analyzer with custom metrics by adding new methods to the `AgentResponseAnalyzer` class. For example, to add a metric for emotional language:

```python
def analyze_emotional_language(self, text):
    """Analyze emotional content in text"""
    # Implementation here
    return emotion_score
```

Then call this method in the `run_analysis` function and add the results to the DataFrame.

### Batch Processing

For analyzing many scenarios, you can create a batch script:

```bash
#!/bin/bash
for scenario in trolley resource_allocation content_moderation; do
    python analyze_responses.py --scenario $scenario --report
done
```

### Exporting Data for Further Analysis

The analyzer creates a pandas DataFrame with analysis results. You can export this for further analysis:

```python
from agent_analyzer import AgentResponseAnalyzer

analyzer = AgentResponseAnalyzer()
analyzer.load_results()
analyzer.run_analysis()

# Export to CSV
analyzer.results_df.to_csv("analysis_data.csv")
```

## Future Improvements

Future versions of this tool could include:
- Interactive web dashboard for exploring results
- Machine learning clustering to find patterns across many scenarios
- Temporal analysis to track how agent responses evolve over time
- Natural language generation to summarize differences between frameworks