## Windows Installation

### Prerequisites
- Python 3.9+ (64-bit version recommended)
- Git for Windows
- At least 4GB of free disk space (more if using larger models)

### Quick Installation
1. Clone the repository:
   ```
   git clone https://github.com/chanfriendly/ai-governance-experiment.git
   cd ai-governance-experiment
   ```

2. Run the Windows installation script:
   ```
   src\windows\setup_windows.bat
   ```
   
   This script will:
   - Create a virtual environment
   - Install PyTorch and other dependencies
   - Download a small model (Phi-2) for intial testing.
   - Configure your project for Windows compatibility

To use larger models on Windows:
1. Download a compatible model (e.g., Phi-2.Q4_K_M.gguf) to your models folder
2. Update the model path in the configuration file
3. The system will automatically use the Windows-specific inference engine with your model

### Running Tests
To test a single agent:
```
python src\test_agent_oumi.py --config configs\agent_test_config.yaml --agent effective_altruism --scenario trolley
```

To test multiple agents:
```
python src\test_multiagent_oumi.py --config configs\agent_test_config.yaml --scenario trolley
```

You can add additional scenarios to /data/scenarios. Be sure to add the scenario to test_agent_oumi.py and test_multiagent_oumi.py by updating their argument parsers.

### Results and Analysis

When you run tests, the system automatically creates a folder in the `/results` directory for each test run. These folders are named with the scenario, date, and timestamp.

Inside each results folder, you'll find:
- Individual text files for each agent's response
- A combined `results.json` file with all responses for multi-agent tests
- A `scenario.txt` file with the original scenario and test configuration

To analyze agent responses, you can run the analysis tools. Please see the ANALYZER_README.md for more detailed instructions on generating visualizations.

This will generate visualizations and analysis reports in the `analysis_results` directory, including:
- HTML reports comparing agent responses
- Charts showing alignment with different philosophical frameworks
- Text similarity analyses between agent responses



