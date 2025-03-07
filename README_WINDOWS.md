

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
   src\windows\install_windows.bat
   ```
   
   This script will:
   - Create a virtual environment
   - Install PyTorch and other dependencies
   - Download a compatible model (GPT-2)
   - Configure your project for Windows compatibility

### Running Tests
To test a single agent:
```
python src\test_agent_oumi.py --config configs\agent_test_config.yaml --agent effective_altruism --scenario trolley
```

To test multiple agents:
```
python src\test_multiagent_oumi.py --config configs\agent_test_config.yaml --scenario trolley
```

### Windows Limitations
The Windows version uses smaller models like GPT-2 instead of the larger GGUF models used on Mac/Linux. This means:
- Responses will be less sophisticated
- Some advanced capabilities may be limited
- Results may not be identical across platforms

For research purposes, we recommend conducting final experiments on Mac/Linux for the best results.


