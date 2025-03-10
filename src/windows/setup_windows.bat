@echo off
echo Installing Windows dependencies for AI Governance Experiment...
echo.

REM Activate venv if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)

echo Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo Installing other dependencies...
pip install -r requirements_win.txt

echo.
echo Loading tiny model as a fallback...
python utils\download_model_win.py --model phi2

echo.
echo Dependencies installed successfully!
echo.
echo To run a test:
echo python src\test_agent_oumi.py --config configs\agent_test_config.yaml --agent effective_altruism --scenario trolley