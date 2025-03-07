# setup_windows_simple.ps1
Write-Host "Setting up AI Governance Experiment for Windows (Simplified)..." -ForegroundColor Green

# Create and activate virtual environment
python -m venv venv
& .\venv\Scripts\Activate.ps1

# Install Windows-friendly dependencies
pip install transformers accelerate ctransformers numpy tqdm pyyaml pandas matplotlib seaborn

# Create directories
if (-not (Test-Path -Path "models")) { New-Item -Path "models" -ItemType Directory }
if (-not (Test-Path -Path "results")) { New-Item -Path "results" -ItemType Directory }

# Download a test model
Write-Host "Downloading test model (this may take some time)..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf" -OutFile "models\phi-2.Q4_K_M.gguf"

Write-Host "Setup complete!" -ForegroundColor Green