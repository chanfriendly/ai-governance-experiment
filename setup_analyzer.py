#!/usr/bin/env python3
"""
Setup script to install all required dependencies for the agent analyzer tool.
"""

import subprocess
import sys
import os

def check_and_install(package_name, import_name=None):
    """Check if a package is installed and install it if not."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} is already installed")
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} installed successfully")

def main():
    print("Setting up environment for agent analyzer tool...")
    
    # Install required packages
    check_and_install("numpy")
    check_and_install("pandas")
    check_and_install("matplotlib")
    check_and_install("seaborn")
    check_and_install("nltk")
    check_and_install("spacy")
    check_and_install("scikit-learn")
    
    # Make sure required NLTK data is downloaded
    try:
        import nltk
        print("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
    
    # Download spaCy model
    try:
        print("Downloading spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy model downloaded successfully")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        print("You may need to run: python -m spacy download en_core_web_sm")
    
    # Make the analyzer scripts executable
    try:
        os.chmod("agent_analyzer.py", 0o755)
        os.chmod("analyze_responses.py", 0o755)
        print("✓ Made scripts executable")
    except Exception as e:
        print(f"Note: Could not make scripts executable: {e}")
    
    print("\nSetup complete! You can now run the analyzer with: python analyze_responses.py")

if __name__ == "__main__":
    main()