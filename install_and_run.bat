@echo off
echo Setting up Face Recognition Application...

# Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8+ from python.org
    pause
    exit /b
)

# Create virtual environment
python -m venv face_recognition_env
call face_recognition_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install PyInstaller for creating executable
pip install pyinstaller

# Create executable
pyinstaller --onefile --windowed face_recognition_app.py
