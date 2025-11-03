@echo off
REM Quick Setup Script for Person Tracking System
REM Run this script to set up the system

echo ========================================
echo Person Tracking System - Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python is installed
echo.

REM Install dependencies
echo Installing dependencies...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo [OK] Dependencies installed successfully!
echo.

REM Check if training data exists
if exist "images\" (
    echo [OK] Found images directory
    dir /b /ad images 2>nul | findstr "." >nul
    if errorlevel 1 (
        echo [WARNING] No person folders found in images/
        echo You need to run: python datacollect.py
    ) else (
        echo [OK] Found person folders in images/
    )
) else (
    echo [WARNING] images/ directory not found
    echo Creating images/ directory...
    mkdir images
    echo You need to collect training data: python datacollect.py
)

echo.

REM Check if model is trained
if exist "keras_model.h5" (
    echo [OK] Face recognition model found
) else (
    echo [WARNING] Face recognition model not found
    echo After collecting data, run: python train_face_recog.py
)

if exist "labels.txt" (
    echo [OK] Labels file found
) else (
    echo [WARNING] Labels file not found
)

echo.

REM Check YOLO model
if exist "best.pt" (
    echo [OK] YOLO model found
) else if exist "yolo11n.pt" (
    echo [OK] YOLO model found
) else (
    echo [INFO] YOLO model not found
    echo System will download yolo11n.pt automatically on first run
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Collect training data:  python datacollect.py
echo 2. Train face model:       python train_face_recog.py
echo 3. Run the system:         python integrated_system.py
echo.
echo For detailed guide:        python start_guide.py
echo To view database:          python view_database.py
echo.

REM Ask if user wants to see the guide
set /p SHOW_GUIDE="Show the quick start guide now? (y/n): "
if /i "%SHOW_GUIDE%"=="y" (
    python start_guide.py
)

pause
