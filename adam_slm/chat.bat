@echo off
REM A.D.A.M. SLM Chat Interface Launcher for Windows

echo 🚀 Starting A.D.A.M. SLM Chat Interface...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Run the chat interface
python chat_interface.py

REM Pause to see any error messages
if errorlevel 1 (
    echo.
    echo ❌ Chat interface exited with an error.
    pause
)

echo.
echo 👋 Thanks for using A.D.A.M. SLM!
pause
