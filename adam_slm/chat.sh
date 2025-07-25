#!/bin/bash
# A.D.A.M. SLM Chat Interface Launcher for Unix/Linux

echo "🚀 Starting A.D.A.M. SLM Chat Interface..."
echo

# Change to the script directory
cd "$(dirname "$0")"

# Run the chat interface
python3 chat_interface.py

# Check exit status
if [ $? -ne 0 ]; then
    echo
    echo "❌ Chat interface exited with an error."
    read -p "Press Enter to continue..."
fi

echo
echo "👋 Thanks for using A.D.A.M. SLM!"
