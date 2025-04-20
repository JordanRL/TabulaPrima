#!/bin/sh
# entrypoint.sh - Script runs when container starts

# Exit immediately if a command exits with a non-zero status.
set -e

TARGET_DIR="/workspace/TabulaPrima" # Define where the code should live

echo "--- Container Startup Script ---"

# Check if the target directory exists and has a .git folder
if [ -d "$TARGET_DIR/.git" ]; then
    echo "Repository found in $TARGET_DIR. Pulling latest changes..."
    # Change directory *before* attempting git operations within it
    cd "$TARGET_DIR"
    # Capture potential errors from git pull
    if ! git pull origin master; then
        echo "Git pull failed." >&2
        exit 1
    fi
else
    echo "Repository not found. Cloning into $TARGET_DIR..."
    # Clone the specific branch you want, redirect stderr to stdout for logging
    # Added error checking directly
    if ! git clone --branch master --single-branch https://github.com/JordanRL/TabulaPrima.git "$TARGET_DIR" 2>&1; then
        echo "Git clone failed." >&2
        exit 1
    fi
    # Change directory *after* successful clone
    cd "$TARGET_DIR"
fi

echo "Code is up-to-date in $TARGET_DIR."
echo "Executing command: $@" # Log what command is about to be run

# Replace the current script process with the command provided in CMD
exec "$@"