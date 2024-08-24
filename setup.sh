#!/bin/bash

# Define the repository URL
repo_url="https://github.com/silarsis/assistant.git"

# Define the directory name
dir_name="assistant"

# Define the python script to run
python_script="./client_gradio.py"

# Check the current dir name in case we're already in the repo
current_dir_name=$(basename "$PWD")

# Check if git is installed
if ! git --version > /dev/null 2>&1; then
    echo "Git is not installed. Please install it and try again."
    exit 1
fi

# Check if python is installed
if ! python --version > /dev/null 2>&1; then
    echo "Python is not installed. Please install it and try again."
    exit 1
fi

# Clone the repository if it doesn't exist
if [ ! -d "$dir_name" ] && [ "$current_dir_name" != "$dir_name" ]; then
    if ! git clone $repo_url; then
        echo "Failed to clone the repository. Please check the URL and try again."
        exit 1
    fi
fi

# Change to the directory
if [ "$current_dir_name" != "$dir_name" ]; then
    cd $dir_name || exit
fi

# Check if poetry is installed
if ! poetry --version > /dev/null 2>&1; then
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Change to the agent directory
cd agent || exit

poetry install

echo "Now, run the following command to start the agent:"
echo "poetry run python $python_script"
