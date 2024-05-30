#!/bin/bash

# Define the repository URL
repo_url="https://github.com/silarsis/assistant.git"

# Define the directory name
dir_name="assistant"

# Define the virtual environment name
venv_name=".venv"

# Define the requirements file path
requirements_file="agent/requirements.txt"

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

# Create the virtual environment if it doesn't exist
if [ ! -d "$venv_name" ]; then
    if ! python -m venv $venv_name; then
        echo "Failed to create the virtual environment. Please check your Python installation and try again."
        exit 1
    fi
fi

# Activate the virtual environment
source $venv_name/bin/activate

# Check if pip is installed
if ! pip --version > /dev/null 2>&1; then
    echo "Pip is not installed. Please install it and try again."
    exit 1
fi

# Install the requirements if they are not already installed
if ! pip freeze | grep -q -f $requirements_file; then
    if ! python -m pip install -U --upgrade-strategy eager --force-reinstall -r $requirements_file; then
        echo "Failed to install the requirements. Please check the requirements file and try again."
        exit 1
    fi
fi

# Change to the agent directory
cd agent || exit

echo "Now, run the following command to start the agent:"
echo "python $python_script"
