# Define the repository URL
$repo_url = "https://github.com/silarsis/assistant.git"

# Define the directory name
$dir_name = "assistant"

# Define the virtual environment name
$venv_name = ".venv"

# Define the requirements file path
$requirements_file = "agent/requirements.txt"

# Define the python script to run
$python_script = "agent/client_gradio.py"

# Check if git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is not installed. Please install it and try again."
    exit 1
}

# Check if python is installed
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed. Please install it and try again."
    exit 1
}

# Clone the repository
try {
    git clone $repo_url
} catch {
    Write-Host "Failed to clone the repository. Please check the URL and try again."
    exit 1
}

# Change to the directory
Set-Location -Path $dir_name

# Create the virtual environment
try {
    python -m venv create $venv_name
} catch {
    Write-Host "Failed to create the virtual environment. Please check your Python installation and try again."
    exit 1
}

# Activate the virtual environment
. $venv_name/bin/activate

# Check if pip is installed
if (!(Get-Command pip -ErrorAction SilentlyContinue)) {
    Write-Host "Pip is not installed. Please install it and try again."
    exit 1
}

# Install the requirements
try {
    python -m pip install -U --upgrade-strategy eager --force-reinstall -r $requirements_file
} catch {
    Write-Host "Failed to install the requirements. Please check the requirements file and try again."
    exit 1
}

# Change to the agent directory
Set-Location -Path agent

# Run the python script
try {
    python $python_script
} catch {
    Write-Host "Failed to run the Python script. Please check the script and try again."
    exit 1
}