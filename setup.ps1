# Define the repository URL
$repo_url = "https://github.com/silarsis/assistant.git"

# Define the directory name
$dir_name = "assistant"

# Define the virtual environment name
$venv_name = ".venv"

# Define the requirements file path
$requirements_file = "agent/requirements.txt"

# Define the python script to run
$python_script = "./client_gradio.py"

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

# Clone the repository if it doesn't exist
if (-Not (Test-Path -Path $dir_name -PathType Container) -and ($current_dir_name -ne $dir_name)) {
    try {
        git clone $repo_url
    } catch {
        Write-Host "Failed to clone the repository. Please check the URL and try again."
        exit 1
    }
}

# Change to the directory
if ($current_dir_name -ne $dir_name) {
    Set-Location -Path $dir_name
}

# Create the virtual environment
if (-Not (Test-Path -Path $venv_name -PathType Container)) {
    try {
        python -m venv create $venv_name
    } catch {
        Write-Host "Failed to create the virtual environment. Please check your Python installation and try again."
        exit 1
    }
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

Write-Host "The assistant is now ready to be used."
Write-Host "To start the assistant, run the following command:"
Write-Host "python $python_script"