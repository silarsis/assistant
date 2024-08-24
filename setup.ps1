# Define the repository URL
$repo_url = "https://github.com/silarsis/assistant.git"

# Define the directory name
$dir_name = "assistant"

# Define the python script to run
$python_script = "./client_gradio.py"

# Check the current dir name in case we're already in the repo
$current_dir_name = Split-Path -Leaf (Get-Location)


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

# Check if poetry is installed
if (!(Get-Command poetry -ErrorAction SilentlyContinue)) {
    if (-Not (Get-Command pipx -ErrorAction SilentlyContinue)) {
        if (-Not (Get-Command scoop -ErrorAction SilentlyContinue)) {
            Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
        }
        scoop install pipx
        pipx ensurepath
    }
    pipx install poetry
}

# Change to the agent directory
Set-Location -Path agent

# Install the requirements
poetry install

Write-Host "The assistant is now ready to be used."
Write-Host "To start the assistant, run the following commands:"
Write-Host "poetry run python $python_script"