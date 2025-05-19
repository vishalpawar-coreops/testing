import os
import subprocess
import time
from dotenv import load_dotenv

# Load GitHub token from .env
load_dotenv()
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise Exception("GITHUB_TOKEN not found in .env file!")

# GitHub repo details
username = "vishalpawar-coreops"
repo = "testing"
repo_url = f"https://{token}@github.com/{username}/{repo}.git"

# Function to run commands with timeout and retry
def run_command_with_retry(cmd, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            print(f"Running command (attempt {attempt+1}/{retries})")
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
            if result.returncode == 0:
                return result
            else:
                print(f"Command failed with exit code {result.returncode}")
                print(f"Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Command timed out after {timeout} seconds")
        
        # Wait before retrying
        if attempt < retries - 1:
            wait_time = 5 * (attempt + 1)  # Exponential backoff
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    raise Exception(f"Command failed after {retries} attempts")

# Test GitHub connectivity
print("Testing GitHub connectivity...")
try:
    subprocess.run(["curl", "-s", "-m", "10", "https://github.com"], check=True)
    print("GitHub is reachable")
except Exception as e:
    print(f"Warning: GitHub connectivity test failed: {e}")

# Check if we're in a git repo already, if not initialize one
if not os.path.exists('.git'):
    print("Initializing git repository...")
    subprocess.run(["git", "init"])
    
    # Configure git user to avoid "please tell me who you are" errors
    subprocess.run(["git", "config", "user.email", "user@example.com"])
    subprocess.run(["git", "config", "user.name", "Automated Push"])

# Remove and re-add remote if it already exists
subprocess.run(["git", "remote", "remove", "origin"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
run_command_with_retry(["git", "remote", "add", "origin", repo_url])

# Make sure on 'main' branch
run_command_with_retry(["git", "checkout", "-B", "main"])

# Only add .py and .yaml files, but check if they exist first
py_files = subprocess.run(["find", ".", "-name", "*.py"], capture_output=True, text=True).stdout.strip()
yaml_files = subprocess.run(["find", ".", "-name", "*.yaml"], capture_output=True, text=True).stdout.strip()

if py_files:
    run_command_with_retry(["git", "add", "*.py"])
if yaml_files:
    run_command_with_retry(["git", "add", "*.yaml"])

# Commit the changes only if there are changes to commit
status_output = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
if status_output.strip():
    try:
        run_command_with_retry(["git", "commit", "-m", "Initial commit: Add only .py and .yaml files"])
    except Exception as e:
        print(f"Commit failed, but continuing: {e}")
else:
    print("No changes to commit")

# Push forcefully to origin/main with increased timeout
print("Pushing to GitHub (this may take a while)...")
run_command_with_retry(["git", "push", "--force", "origin", "main"], timeout=120)

print("GitHub push completed successfully")
