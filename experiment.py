import os
import subprocess
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def run_stage(stage_name, stage_details):
    """
    Run a specific stage of the pipeline.
    """
    logging.info(f"Running stage: {stage_name}")

    # Set the pipeline directory to the current working directory
    pipeline_dir = os.getcwd()
    logging.info(f"Working directory set to: {pipeline_dir}")

    # Change the working directory to the pipeline directory
    os.chdir(pipeline_dir)

    # Check for dependencies
    dependencies = stage_details.get("dependencies", [])
    for dep in dependencies:
        if not os.path.exists(dep):
            raise FileNotFoundError(f"Dependency {dep} not found for stage {stage_name}")

    # Load the config file
    config_files = stage_details.get("config", [])
    for config_file in config_files:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found for stage {stage_name}")
        logging.info(f"Using config file: {config_file}")

    # Execute the scripts
    scripts = stage_details.get("script", [])
    for script in scripts:
        if not os.path.exists(script):
            raise FileNotFoundError(f"Script {script} not found for stage {stage_name}")
        logging.info(f"Executing script: {script}")
        subprocess.run(["python3", script], check=True)

    # Verify outputs
    outputs = stage_details.get("output", [])
    for output in outputs:
        if not os.path.exists(output):
            raise FileNotFoundError(f"Expected output {output} not found for stage {stage_name}")
        logging.info(f"Output verified: {output}")

def main():
    # Load the pipeline configuration
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Pipeline configuration file {config_file} not found")
    
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Process each stage
    for stage_name in config["stages"]:
        stage_details = config.get(stage_name)
        if not stage_details:
            raise ValueError(f"No details found for stage {stage_name}")
        run_stage(stage_name, stage_details)

if __name__ == "__main__":
    main()
