import subprocess
import sys
import os
import shutil
from console import TPConsole, Colors


def run_command(command: list[str], description: str) -> bool:
    """Runs a shell command using subprocess and updates Rich progress."""
    console = TPConsole()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # You might want to stream stdout/stderr to the console or a log file for more detail
        stdout, stderr = process.communicate()
        console.update_progress_task("subprocess", advance=1)

        if process.returncode != 0:
            console.print(Colors.error(f"Error executing '{' '.join(command)}':"))
            console.print(f"[{Colors.MED_GREY}]{stderr}[/{Colors.MED_GREY}]")
            console.update_progress_task("subprocess", description=f"[{Colors.RED}]Failed:[/{Colors.RED}] {description}")
            return False
        else:
            # console.print(f"[dim]{stdout}[/dim]") # Optionally print stdout
            return True
    except Exception as e:
        console.print(Colors.error(f"Exception during '{' '.join(command)}': {e}"))
        return False

def run_bootstrap():
    """Performs the environment bootstrap process."""
    # Define the steps
    steps = [
        {"cmd": ["apt-get", "update"], "desc": "Update package lists", "step": "apt-get"},
        {"cmd": ["apt-get", "install", "-y", "git", "build-essential"], "desc": "Install git & build-essential", "step": "apt-get"},
        #{"cmd": ["git", "clone", "https://github.com/deepseek-ai/FlashMLA.git"], "desc": "Clone FlashMLA repository", "step": "FlashMLA"},
        # Install FlashMLA (requires running from within the directory)
        #{"cmd": [sys.executable, "setup.py", "install"], "desc": "Install FlashMLA", "cwd": "FlashMLA", "step": "FlashMLA"},
        # Create project directories
        {"func": lambda: os.makedirs("dataset_cache", exist_ok=True), "desc": "Create dataset_cache dir", "step": "Directories"},
        {"func": lambda: os.makedirs("checkpoints", exist_ok=True), "desc": "Create checkpoints dir", "step": "Directories"},
        {"func": lambda: os.makedirs("models", exist_ok=True), "desc": "Create models dir", "step": "Directories"},
        # Install project dependencies (Example - Adapt!)
        {"cmd": [sys.executable, "-m", "pip", "install", "-e", "."], "desc": "Install TabulaPrima dependencies", "step": "Dependencies"},
        # Optional dependencies (Example - Adapt!)
        # {"cmd": [sys.executable, "-m", "pip", "install", "tensorboard", "bitsandbytes"], "desc": "Install optional dependencies"},
    ]

    total_steps = len(steps)
    console = TPConsole()
    TPConsole().create_progress_task("bootstrapping", "Bootstrap", total_steps)
    TPConsole().rule(Colors.highlight("Checking Prerequisites"))

    # Basic Prerequisite Checks (can be expanded)
    if not shutil.which("python"):
        console.print(Colors.error("Error: Python not found in PATH."))
        return False
    if not shutil.which("nvcc"):
        console.print(Colors.warning("Warning: nvcc not found. FlashMLA installation might fail if CUDA is not properly set up."))
        # Decide whether to exit or proceed cautiously
        # return False

    console.print(Colors.success("✓ Prerequisites seem okay."))

    console.rule(Colors.highlight("Executing Bootstrap Steps"))
    # Execute steps
    for step in steps:
        console.update_progress_task("bootstrapping", description=f"Bootstrap [{step['step']}]")
        success = False
        original_cwd = os.getcwd()
        if "cmd" in step:
            cwd = step.get("cwd")
            if cwd:
                try:
                    os.chdir(cwd)
                    console.print(f"[dim]Changed directory to {os.getcwd()}[/dim]")
                except Exception as e:
                    console.print(Colors.error(f"Error changing directory to '{cwd}': {e}"))
                    console.remove_progress_task("bootstrapping")
                    return False # Stop if we can't change directory

            console.create_progress_task("subprocess", step['desc'])
            success = run_command(step["cmd"], step["desc"])

            if cwd:
                os.chdir(original_cwd) # Change back to original directory
                console.print(f"[dim]Returned to directory {os.getcwd()}[/dim]")

        elif "func" in step:
            try:
                step["func"]()
                success = True
            except Exception as e:
                console.print(Colors.error(f"Error executing function for '{step['desc']}': {e}"))
                success = False

        if not success:
            console.print(Colors.error(f"Bootstrap process failed at step: {step['desc']}."))
            return False # Stop the bootstrap on failure

        console.update_progress_task("bootstrapping", advance=1)

    console.rule(Colors.highlight("Bootstrap Complete"))
    return True