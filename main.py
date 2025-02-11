import os
import json
import pathlib
from datetime import datetime
from baseline import run as run_baseline
from noised import run_noised


def setup(root_path: pathlib.Path, config_path: str):
    # Create the backup directory and move all results files there
    os.makedirs(os.path.join(root_path, "backup"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "results"), exist_ok=True)
    results_path = os.path.join(root_path, "results")
    backup_path = os.path.join(root_path, "backup")
    # move all files from results to back up directory with timestamp
    for file in os.listdir(results_path):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        backup_dir = os.path.join(backup_path, timestamp)
        os.makedirs(backup_dir, exist_ok=True)
        old_path = os.path.join(results_path, file)
        new_path = os.path.join(backup_dir, file)
        os.rename(old_path, new_path)

    # load config file and get environments
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create subfolders for each environment
    environments = [env["name"] for env in config["environments"]]
    print("Environments in config:", environments)
    for environment in environments:
        os.makedirs(os.path.join(root_path, "results", environment, "pcmci"), exist_ok=True)
        os.makedirs(os.path.join(root_path, "results", environment, "plots"), exist_ok=True)
        os.makedirs(os.path.join(root_path, "results", environment, "plots", "baseline"), exist_ok=True)
        os.makedirs(os.path.join(root_path, "results", environment, "csv"), exist_ok=True)
        os.makedirs(os.path.join(root_path, "results", environment, "models"), exist_ok=True)


def main(root_path: pathlib.Path):
    config_path = os.path.join(root_path, "config.json")
    setup(root_path, config_path)

    print("Hello World!")

    # Load config, loop over each environment in "environments", and run baseline + noised
    with open(config_path, "r") as f:
        config = json.load(f)

    for env_item in config["environments"]:
        env_name = env_item["name"]
        print(f"\n===== RUNNING BASELINE FOR ENV={env_name} =====")
        baseline_seed = run_baseline(config_path=config_path, env_name=env_name)

        print(f"===== RUNNING NOISED FOR ENV={env_name} =====")
        run_noised(config_path=config_path, env_name=env_name, baseline_seed=baseline_seed)


if __name__ == '__main__':
    main(root_path=pathlib.Path().absolute())