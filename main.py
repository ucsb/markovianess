import os
import json
import pathlib
import time
from datetime import datetime
from baseline import run as run_baseline
from noised import run_noised
from dropped import run_dropped


def setup(root_path: pathlib.Path, config_path: str):
    # Create the backup directory and move all results files there
    os.makedirs(os.path.join(root_path, "backup"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "results"), exist_ok=True)
    results_path = os.path.join(root_path, "results")
    backup_path = os.path.join(root_path, "backup")
    # move all files from results to back up directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    backup_dir = os.path.join(backup_path, timestamp)
    for file in os.listdir(results_path):
        os.makedirs(backup_dir, exist_ok=True)
        old_path = os.path.join(results_path, file)
        new_path = os.path.join(backup_dir, file)
        os.rename(old_path, new_path)

    # if logs.txt exists, move it to backup directory
    logs_path = os.path.join(root_path, "logs.txt")
    if os.path.exists(logs_path):
        os.rename(logs_path, os.path.join(backup_dir, "logs.txt"))

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

    start_time = time.perf_counter()
    for env_item in config["environments"]:
        env_name = env_item["name"]
        print(f"\n===== RUNNING BASELINE FOR ENV={env_name} =====")
        baseline_seed = run_baseline(config_path=config_path, env_name=env_name)
        b_end_time = time.perf_counter()
        print(f"Baseline time: {b_end_time - start_time}")

        print(f"===== RUNNING NOISED FOR ENV={env_name} =====")
        run_noised(config_path=config_path, env_name=env_name, baseline_seed=baseline_seed)
        n_end_time = time.perf_counter()
        print(f"Noised time: {n_end_time - b_end_time}")


        print(f"===== RUNNING DROPPED FOR ENV={env_name} =====")
        run_dropped(config_path, env_name, baseline_seed)
        d_end_time = time.perf_counter()
        print(f"Dropped time: {d_end_time - n_end_time}")
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time}")


if __name__ == '__main__':
    main(root_path=pathlib.Path().absolute())