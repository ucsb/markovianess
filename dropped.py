#!/usr/bin/env python
"""
dropped.py

Implements a pipeline for dropping one dimension of the observation (DropDimensionWrapper).
For each dimension:
 1) Remove that dimension from the observation
 2) Train a new RPPO model
 3) Gather multiple rollouts for PCMCI (Fisher's method) => Markov scores
 4) Plot:
    - Combined figure per dimension vs. baseline
    - Rewards vs. dimension
    - Rewards vs. Markovian score
    - Etc.

Outputs go in results/<ENV>/dropped for each environment.
"""

import os
import json
import time
import random
import argparse
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2

from rppo import RPPO
from conditional_independence_test import (
    ConditionalIndependenceTest,
    get_markov_violation_score
)

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - dropped: %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),   # Writes logs to file
        logging.StreamHandler()            # Writes logs to console (stdout)
    ]
)

###############################################################################
# Reward Tracking Callback
###############################################################################
class RewardTrackingCallback(BaseCallback):
    """
    Records total reward per episode for a single-environment scenario.
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        return True

    def get_rewards(self):
        return self.episode_rewards


###############################################################################
# DropDimensionWrapper
###############################################################################
class DropDimensionWrapper(gym.Wrapper):
    """
    A gym wrapper that removes one dimension from the observation space (1D Box).
    E.g. if shape=(4,) and drop_dim_index=2, new shape=(3,).
    """
    def __init__(self, env, drop_dim_index):
        super().__init__(env)
        self.drop_dim = drop_dim_index

        original_obs_space = env.observation_space
        if not isinstance(original_obs_space, gym.spaces.Box):
            raise NotImplementedError("DropDimensionWrapper only supports 1D Box spaces.")
        if len(original_obs_space.shape) != 1:
            raise ValueError(f"Expected 1D obs shape, got {original_obs_space.shape}.")

        obs_dim = original_obs_space.shape[0]
        if not (0 <= drop_dim_index < obs_dim):
            raise ValueError(f"Invalid drop_dim_index={drop_dim_index} for obs_dim={obs_dim}.")

        new_shape = (obs_dim - 1,)

        low = np.delete(original_obs_space.low, drop_dim_index)
        high = np.delete(original_obs_space.high, drop_dim_index)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=original_obs_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.delete(obs, self.drop_dim)
        return obs, info

    def step(self, action):
        output = self.env.step(action)
        # Gymnasium returns 5 items; older Gym returns 4
        if len(output) == 5:
            obs, reward, done, truncated, info = output
        else:
            obs, reward, done, info = output
            truncated = done
        obs = np.delete(obs, self.drop_dim)
        return obs, reward, done, truncated, info


def make_dropped_env(env_name, drop_dim_index, seed=None):
    """
    Single-env vectorized environment dropping one dimension from the observation.
    """
    def _env_fn():
        e = gym.make(env_name)
        if seed is not None:
            e.reset(seed=seed)
        return DropDimensionWrapper(e, drop_dim_index)
    return make_vec_env(_env_fn, n_envs=1, seed=seed)


###############################################################################
# Utility: Smoothing
###############################################################################
def _smooth_reward_curve(episodes, rewards, window=10):
    if len(rewards) < window:
        return episodes, rewards
    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    return episodes[window - 1:], smoothed

###############################################################################
# Single figure for each dropped dimension (overlaid with baseline)
###############################################################################
def plot_dropped_learning_curves_all_dims(
    df_rewards: pd.DataFrame,
    baseline_csv: str = None,
    output_dir: str = ".",
    smooth_window: int = 10
):
    """
    Produces a single figure for the environment that overlays each dropped
    dimension's learning curve on the same plot. Baseline is added in black
    if found. This parallels the multi-line approach in noised.py.

    The DataFrame should have columns:
      ["Environment", "DroppedDim", "Episode", "Reward"]
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_rewards.empty:
        logging.info("[DroppedObs] No dropped data to plot.")
        return

    env_name = df_rewards["Environment"].iloc[0]

    # Attempt to load baseline if provided
    baseline_df = None
    if baseline_csv and os.path.isfile(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        bdf = bdf[bdf["Environment"] == env_name]
        if not bdf.empty:
            baseline_df = bdf

    plt.figure(figsize=(8, 5))

    # Plot each dropped dimension
    for dropped_dim_id, df_dim in df_rewards.groupby("DroppedDim"):
        df_dim = df_dim.sort_values("Episode")
        episodes = df_dim["Episode"].values
        rewards = df_dim["Reward"].values

        x_smooth, y_smooth = _smooth_reward_curve(episodes, rewards, window=smooth_window)
        label_str = f"Dropped Dim={dropped_dim_id}"
        plt.plot(x_smooth, y_smooth, linewidth=2, label=label_str)

    # Overlay baseline if found
    if baseline_df is not None:
        base_sorted = baseline_df.sort_values("Episode")
        bx = base_sorted["Episode"].values
        # typical baseline column might be "TotalReward"
        by = base_sorted["TotalReward"].values
        bx_smooth, by_smooth = _smooth_reward_curve(bx, by, window=smooth_window)
        plt.plot(bx_smooth, by_smooth, color="black", linewidth=3, label="Baseline")

    plt.title(f"{env_name}: All Dropped Dimensions (Overlaid)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    out_fname = f"{env_name}_all_dropped_dims_overlaid.png"
    out_path = os.path.join(output_dir, out_fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"[DroppedObs] Combined multi-dim dropped plot saved => {out_path}")

###############################################################################
# Multi-run PCMCI with Fisher's method
###############################################################################
class DroppedObservationsExperiments:
    def __init__(self, config, root_path="."):
        """
        config: must have "environments" (list of env configs) and possibly other fields.
        root_path: The base folder with 'results' subfolder, etc.
        """
        self.config = config
        self.root_path = root_path
        self.reward_records = []
        self.markov_records = []

    def fishers_method(self, pvals, epsilon=1e-15):
        """
        Combine multiple p-values using Fisher's method (like noised.py).
        """
        pvals = np.array(pvals, dtype=float)
        pvals = np.clip(pvals, epsilon, 1 - epsilon)
        statistic = -2.0 * np.sum(np.log(pvals))
        df = 2 * len(pvals)
        return 1.0 - chi2.cdf(statistic, df)

    def gather_and_run_pcmci(self, model, env_name, drop_dim_index, steps=2000, seed=None):
        """
        Creates a dropped-dim environment, collects `steps` observations using `model`,
        then runs PCMCI and returns (val_matrix, p_matrix).
        """
        test_env = make_dropped_env(env_name, drop_dim_index=drop_dim_index, seed=seed)

        obs = test_env.reset()
        obs_list = []
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            output = test_env.step(action)
            if len(output) == 5:
                obs, reward, done, truncated, info = output
            else:
                obs, reward, done, info = output
                truncated = done
            obs_list.append(obs[0])  # shape = (1, obs_dim-1)
            if done[0] or truncated[0]:
                obs = test_env.reset()
        test_env.close()

        obs_array = np.array(obs_list)

        # Run PCMCI
        cit = ConditionalIndependenceTest()
        results_dict = cit.run_pcmci(
            observations=obs_array,
            tau_min=1,
            tau_max=2,
            alpha_level=0.05,
            pc_alpha=None,
            env_id=env_name,
            label=f"dropped_dim_{drop_dim_index}_seed_{seed}",
            results_dir=os.path.join(self.root_path, "results", env_name, "dropped", "pcmci")
        )
        return results_dict["val_matrix"], results_dict["p_matrix"]

    def run_multiple_pcmci_fisher(
        self, model, env_name, drop_dim_index, num_runs=5, steps=2000
    ):
        """
        Collect multiple rollouts (num_runs) for PCMCI, combine p-values via Fisher
        and average the val_matrix. Return the Markov violation score.
        """
        val_list = []
        p_list = []
        for _ in range(num_runs):
            seed = random.randint(0, 9999)
            val_m, p_m = self.gather_and_run_pcmci(
                model, env_name, drop_dim_index,
                steps=steps, seed=seed
            )
            val_list.append(val_m)
            p_list.append(p_m)

        val_arr = np.stack(val_list, axis=0)  # shape=(num_runs, N, N, L)
        p_arr   = np.stack(p_list, axis=0)   # shape=(num_runs, N, N, L)

        avg_val_matrix = np.mean(val_arr, axis=0)

        n_runs, N, _, L = p_arr.shape
        combined_p_matrix = np.zeros((N, N, L), dtype=float)
        for i in range(N):
            for j in range(N):
                for k in range(L):
                    pvals_for_link = p_arr[:, i, j, k]
                    combined_p_matrix[i, j, k] = self.fishers_method(pvals_for_link)

        # Compute Markov violation
        mk_score = get_markov_violation_score(
            p_matrix=combined_p_matrix,
            val_matrix=avg_val_matrix,
            alpha_level=0.05
        )
        return mk_score

    def run(self, env_name=None, baseline_seed=None):
        """
        If env_name is provided, run only for that environment.
        Otherwise, run for all in the config.

        The results + Markov analysis are saved in results/<ENV>/dropped.
        """
        envs = self.config["environments"]
        if env_name:
            envs = [e for e in envs if e["name"] == env_name]
        if not envs:
            logging.warning(f"[DroppedObs] No matching environment for {env_name}")
            return

        for env_item in envs:
            name = env_item["name"]
            time_steps = env_item["time_steps"]
            n_envs = env_item["n_envs"]
            obs_names = env_item.get("observations", [])
            obs_dim_count = len(obs_names)

            # Folders
            env_path = os.path.join(self.root_path, "results", name)
            dropped_path = os.path.join(env_path, "dropped")
            pcmci_path = os.path.join(dropped_path, "pcmci")
            os.makedirs(dropped_path, exist_ok=True)
            os.makedirs(pcmci_path, exist_ok=True)

            logging.info(f"[DroppedObs] Start environment: {name}, obs_dim_count={obs_dim_count}")

            # For each dimension to drop
            for dim_id in range(obs_dim_count):
                used_seed = baseline_seed if baseline_seed is not None else random.randint(0, 9999)
                logging.info(f"[DroppedObs] -> Dropping dimension {dim_id}, seed={used_seed}")

                # 1) Make environment without that dimension
                venv = make_dropped_env(name, drop_dim_index=dim_id, seed=used_seed)

                # 2) Train RL
                model = RPPO("MlpPolicy", venv, verbose=0, learning_rate=3e-4)
                callback = RewardTrackingCallback()
                model.learn(total_timesteps=time_steps, callback=callback)
                ep_rewards = callback.get_rewards()
                venv.close()

                # Store reward curve
                for i, rew_val in enumerate(ep_rewards):
                    self.reward_records.append({
                        "Environment": name,
                        "DroppedDim": dim_id,
                        "Episode": i+1,
                        "Reward": rew_val
                    })

                # 3) MULTIPLE PCMCI runs => Fisher
                NUM_PCMCI_RUNS = 15  # or 10, etc.
                mk_score = self.run_multiple_pcmci_fisher(
                    model=model,
                    env_name=name,
                    drop_dim_index=dim_id,
                    num_runs=NUM_PCMCI_RUNS,
                    steps=2000
                )
                logging.info(f"[DroppedObs] Markov Score => {mk_score:.4f} (Fisher from {NUM_PCMCI_RUNS} runs)")

                # 4) Final metrics
                final_rew = np.mean(ep_rewards[-10:]) if len(ep_rewards) > 10 else np.mean(ep_rewards)
                self.markov_records.append({
                    "Environment": name,
                    "DroppedDim": dim_id,
                    "MarkovScore": mk_score,
                    "MeanFinalReward": final_rew
                })

            # Done dropping each dimension => Save CSV & produce plots
            env_rewards = [r for r in self.reward_records if r["Environment"] == name]
            env_markov = [m for m in self.markov_records if m["Environment"] == name]

            df_rewards = pd.DataFrame(env_rewards)
            df_markov = pd.DataFrame(env_markov)

            dropped_rewards_csv = os.path.join(dropped_path, "dropped_rewards.csv")
            dropped_markov_csv = os.path.join(dropped_path, "dropped_markov.csv")
            df_rewards.to_csv(dropped_rewards_csv, index=False)
            df_markov.to_csv(dropped_markov_csv, index=False)

            logging.info(f"[DroppedObs] Wrote dropped rewards => {dropped_rewards_csv}")
            logging.info(f"[DroppedObs] Wrote dropped Markov => {dropped_markov_csv}")

            # Plot the results
            baseline_csv_path = os.path.join(env_path, "csv", "baseline_learning_curve.csv")
            plot_dropped_learning_curves_all_dims(
                df_rewards,
                baseline_csv=baseline_csv_path,
                output_dir=dropped_path,
                smooth_window=10
            )
            # You can add further dimension vs. Markov plots if desired.

###############################################################################
# Main entry points
###############################################################################
def run_dropped(config_path="config.json", env_name=None, baseline_seed=None):
    """
    We reuse the baseline_seed if given, to ensure environment initialization is consistent.
    """
    if not os.path.exists(config_path):
        logging.error(f"Config file '{config_path}' not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    root_path = os.path.abspath(".")
    runner = DroppedObservationsExperiments(config, root_path=root_path)

    start_t = time.perf_counter()
    runner.run(env_name=env_name, baseline_seed=baseline_seed)
    end_t = time.perf_counter()
    logging.info(f"[DroppedObs] Done! Total time: {(end_t - start_t):.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Dropped Observations Experiments")
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Name of the environment to drop dims from.")
    args = parser.parse_args()

    run_dropped(config_path=args.config_path, env_name=args.env)

if __name__ == "__main__":
    main()