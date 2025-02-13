#!/usr/bin/env python
"""
noised.py

Implements a complete pipeline:
1) Noise injection logic (Gaussian) on a specific observation dimension
2) Train an RPPO model from scratch with that noise
3) Compare learning curves vs. baseline
4) Run PCMCI for each dimension with multiple rollouts + Fisher's method, store Markovian score
5) Plot:
   - One combined figure per noise variance (lines for each dimension) vs. baseline
   - Rewards vs. Noise
   - Rewards vs. Markovian score
   - Correlation between Markov Score & noise

Saves outputs into the same "results" subfolders as main.py for each environment.
"""

import os
import json
import time
import random
import argparse
import logging
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Our custom PPO logic (with partial noise support)
from rppo import RPPO
# PCMCI + Markovian score logic
from conditional_independence_test import ConditionalIndependenceTest, get_markov_violation_score

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - noised: %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),   # Writes logs to file
        logging.StreamHandler()            # Writes logs to console (stdout)
    ]
)

###############################################################################
# Basic callback to track episode rewards
###############################################################################
class RewardTrackingCallback(BaseCallback):
    """
    Tracks the total reward per episode for a single-environment scenario.
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
# Wrapper to inject noise into a single dimension of the observation
###############################################################################
class NoisyObservationWrapper(gym.Wrapper):
    """
    Gym wrapper that adds Gaussian noise to a single dimension of the observation.
    dimension_to_noisify = None => apply to all dims
    """
    def __init__(self, env, dimension_to_noisify=None, mean=0.0, variance=0.01):
        super().__init__(env)
        self.dim_to_noisify = dimension_to_noisify
        self.mean = mean
        self.variance = variance
        self.std = np.sqrt(variance)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # Gymnasium returns 5 items: (obs, reward, done, truncated, info)
        output = self.env.step(action)
        if len(output) == 5:
            obs, reward, done, truncated, info = output
        else:
            obs, reward, done, info = output
            truncated = done  # older gym fallback

        # Inject noise
        if self.dim_to_noisify is None:
            # noise on all dims
            obs = obs + np.random.normal(self.mean, self.std, size=obs.shape)
        else:
            noise = np.random.normal(self.mean, self.std)
            obs = np.array(obs, dtype=np.float32)
            obs[self.dim_to_noisify] += noise

        return obs, reward, done, truncated, info


def make_noisy_env(
    env_name,
    dimension_to_noisify=None,
    mean=0.0,
    variance=0.01,
    seed=None
):
    """
    Creates a vectorized environment that wraps a single environment with NoisyObservationWrapper.
    """
    def _env_fn():
        e = gym.make(env_name)
        if seed is not None:
            e.reset(seed=seed)
        e = NoisyObservationWrapper(
            e,
            dimension_to_noisify=dimension_to_noisify,
            mean=mean,
            variance=variance
        )
        return e

    venv = make_vec_env(_env_fn, n_envs=1, seed=seed)
    return venv


###############################################################################
# Utility: rolling average for smoothing
###############################################################################
def _smooth_reward_curve(episodes, rewards, window=10):
    """
    Returns a rolling mean of the rewards. If <window points, just returns raw data.
    """
    if len(rewards) < window:
        return episodes, rewards
    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    return episodes[window - 1:], smoothed


###############################################################################
# 1. Single plot for each variance (lines for each dimension) + baseline
###############################################################################
def plot_noised_learning_curves_all_dims(
    df_rewards: pd.DataFrame,
    baseline_csv: str = None,
    output_dir: str = ".",
    smooth_window: int = 10
):
    """
    For each noise variance, produce one figure:
      - multiple lines, one per (ObsDim, ObsName),
      - plus black baseline if found.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_rewards.empty:
        logging.info("No noised data to plot.")
        return

    env_name = df_rewards["Environment"].iloc[0]

    # Attempt to load baseline
    baseline_df = None
    if baseline_csv and os.path.isfile(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        baseline_df = bdf[bdf["Environment"] == env_name]
        if baseline_df.empty:
            baseline_df = None

    # Group by noise variance
    for var_val, df_var in df_rewards.groupby("NoiseVariance"):
        plt.figure(figsize=(8, 5))

        # Group by (ObsDim, ObsName) => label lines by dimension name
        for (dim_id, obs_name), df_dim in df_var.groupby(["ObsDim", "ObsName"]):
            df_dim = df_dim.sort_values("Episode")
            episodes = df_dim["Episode"].values
            rewards = df_dim["Reward"].values

            x_smooth, y_smooth = _smooth_reward_curve(episodes, rewards, window=smooth_window)
            label_str = f"Dim={dim_id}({obs_name})"
            plt.plot(x_smooth, y_smooth, label=label_str, linewidth=2)

        # Overplot baseline if available
        if baseline_df is not None:
            baseline_sorted = baseline_df.sort_values("Episode")
            b_eps = baseline_sorted["Episode"].values
            b_rew = baseline_sorted["TotalReward"].values
            bx_smooth, by_smooth = _smooth_reward_curve(b_eps, b_rew, window=smooth_window)
            plt.plot(bx_smooth, by_smooth, color="black", linewidth=3, label="Baseline")

        plt.title(f"{env_name}, Noise Var={var_val} (All Dims + Baseline)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        out_fname = f"{env_name}_var_{var_val}_all_dims_overlaid.png"
        out_path = os.path.join(output_dir, out_fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        logging.info(f"Noised multi-dim plot (plus baseline) saved to: {out_path}")


###############################################################################
# 2. Other plots remain the same: Rewards vs. Noise, Rewards vs. Markov, Noise vs. Markov Corr
###############################################################################
def plot_rewards_vs_noise(df_markov, output_dir="."):
    """
    Single plot per environment, X-axis=NoiseVariance, Y-axis=MeanFinalReward.
    Lines for each dimension.
    """
    os.makedirs(output_dir, exist_ok=True)
    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        for dim_id, gdf in env_df.groupby("ObsDim"):
            gdf_sorted = gdf.sort_values("NoiseVariance")
            ax.plot(
                gdf_sorted["NoiseVariance"],
                gdf_sorted["MeanFinalReward"],
                marker="o",
                label=f"dim={dim_id}"
            )
        ax.set_title(f"Rewards vs Noise - {env_name}")
        ax.set_xlabel("NoiseVariance")
        ax.set_ylabel("MeanFinalReward")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_rewards_vs_noise.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def plot_rewards_vs_markov(df_markov, output_dir="."):
    """
    Single plot per environment, X-axis=MarkovScore, Y-axis=MeanFinalReward.
    Lines for each dimension.
    """
    os.makedirs(output_dir, exist_ok=True)
    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        for dim_id, gdf in env_df.groupby("ObsDim"):
            gdf_sorted = gdf.sort_values("MarkovScore")
            ax.plot(
                gdf_sorted["MarkovScore"],
                gdf_sorted["MeanFinalReward"],
                marker="o",
                label=f"dim={dim_id}"
            )
        ax.set_title(f"Rewards vs Markovian Score - {env_name}")
        ax.set_xlabel("MarkovScore")
        ax.set_ylabel("MeanFinalReward")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_rewards_vs_markov.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def plot_noise_vs_markov_corr(df_markov, output_dir="."):
    """
    Single plot per environment, X-axis=NoiseVariance, Y-axis=MarkovScore, one dimension per line.
    Also compute correlation.
    """
    os.makedirs(output_dir, exist_ok=True)
    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6,5))
        for dim_id, gdf in env_df.groupby("ObsDim"):
            gdf_sorted = gdf.sort_values("NoiseVariance")
            ax.plot(
                gdf_sorted["NoiseVariance"],
                gdf_sorted["MarkovScore"],
                marker="o",
                label=f"dim={dim_id}"
            )
        if len(env_df) >= 2:
            corr_all = env_df[["NoiseVariance","MarkovScore"]].corr().iloc[0,1]
        else:
            corr_all = float('nan')
        ax.set_title(f"{env_name} - Noise vs Markov Score (corr={corr_all:.2f})")
        ax.set_xlabel("NoiseVariance")
        ax.set_ylabel("MarkovScore")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_noise_vs_markov_corr.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


###############################################################################
# NoisedExperiments
###############################################################################
class NoisedExperiments:
    def __init__(self, config, root_path="."):
        """
        config must contain:
         - "environments": [ { "name":..., "time_steps":..., "n_envs":..., "observations": [...] }, ... ]
         - "noise_strategies": { "gaussian": [ { "mean":0, "variance":0.01 }, ... ] }

        root_path: the root directory from main.py (where 'results' folder is).
        """
        self.config = config
        self.root_path = root_path
        self.reward_records = []
        self.markov_records = []

    def fishers_method(self, pvals, epsilon=1e-15):
        """
        Combine multiple p-values using Fisher's method.
        """
        pvals = np.array(pvals, dtype=float)
        pvals = np.clip(pvals, epsilon, 1 - epsilon)
        statistic = -2.0 * np.sum(np.log(pvals))
        df = 2 * len(pvals)
        return 1.0 - chi2.cdf(statistic, df)

    def gather_and_run_pcmci(self, model, env_name, dim_id, mean, variance, steps=2000, seed=None):
        """
        Creates a noised environment, collects `steps` observations using `model`,
        then runs PCMCI and returns (val_matrix, p_matrix).
        """
        test_env = make_noisy_env(
            env_name=env_name,
            dimension_to_noisify=dim_id,
            mean=mean,
            variance=variance,
            seed=seed
        )

        obs = test_env.reset()
        obs_list = []
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            output = test_env.step(action)
            # Gymnasium returns (obs, reward, done, truncated, info)
            if len(output) == 5:
                obs, reward, done, truncated, info = output
            else:
                obs, reward, done, info = output
                truncated = done

            obs_list.append(obs[0])  # if obs.shape=(1, obs_dim)
            if done[0] or truncated[0]:
                obs = test_env.reset()
        test_env.close()

        obs_array = np.array(obs_list)

        # Now run PCMCI
        cit = ConditionalIndependenceTest()
        results_dict = cit.run_pcmci(
            observations=obs_array,
            tau_min=1,  # adjust as needed
            tau_max=2,  # adjust as needed
            alpha_level=0.05,
            pc_alpha=None,
            env_id=env_name,
            label=f"dim_{dim_id}_var_{variance}_seed_{seed}",
            results_dir=os.path.join(self.root_path, "results", env_name, "noised", "pcmci")
        )
        return results_dict["val_matrix"], results_dict["p_matrix"]

    def run_multiple_pcmci_fisher(self, model, env_name, dim_id, mean, variance,
                                  num_runs=5, steps=2000):
        """
        Performs multiple rollouts for the given dimension/variance,
        runs PCMCI on each, and combines the results with:
          - averaged val_matrix
          - Fisher's method for p_matrix
        Returns the combined (val_matrix, p_matrix, markov_score).
        """
        val_list = []
        p_list = []

        for _ in range(num_runs):
            seed = random.randint(0, 9999)
            val_m, p_m = self.gather_and_run_pcmci(
                model, env_name, dim_id, mean, variance,
                steps=steps, seed=seed
            )
            val_list.append(val_m)
            p_list.append(p_m)

        val_arr = np.stack(val_list, axis=0)  # shape=(num_runs, N, N, L)
        p_arr = np.stack(p_list, axis=0)      # shape=(num_runs, N, N, L)

        # 1) Average val_matrix across runs
        avg_val_matrix = np.mean(val_arr, axis=0)

        # 2) Combine p_matrices across runs using Fisher’s method (element-wise)
        n_runs, N, _, L = p_arr.shape
        combined_p_matrix = np.zeros((N, N, L), dtype=float)

        for i in range(N):
            for j in range(N):
                for k in range(L):
                    pvals_for_link = p_arr[:, i, j, k]
                    combined_p_matrix[i, j, k] = self.fishers_method(pvals_for_link)

        # 3) Compute Markov violation
        mk_score = get_markov_violation_score(
            p_matrix=combined_p_matrix,
            val_matrix=avg_val_matrix,
            alpha_level=0.05
        )
        return avg_val_matrix, combined_p_matrix, mk_score

    def run(self, env_name=None, baseline_seed=None):
        """
        Main pipeline:
          - For each environment in config (or just env_name),
          - For each observation dimension,
          - For each noise (mean,var),
            => train model, log rewards
            => gather multiple PCMCI runs, combine with Fisher => Markov Score
        """
        envs = self.config["environments"]
        if env_name:
            envs = [e for e in envs if e["name"] == env_name]

        if not envs:
            logging.warning(f"[Noised] No matching environment for env_name={env_name}. Exiting.")
            return

        noise_list = self.config["noise_strategies"]["gaussian"]  # only 'gaussian' for now

        for env_item in envs:
            name = env_item["name"]
            time_steps = env_item["time_steps"]
            n_envs = env_item["n_envs"]
            obs_names = env_item.get("observations", [])
            obs_dim = len(obs_names)

            # Setup environment-specific results directories
            env_path = os.path.join(self.root_path, "results", name)
            noised_path = os.path.join(env_path, "noised")
            pcmci_path = os.path.join(noised_path, "pcmci")
            os.makedirs(noised_path, exist_ok=True)
            os.makedirs(pcmci_path, exist_ok=True)

            logging.info(f"[Noised] Start environment: {name}, obs_dim={obs_dim}")

            # For each dimension
            for dim_id in range(obs_dim):
                obs_dim_name = obs_names[dim_id]
                # For each noise param
                for noise_params in noise_list:
                    mean = noise_params.get("mean", 0.0)
                    var = noise_params.get("variance", 0.01)

                    logging.info(f"[Noised] env={name}, dim_id={dim_id}, mean={mean}, var={var}")

                    # 1) Create the environment with noise
                    used_seed = baseline_seed if baseline_seed is not None else random.randint(0, 100)
                    venv = make_noisy_env(
                        env_name=name,
                        dimension_to_noisify=dim_id,
                        mean=mean,
                        variance=var,
                        seed=used_seed
                    )

                    # 2) Train from scratch
                    model = RPPO("MlpPolicy", venv, verbose=0, learning_rate=3e-4)
                    callback = RewardTrackingCallback()
                    model.learn(total_timesteps=time_steps, callback=callback)
                    ep_rewards = callback.get_rewards()
                    venv.close()

                    # Store the entire reward curve
                    for i, rew in enumerate(ep_rewards):
                        self.reward_records.append({
                            "Environment": name,
                            "ObsDim": dim_id,
                            "ObsName": obs_dim_name,
                            "NoiseVariance": var,
                            "Episode": i + 1,
                            "Reward": rew
                        })

                    # 3) Instead of a single-run PCMCI, gather multiple runs + Fisher
                    NUM_PCMCI_RUNS = 15  # or 10, 15, etc.
                    avg_val_matrix, combined_p_matrix, mk_score = self.run_multiple_pcmci_fisher(
                        model=model,
                        env_name=name,
                        dim_id=dim_id,
                        mean=mean,
                        variance=var,
                        num_runs=NUM_PCMCI_RUNS,
                        steps=2000
                    )

                    logging.info(f"[Noised] Markov Score => {mk_score:.4f} (Fisher from {NUM_PCMCI_RUNS} runs)")

                    # 4) Store final metrics
                    final_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) > 10 else np.mean(ep_rewards)
                    self.markov_records.append({
                        "Environment": name,
                        "ObsDim": dim_id,
                        "ObsName": obs_dim_name,
                        "NoiseVariance": var,
                        "MarkovScore": mk_score,
                        "MeanFinalReward": final_reward
                    })

            # After finishing this environment, write out CSV & produce plots for *just* that env
            env_rewards = [r for r in self.reward_records if r["Environment"] == name]
            env_markov = [m for m in self.markov_records if m["Environment"] == name]

            df_rewards = pd.DataFrame(env_rewards)
            df_markov = pd.DataFrame(env_markov)

            # Save to environment-specific subfolder
            reward_csv = os.path.join(noised_path, "noised_rewards.csv")
            markov_csv = os.path.join(noised_path, "noised_markov.csv")

            df_rewards.to_csv(reward_csv, index=False)
            df_markov.to_csv(markov_csv, index=False)
            logging.info(f"[Noised] Wrote noised rewards to {reward_csv}")
            logging.info(f"[Noised] Wrote noised Markov to {markov_csv}")

            # Now produce the new single-figure learning curve for each variance
            baseline_csv_path = os.path.join(self.root_path, "results", name, "csv", "baseline_learning_curve.csv")
            plot_noised_learning_curves_all_dims(
                df_rewards,
                baseline_csv=baseline_csv_path,
                output_dir=noised_path,
                smooth_window=10
            )
            # The other plots remain
            plot_rewards_vs_noise(df_markov, output_dir=noised_path)
            plot_rewards_vs_markov(df_markov, output_dir=noised_path)
            plot_noise_vs_markov_corr(df_markov, output_dir=noised_path)


###############################################################################
# run_noised (for calling from main.py), main() for direct CLI
###############################################################################
def run_noised(config_path="config.json", env_name=None, baseline_seed=None):
    """
    We load config, create a NoisedExperiments, and run for a specific env_name (optional).
    The results are saved under results/<ENV>/noised, consistent with main.py logic.
    """
    if not os.path.exists(config_path):
        logging.error(f"Config file '{config_path}' not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    root_path = os.path.abspath(".")
    runner = NoisedExperiments(config, root_path=root_path)

    start_t = time.perf_counter()
    runner.run(env_name=env_name, baseline_seed=baseline_seed)
    end_t = time.perf_counter()
    logging.info(f"[Noised] Done! Total time: {((end_t - start_t)/60):.2f}m")


def main():
    parser = argparse.ArgumentParser(description="Noised Observations Experiments")
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Name of the environment to use from config.json.")
    args = parser.parse_args()

    run_noised(config_path=args.config_path, env_name=args.env)


if __name__ == "__main__":
    main()