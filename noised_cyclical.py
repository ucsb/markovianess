#!/usr/bin/env python
"""
noised_cyclical.py

Implements a complete pipeline for injecting cyclical (seasonal) noise
into specific observation dimensions, training an RL agent, and analyzing
Markovian properties using PCMCI + Fisher's method.

The cyclical noise has a time-varying variance:
    cyc_var(t) = base_variance + amplitude * sin(2π * frequency * t)

At each step t, we draw noise ~ Normal(0, sqrt(cyc_var(t))) if cyc_var(t) > 0,
clamping cyc_var(t) to a small positive number if it goes below zero.

Results (learning curves, CSV logs, Markov violations, etc.) are saved in:
  results/<ENV>/noised_cyclical/
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

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from rppo import RPPO
from conditional_independence_test import (
    ConditionalIndependenceTest,
    get_markov_violation_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - noised_cyc: %(message)s"
)

###############################################################################
# Callback to track episode rewards
###############################################################################
class RewardTrackingCallback(BaseCallback):
    """
    Tracks total reward per episode for a single-environment scenario.
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
# Cyclical Noise Wrapper
###############################################################################
class CyclicalObservationWrapper(gym.Wrapper):
    """
    A gym wrapper that adds cyclical (seasonal) noise to a selected
    dimension (or all dimensions) of the observation at each step.

    Noise variance changes over time as:
        cyc_var(t) = base_variance + amplitude * sin(2π * frequency * t)

    - dimension_to_noisify=None => apply cyclical noise to *all* observation dims
    - otherwise, only dimension_to_noisify is noised
    - If cyc_var(t) < 0, we clamp it to a small positive value so we can take sqrt().
    """
    def __init__(self, env, dimension_to_noisify=None, base_variance=0.1,
                 amplitude=0.5, frequency=0.005):
        super().__init__(env)
        self.dim_to_noisify = dimension_to_noisify
        self.base_variance = base_variance
        self.amplitude = amplitude
        self.frequency = frequency

        # We'll keep track of step count to modulate noise
        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        return obs, info

    def step(self, action):
        output = self.env.step(action)
        if len(output) == 5:
            obs, reward, done, truncated, info = output
        else:
            obs, reward, done, info = output
            truncated = done

        obs = np.array(obs, dtype=np.float32)

        # Compute cyclical variance at current step
        # cyc_var(t) = base_variance + amplitude * sin(2π * frequency * t)
        sin_arg = 2.0 * np.pi * self.frequency * self.step_count
        cyc_var = self.base_variance + self.amplitude * np.sin(sin_arg)
        # Ensure cyc_var is never negative to avoid sqrt of negative
        cyc_var_clamped = max(cyc_var, 1e-8)
        std_dev = np.sqrt(cyc_var_clamped)

        # Now sample noise ~ Normal(0, sqrt(cyc_var_clamped))
        if self.dim_to_noisify is None:
            # apply noise to all dims
            noise = np.random.normal(0.0, std_dev, size=obs.shape)
            obs += noise.astype(np.float32)
        else:
            # only one dimension
            noise_val = np.random.normal(0.0, std_dev)
            obs[self.dim_to_noisify] += noise_val

        self.step_count += 1
        return obs, reward, done, truncated, info


def make_cyclical_env(env_name, dimension_to_noisify=None,
                      base_variance=0.1, amplitude=0.5, frequency=0.005,
                      seed=None):
    """
    Creates a vectorized environment (n_envs=1) that wraps the base environment
    in cyclical noise logic for the specified dimension.
    """
    def _env_fn():
        e = gym.make(env_name)
        if seed is not None:
            e.reset(seed=seed)
        e = CyclicalObservationWrapper(
            e,
            dimension_to_noisify=dimension_to_noisify,
            base_variance=base_variance,
            amplitude=amplitude,
            frequency=frequency
        )
        return e

    venv = make_vec_env(_env_fn, n_envs=1, seed=seed)
    return venv


###############################################################################
# Utility: rolling average for smoothing
###############################################################################
def _smooth_reward_curve(episodes, rewards, window=10):
    """
    Returns a rolling mean of the rewards. If <window data points, just return raw data.
    """
    if len(rewards) < window:
        return episodes, rewards
    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    return episodes[window - 1:], smoothed


###############################################################################
# Plot 1: Learning curves for each cyclical noise setting (lines for each dim) + baseline
###############################################################################
def plot_cyclical_learning_curves_all_dims(
    df_rewards: pd.DataFrame,
    baseline_csv: str = None,
    output_dir: str = ".",
    smooth_window: int = 10
):
    """
    For each distinct (base_variance, amplitude, frequency) setting, produce one figure:
      - multiple lines, one per (ObsDim, ObsName)
      - plus black baseline if found

    The DataFrame should have columns:
      ["Environment", "ObsDim", "ObsName", "base_variance", "amplitude", "frequency",
       "Episode", "Reward"]
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_rewards.empty:
        logging.info("[CycNoised] No data to plot for cyclical noise.")
        return

    env_name = df_rewards["Environment"].iloc[0]

    # Attempt to load baseline CSV
    baseline_df = None
    if baseline_csv and os.path.isfile(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        baseline_df = bdf[bdf["Environment"] == env_name]
        if baseline_df.empty:
            baseline_df = None

    # Group by each cyclical noise setting
    grouped = df_rewards.groupby(["base_variance", "amplitude", "frequency"])
    for (base_var, amp, freq), df_cyc in grouped:
        plt.figure(figsize=(8, 5))

        # Group by dimension => lines
        for (dim_id, obs_name), df_dim in df_cyc.groupby(["ObsDim", "ObsName"]):
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

        plt.title(f"{env_name}: cyc(base_var={base_var}, amp={amp}, freq={freq})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        out_fname = (
            f"{env_name}_cyc_bv_{base_var}_amp_{amp}_freq_{freq}_all_dims.png"
        )
        out_path = os.path.join(output_dir, out_fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        logging.info(f"[CycNoised] Multi-dim cyc plot saved => {out_path}")


###############################################################################
# Plot 2: Rewards vs cyc params, Markov Score vs rewards, etc.
###############################################################################
def plot_rewards_vs_cycparams(df_markov, output_dir="."):
    """
    Single plot per environment: group by dimension. X-axis=base_variance or amplitude,
    or produce more advanced grouping. For simplicity, we'll just do a 2D plot with
    amplitude on the X-axis, lines for each base_variance, color-coded by dimension.
    If your config has multiple frequencies, you can do a 3D or make separate plots by frequency.

    We'll produce a separate figure for each frequency.
    """
    os.makedirs(output_dir, exist_ok=True)
    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        # We'll group by frequency to produce separate plots
        for freq_val, df_freq in env_df.groupby("frequency"):
            fig, ax = plt.subplots(figsize=(7,5))
            grouped = df_freq.groupby(["ObsDim", "base_variance"])
            for (dim_id, bv), gdf in grouped:
                gdf_sorted = gdf.sort_values("amplitude")
                ax.plot(
                    gdf_sorted["amplitude"],
                    gdf_sorted["MeanFinalReward"],
                    marker="o",
                    label=f"dim={dim_id}, bv={bv}"
                )
            ax.set_title(f"Rewards vs Amplitude (freq={freq_val}) - {env_name}")
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("MeanFinalReward")
            ax.legend()
            ax.grid(True)
            outpath = os.path.join(output_dir, f"{env_name}_freq_{freq_val}_rewards_vs_amp.png")
            fig.savefig(outpath, dpi=150)
            plt.close(fig)


def plot_rewards_vs_markov(df_markov, output_dir="."):
    """
    Single plot per environment: X-axis=MarkovScore, Y-axis=MeanFinalReward,
    lines or scatter grouped by dimension & cyc parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7,5))
        grouped = env_df.groupby(["ObsDim"])
        for dim_id, gdf in grouped:
            gdf_sorted = gdf.sort_values("MarkovScore")
            label_str = f"dim={dim_id}"
            ax.plot(
                gdf_sorted["MarkovScore"],
                gdf_sorted["MeanFinalReward"],
                marker="o",
                label=label_str
            )
        ax.set_title(f"Rewards vs Markov Score - {env_name} (Cyclical)")
        ax.set_xlabel("MarkovScore")
        ax.set_ylabel("MeanFinalReward")
        ax.legend()
        ax.grid(True)

        outpath = os.path.join(output_dir, f"{env_name}_rewards_vs_markov.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def plot_cyc_vs_markov_corr(df_markov, output_dir="."):
    """
    Single plot per environment: X-axis=Amplitude or base_variance, Y-axis=MarkovScore.
    We'll produce one figure for each frequency to reduce clutter.

    Also compute correlation if desired.
    """
    os.makedirs(output_dir, exist_ok=True)
    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        for freq_val, gdf_freq in env_df.groupby("frequency"):
            fig, ax = plt.subplots(figsize=(7,5))

            # group by dimension, base_variance => lines with amplitude as x
            grouped = gdf_freq.groupby(["ObsDim", "base_variance"])
            for (dim_id, bv), gdf in grouped:
                gdf_sorted = gdf.sort_values("amplitude")
                ax.plot(
                    gdf_sorted["amplitude"],
                    gdf_sorted["MarkovScore"],
                    marker="o",
                    label=f"dim={dim_id}, bv={bv}"
                )

            # correlation among amplitude <-> MarkovScore
            if len(gdf_freq) >= 2:
                corr_all = gdf_freq[["amplitude", "MarkovScore"]].corr().iloc[0,1]
            else:
                corr_all = float('nan')

            ax.set_title(f"{env_name} cyc freq={freq_val} (corr={corr_all:.2f})")
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("MarkovScore")
            ax.legend()
            ax.grid(True)

            outpath = os.path.join(output_dir, f"{env_name}_freq_{freq_val}_amp_vs_markov.png")
            fig.savefig(outpath, dpi=150)
            plt.close(fig)


###############################################################################
# Class for running cyclical noised experiments
###############################################################################
class CyclicalNoisedExperiments:
    """
    1) For each environment, for each dimension, for each cyclical noise
       setting in config["noise_strategies"]["cyclical"]:
         => train a new model from scratch
         => gather multiple rollouts for PCMCI (Fisher)
         => store Markov score + final reward
         => produce plots + CSV logs
    """
    def __init__(self, config, root_path="."):
        self.config = config
        self.root_path = root_path

        self.reward_records = []  # store entire reward curves
        self.markov_records = []  # store final Markov scores, final rewards

    def fishers_method(self, pvals, epsilon=1e-15):
        """
        Combine multiple p-values using Fisher's method.
        """
        pvals = np.array(pvals, dtype=float)
        pvals = np.clip(pvals, epsilon, 1 - epsilon)
        statistic = -2.0 * np.sum(np.log(pvals))
        df = 2 * len(pvals)
        return 1.0 - chi2.cdf(statistic, df)

    def gather_and_run_pcmci(self, model, env_name, dim_id, base_var,
                             amplitude, frequency, steps=2000, seed=None):
        """
        Create an environment with cyclical noise, gather `steps` observations
        from the trained model, then run PCMCI. Return (val_matrix, p_matrix).
        """
        test_env = make_cyclical_env(
            env_name=env_name,
            dimension_to_noisify=dim_id,
            base_variance=base_var,
            amplitude=amplitude,
            frequency=frequency,
            seed=seed
        )

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
            obs_list.append(obs[0])  # if shape=(1, obs_dim)
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
            label=f"cyc_dim_{dim_id}_bv_{base_var}_amp_{amplitude}_freq_{frequency}_seed_{seed}",
            results_dir=os.path.join(self.root_path, "results", env_name, "noised_cyclical", "pcmci")
        )
        return results_dict["val_matrix"], results_dict["p_matrix"]

    def run_multiple_pcmci_fisher(self, model, env_name, dim_id,
                                  base_var, amplitude, frequency,
                                  num_runs=5, steps=2000):
        """
        Perform multiple rollouts for PCMCI, combine p-values with Fisher, average val_matrix.
        Returns (avg_val_matrix, combined_p_matrix, markov_score).
        """
        val_list = []
        p_list = []

        for _ in range(num_runs):
            seed = random.randint(0, 9999)
            val_m, p_m = self.gather_and_run_pcmci(
                model=model,
                env_name=env_name,
                dim_id=dim_id,
                base_var=base_var,
                amplitude=amplitude,
                frequency=frequency,
                steps=steps,
                seed=seed
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

        mk_score = get_markov_violation_score(
            p_matrix=combined_p_matrix,
            val_matrix=avg_val_matrix,
            alpha_level=0.05
        )
        return avg_val_matrix, combined_p_matrix, mk_score

    def run(self, env_name=None, baseline_seed=None):
        """
        Main pipeline:
         - For each environment, dimension, cyc config in config["noise_strategies"]["cyclical"]
           => train model, track rewards
           => run PCMCI (Fisher)
           => store Markov score
        """
        envs = self.config["environments"]
        if env_name:
            envs = [e for e in envs if e["name"] == env_name]
        if not envs:
            logging.warning(f"[CycNoised] No matching environment for env={env_name}.")
            return

        cyc_list = self.config["noise_strategies"].get("cyclical", [])
        if not cyc_list:
            logging.warning("[CycNoised] No 'cyclical' noise configs found in config.")
            return

        for env_item in envs:
            name = env_item["name"]
            time_steps = env_item["time_steps"]
            n_envs = env_item["n_envs"]
            obs_names = env_item.get("observations", [])
            obs_dim = len(obs_names)

            env_path = os.path.join(self.root_path, "results", name)
            noised_path = os.path.join(env_path, "noised_cyclical")
            pcmci_path = os.path.join(noised_path, "pcmci")
            os.makedirs(noised_path, exist_ok=True)
            os.makedirs(pcmci_path, exist_ok=True)

            logging.info(f"[CycNoised] Starting environment={name}, {obs_dim} obs dims...")

            for dim_id in range(obs_dim):
                obs_dim_name = obs_names[dim_id]
                for cyc_params in cyc_list:
                    base_var = cyc_params.get("base_variance", 0.1)
                    amplitude = cyc_params.get("amplitude", 0.5)
                    frequency = cyc_params.get("frequency", 0.005)
                    desc = cyc_params.get("description", "")

                    logging.info(
                        f"[CycNoised] env={name}, dim={dim_id}({obs_dim_name}), "
                        f"base_var={base_var}, amp={amplitude}, freq={frequency}, desc={desc}"
                    )

                    # 1) Create environment with cyc noise
                    used_seed = baseline_seed if baseline_seed is not None else random.randint(0, 9999)
                    venv = make_cyclical_env(
                        env_name=name,
                        dimension_to_noisify=dim_id,
                        base_variance=base_var,
                        amplitude=amplitude,
                        frequency=frequency,
                        seed=used_seed
                    )

                    # 2) Train from scratch
                    model = RPPO("MlpPolicy", venv, verbose=0, learning_rate=3e-4)
                    callback = RewardTrackingCallback()
                    model.learn(total_timesteps=time_steps, callback=callback)
                    ep_rewards = callback.get_rewards()
                    venv.close()

                    # Record full reward curve
                    for i, rew in enumerate(ep_rewards):
                        self.reward_records.append({
                            "Environment": name,
                            "ObsDim": dim_id,
                            "ObsName": obs_dim_name,
                            "base_variance": base_var,
                            "amplitude": amplitude,
                            "frequency": frequency,
                            "Episode": i + 1,
                            "Reward": rew
                        })

                    # 3) Multiple PCMCI runs => Fisher
                    NUM_PCMCI_RUNS = 15
                    _, _, mk_score = self.run_multiple_pcmci_fisher(
                        model=model,
                        env_name=name,
                        dim_id=dim_id,
                        base_var=base_var,
                        amplitude=amplitude,
                        frequency=frequency,
                        num_runs=NUM_PCMCI_RUNS,
                        steps=2000
                    )

                    logging.info(f"[CycNoised] Markov Score => {mk_score:.4f} (Fisher from {NUM_PCMCI_RUNS} runs)")

                    # 4) Final metrics
                    final_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) > 10 else np.mean(ep_rewards)
                    self.markov_records.append({
                        "Environment": name,
                        "ObsDim": dim_id,
                        "ObsName": obs_dim_name,
                        "base_variance": base_var,
                        "amplitude": amplitude,
                        "frequency": frequency,
                        "MarkovScore": mk_score,
                        "MeanFinalReward": final_reward
                    })

            # Done with environment => Save CSV & produce plots
            env_rewards = [r for r in self.reward_records if r["Environment"] == name]
            env_markov = [m for m in self.markov_records if m["Environment"] == name]

            df_rewards = pd.DataFrame(env_rewards)
            df_markov = pd.DataFrame(env_markov)

            reward_csv = os.path.join(noised_path, "cyc_noised_rewards.csv")
            markov_csv = os.path.join(noised_path, "cyc_noised_markov.csv")
            df_rewards.to_csv(reward_csv, index=False)
            df_markov.to_csv(markov_csv, index=False)

            logging.info(f"[CycNoised] Wrote cyclical noised rewards => {reward_csv}")
            logging.info(f"[CycNoised] Wrote cyclical noised Markov => {markov_csv}")

            # Generate the plots
            baseline_csv_path = os.path.join(env_path, "csv", "baseline_learning_curve.csv")
            plot_cyclical_learning_curves_all_dims(df_rewards, baseline_csv=baseline_csv_path, output_dir=noised_path)
            plot_rewards_vs_cycparams(df_markov, output_dir=noised_path)
            plot_rewards_vs_markov(df_markov, output_dir=noised_path)
            plot_cyc_vs_markov_corr(df_markov, output_dir=noised_path)


###############################################################################
# Entry function
###############################################################################
def run_cyc_noised(config_path="config.json", env_name=None, baseline_seed=None):
    """
    Loads config, runs CyclicalNoisedExperiments for the specified environment (optional).
    Results are stored in results/<ENV>/noised_cyclical.
    """
    if not os.path.exists(config_path):
        logging.error(f"Config file '{config_path}' not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    root_path = os.path.abspath("deploy")
    runner = CyclicalNoisedExperiments(config, root_path=root_path)

    start_t = time.perf_counter()
    runner.run(env_name=env_name, baseline_seed=baseline_seed)
    end_t = time.perf_counter()
    logging.info(f"[CycNoised] Done! Total time: {(end_t - start_t):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Cyclical Noised Observations Experiments")
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Name of the environment to use from config.json.")
    args = parser.parse_args()

    run_cyc_noised(config_path=args.config_path, env_name=args.env)

if __name__ == "__main__":
    main()