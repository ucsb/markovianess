#!/usr/bin/env python
"""
noised_auto_regressive.py

Implements a complete pipeline for injecting AR(1) noise into specific
observation dimensions, training an RL agent (RPPO), and analyzing
Markovian properties using PCMCI with Fisher's method. Outputs and plots
are stored under results/<ENV>/noised_ar/.
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
    format='%(asctime)s - %(levelname)s - noised_auto_regressive: %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),   # Writes logs to file
        logging.StreamHandler()            # Writes logs to console (stdout)
    ]
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
# AR(1) Noise Wrapper
###############################################################################
class ARObservationWrapper(gym.Wrapper):
    """
    A gym wrapper that adds AR(1) noise to a selected dimension (or all dimensions)
    of the observation at each step.

    AR(1) process for noise_t:
        noise_t = phi * noise_{t-1} + eps_t,
        eps_t ~ Normal(0, variance)

    - `dimension_to_noisify=None` => apply AR noise to *all* observation dims
    - otherwise, only the dimension at `dimension_to_noisify` gets AR noise.
    - `phi` is the AR(1) coefficient.
    - `variance` is the process variance for eps_t.
    """
    def __init__(self, env, dimension_to_noisify=None, phi=0.9, variance=0.01):
        super().__init__(env)
        self.dim_to_noisify = dimension_to_noisify
        self.phi = phi
        self.variance = variance
        self.std = np.sqrt(variance)

        # Keep track of the previous noise value(s). If dimension_to_noisify is None,
        # we'll track a noise vector with the same dimension as the obs.
        obs_dim = self.observation_space.shape[0]
        if self.dim_to_noisify is None:
            self.prev_noise = np.zeros(obs_dim, dtype=np.float32)
        else:
            self.prev_noise = 0.0  # single float for a single dimension

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Reset the AR noise on environment reset
        if self.dim_to_noisify is None:
            self.prev_noise = np.zeros_like(self.prev_noise, dtype=np.float32)
        else:
            self.prev_noise = 0.0

        return obs, info

    def step(self, action):
        output = self.env.step(action)
        if len(output) == 5:
            obs, reward, done, truncated, info = output
        else:
            obs, reward, done, info = output
            truncated = done

        obs = np.array(obs, dtype=np.float32)

        # Generate new noise for the current step
        if self.dim_to_noisify is None:
            # AR noise for *each* dimension
            eps = np.random.normal(loc=0.0, scale=self.std, size=obs.shape)
            noise_t = self.phi * self.prev_noise + eps
            obs = obs + noise_t
            self.prev_noise = noise_t  # store for next step
        else:
            # AR noise for just one dimension
            eps = np.random.normal(loc=0.0, scale=self.std)
            noise_t = self.phi * self.prev_noise + eps
            obs[self.dim_to_noisify] += noise_t
            self.prev_noise = noise_t

        return obs, reward, done, truncated, info


def make_ar_env(env_name, dimension_to_noisify=None, phi=0.9, variance=0.01, seed=None):
    """
    Creates a vectorized environment (with n_envs=1) that wraps the base environment
    in AR(1)-noise logic for the specified dimension.
    """
    def _env_fn():
        e = gym.make(env_name)
        if seed is not None:
            e.reset(seed=seed)
        e = ARObservationWrapper(
            e,
            dimension_to_noisify=dimension_to_noisify,
            phi=phi,
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
    Returns a rolling mean of the rewards. If <window data points, just return raw data.
    """
    if len(rewards) < window:
        return episodes, rewards
    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    return episodes[window - 1:], smoothed


###############################################################################
# Plot 1: Learning curves for each AR noise setting (lines for each dimension) + baseline
###############################################################################
def plot_ar_learning_curves_all_dims(
    df_rewards: pd.DataFrame,
    baseline_csv: str = None,
    output_dir: str = ".",
    smooth_window: int = 10
):
    """
    For each distinct (phi, variance) setting, produce a single figure:
      - multiple lines, one per (ObsDim, ObsName),
      - plus black baseline if found.

    The df_rewards should have columns:
      ["Environment", "ObsDim", "ObsName", "AR_phi", "AR_variance", "Episode", "Reward"]
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_rewards.empty:
        logging.info("[ARNoised] No data to plot for AR noise.")
        return

    env_name = df_rewards["Environment"].iloc[0]

    # Attempt to load baseline CSV
    baseline_df = None
    if baseline_csv and os.path.isfile(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        baseline_df = bdf[bdf["Environment"] == env_name]
        if baseline_df.empty:
            baseline_df = None

    # Group by each AR setting (phi, variance)
    grouped = df_rewards.groupby(["AR_phi", "AR_variance"])
    for (phi_val, var_val), df_var in grouped:
        plt.figure(figsize=(8, 5))

        # Group by dimension => lines
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

        plt.title(f"{env_name} - AR(1) Noise [phi={phi_val}, var={var_val}] (All Dims)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        out_fname = f"{env_name}_ar_phi_{phi_val}_var_{var_val}_all_dims.png"
        out_path = os.path.join(output_dir, out_fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        logging.info(f"[ARNoised] Multi-dim AR(1) plot saved => {out_path}")


###############################################################################
# Plot 2: Rewards vs AR parameters, Markov Score vs rewards, etc.
###############################################################################
def plot_rewards_vs_arparams(df_markov, output_dir="."):
    """
    Single plot per environment showing X-axis=(phi, variance) or just variance
    vs. Y-axis=MeanFinalReward. We'll produce lines grouped by dimension, or show a scatter.
    For simplicity, show separate lines for each dimension, with phi labeled in legend.
    """
    os.makedirs(output_dir, exist_ok=True)
    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        # We'll treat each dimension as a separate line, X-axis = variance, or we can break it out by phi in a legend.
        # If you prefer a more advanced multi-axis approach, you can adapt this snippet.
        # For clarity, just do: X-axis=variance, separate lines for each (dim_id, phi).
        fig, ax = plt.subplots(figsize=(7,5))

        grouped = env_df.groupby(["ObsDim", "AR_phi"])
        for (dim_id, phi_val), gdf in grouped:
            gdf_sorted = gdf.sort_values("AR_variance")
            ax.plot(
                gdf_sorted["AR_variance"],
                gdf_sorted["MeanFinalReward"],
                marker="o",
                label=f"dim={dim_id}, phi={phi_val}"
            )

        ax.set_title(f"Rewards vs AR Variance - {env_name}")
        ax.set_xlabel("AR Variance")
        ax.set_ylabel("MeanFinalReward")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_rewards_vs_ar_variance.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def plot_rewards_vs_markov(df_markov, output_dir="."):
    """
    Single plot per environment: X-axis=MarkovScore, Y-axis=MeanFinalReward,
    lines or scatter grouped by dimension & AR parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7,5))
        grouped = env_df.groupby(["ObsDim", "AR_phi"])
        for (dim_id, phi_val), gdf in grouped:
            gdf_sorted = gdf.sort_values("MarkovScore")
            ax.plot(
                gdf_sorted["MarkovScore"],
                gdf_sorted["MeanFinalReward"],
                marker="o",
                label=f"dim={dim_id}, phi={phi_val}"
            )
        ax.set_title(f"Rewards vs Markov Score - {env_name}")
        ax.set_xlabel("MarkovScore")
        ax.set_ylabel("MeanFinalReward")
        ax.legend()
        ax.grid(True)

        outpath = os.path.join(output_dir, f"{env_name}_rewards_vs_markov.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def plot_ar_vs_markov_corr(df_markov, output_dir="."):
    """
    Single plot per environment: X-axis=AR Variance, Y-axis=MarkovScore, lines grouped by dimension & phi.
    Also compute correlation if desired.
    """
    os.makedirs(output_dir, exist_ok=True)
    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7,5))
        grouped = env_df.groupby(["ObsDim", "AR_phi"])
        for (dim_id, phi_val), gdf in grouped:
            gdf_sorted = gdf.sort_values("AR_variance")
            ax.plot(
                gdf_sorted["AR_variance"],
                gdf_sorted["MarkovScore"],
                marker="o",
                label=f"dim={dim_id}, phi={phi_val}"
            )
        # Compute correlation if enough points
        if len(env_df) >= 2:
            corr_all = env_df[["AR_variance","MarkovScore"]].corr().iloc[0,1]
        else:
            corr_all = float('nan')

        ax.set_title(f"{env_name} - AR Var vs Markov Score (corr={corr_all:.2f})")
        ax.set_xlabel("AR Variance")
        ax.set_ylabel("MarkovScore")
        ax.legend()
        ax.grid(True)

        outpath = os.path.join(output_dir, f"{env_name}_ar_variance_vs_markov_corr.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


###############################################################################
# Class for running AR(1) noised experiments
###############################################################################
class ARNoisedExperiments:
    """
    Similar to the pipeline in noised_gaussian.py, but for auto-regressive noise.
    1) For each env:
       - For each dimension to noisify
       - For each AR(1) setting: (phi, variance)
         => create env, train RPPO, gather multiple PCMCI rollouts, combine with Fisher
         => store results
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

    def gather_and_run_pcmci(self, model, env_name, dim_id, phi, variance,
                             steps=2000, seed=None):
        """
        Create an environment with AR(1) noise, gather `steps` observations
        from the trained model, then run PCMCI. Return (val_matrix, p_matrix).
        """
        test_env = make_ar_env(
            env_name=env_name,
            dimension_to_noisify=dim_id,
            phi=phi,
            variance=variance,
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
            label=f"ar_dim_{dim_id}_phi_{phi}_var_{variance}_seed_{seed}",
            results_dir=os.path.join(self.root_path, "results", env_name, "noised_ar", "pcmci")
        )
        return results_dict["val_matrix"], results_dict["p_matrix"]

    def run_multiple_pcmci_fisher(self, model, env_name, dim_id, phi, variance,
                                  num_runs=5, steps=2000):
        """
        Gathers multiple rollouts for PCMCI (num_runs),
        then combines the p-values with Fisher's method and averages val_matrix.
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
                phi=phi,
                variance=variance,
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

        # Fisher combine
        for i in range(N):
            for j in range(N):
                for k in range(L):
                    pvals_for_link = p_arr[:, i, j, k]
                    combined_p_matrix[i, j, k] = self.fishers_method(pvals_for_link)

        # Markov violation
        mk_score = get_markov_violation_score(
            p_matrix=combined_p_matrix,
            val_matrix=avg_val_matrix,
            alpha_level=0.05
        )
        return avg_val_matrix, combined_p_matrix, mk_score

    def run(self, env_name=None, baseline_seed=None):
        """
        Main pipeline:
         - For each environment, for each dimension, for each AR(1) setting from config:
           => train model, track rewards
           => run PCMCI (Fisher)
           => store Markov score
        """
        envs = self.config["environments"]
        if env_name:
            envs = [e for e in envs if e["name"] == env_name]
        if not envs:
            logging.warning(f"[ARNoised] No matching environment for env={env_name}.")
            return

        # We expect auto_regressive noise configs in config["noise_strategies"]["auto_regressive"]
        ar_noise_list = self.config["noise_strategies"].get("auto_regressive", [])
        if not ar_noise_list:
            logging.warning("[ARNoised] No 'auto_regressive' noise configs found in config.")
            return

        for env_item in envs:
            name = env_item["name"]
            time_steps = env_item["time_steps"]
            n_envs = env_item["n_envs"]
            obs_names = env_item.get("observations", [])
            obs_dim = len(obs_names)

            env_path = os.path.join(self.root_path, "results", name)
            noised_path = os.path.join(env_path, "noised_ar")
            pcmci_path = os.path.join(noised_path, "pcmci")
            os.makedirs(noised_path, exist_ok=True)
            os.makedirs(pcmci_path, exist_ok=True)

            logging.info(f"[ARNoised] Starting environment={name} with {obs_dim} obs dims...")

            for dim_id in range(obs_dim):
                obs_dim_name = obs_names[dim_id]
                for ar_params in ar_noise_list:
                    phi = ar_params.get("phi", 0.9)
                    variance = ar_params.get("variance", 0.01)
                    desc = ar_params.get("description", "")

                    logging.info(f"[ARNoised] env={name}, dim={dim_id}({obs_dim_name}), phi={phi}, var={variance}, desc={desc}")

                    # 1) Create environment with AR noise
                    used_seed = baseline_seed if baseline_seed is not None else random.randint(0, 9999)
                    venv = make_ar_env(
                        env_name=name,
                        dimension_to_noisify=dim_id,
                        phi=phi,
                        variance=variance,
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
                            "AR_phi": phi,
                            "AR_variance": variance,
                            "Episode": i+1,
                            "Reward": rew
                        })

                    # 3) Multiple PCMCI runs => combine with Fisher
                    NUM_PCMCI_RUNS = 15
                    _, _, mk_score = self.run_multiple_pcmci_fisher(
                        model=model,
                        env_name=name,
                        dim_id=dim_id,
                        phi=phi,
                        variance=variance,
                        num_runs=NUM_PCMCI_RUNS,
                        steps=2000
                    )

                    logging.info(f"[ARNoised] Markov Score => {mk_score:.4f} (Fisher from {NUM_PCMCI_RUNS} runs)")

                    # 4) Final metrics
                    final_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) > 10 else np.mean(ep_rewards)
                    self.markov_records.append({
                        "Environment": name,
                        "ObsDim": dim_id,
                        "ObsName": obs_dim_name,
                        "AR_phi": phi,
                        "AR_variance": variance,
                        "MarkovScore": mk_score,
                        "MeanFinalReward": final_reward
                    })

            # After finishing this environment, save CSV & produce plots
            env_rewards = [r for r in self.reward_records if r["Environment"] == name]
            env_markov = [m for m in self.markov_records if m["Environment"] == name]

            df_rewards = pd.DataFrame(env_rewards)
            df_markov = pd.DataFrame(env_markov)

            reward_csv = os.path.join(noised_path, "ar_noised_rewards.csv")
            markov_csv = os.path.join(noised_path, "ar_noised_markov.csv")
            df_rewards.to_csv(reward_csv, index=False)
            df_markov.to_csv(markov_csv, index=False)
            logging.info(f"[ARNoised] Wrote AR noised rewards => {reward_csv}")
            logging.info(f"[ARNoised] Wrote AR noised Markov => {markov_csv}")

            # Plots
            baseline_csv_path = os.path.join(env_path, "csv", "baseline_learning_curve.csv")
            plot_ar_learning_curves_all_dims(df_rewards, baseline_csv=baseline_csv_path, output_dir=noised_path)
            plot_rewards_vs_arparams(df_markov, output_dir=noised_path)
            plot_rewards_vs_markov(df_markov, output_dir=noised_path)
            plot_ar_vs_markov_corr(df_markov, output_dir=noised_path)


###############################################################################
# Entry function
###############################################################################
def run_ar_noised(config_path="config.json", env_name=None, baseline_seed=None):
    """
    Loads config, runs ARNoisedExperiments for the specified environment (optional).
    Results are stored in results/<ENV>/noised_ar.
    """
    if not os.path.exists(config_path):
        logging.error(f"Config file '{config_path}' not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    root_path = os.path.abspath(".")
    runner = ARNoisedExperiments(config, root_path=root_path)

    start_t = time.perf_counter()
    runner.run(env_name=env_name, baseline_seed=baseline_seed)
    end_t = time.perf_counter()
    logging.info(f"[ARNoised] Done! Total time: {((end_t - start_t)/60):.2f}m")


def main():
    parser = argparse.ArgumentParser(description="Auto-Regressive (AR) Noised Observations Experiments")
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Name of the environment to use from config.json.")
    args = parser.parse_args()

    run_ar_noised(config_path=args.config_path, env_name=args.env)

if __name__ == "__main__":
    main()