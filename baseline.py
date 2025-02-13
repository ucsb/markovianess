import logging
import time
import random
import uuid
import os
import sys
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from rppo import RPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from conditional_independence_test import (
    ConditionalIndependenceTest,
    get_markov_violation_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - baseline: %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),   # Writes logs to file
        logging.StreamHandler()            # Writes logs to console (stdout)
    ]
)


class RewardTrackingCallback(BaseCallback):
    """
    Records total reward per episode in a list for retrieval after training.
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


class BaselineExperiments:
    """
    Baseline RL experiments for a single environment, single training run.
    Also runs PCMCI-based conditional independence tests on random or trained rollouts.
    """

    def __init__(self, environment, config):
        """
        :param environment: (str) Environment name, e.g., "CartPole-v1".
        :param config: (dict) Configuration dictionary from config.json.
        """
        self.environment = environment
        self.config = config
        self.last_seed = None

        # Training parameters
        self.time_steps = self.config["time_steps"]
        self.num_envs = self.config["n_envs"]

        # PCMCI config
        self.pcmci_config = self.config.get("pcmci", {})
        self.tau_min = self.pcmci_config.get("tau_min", 1)
        self.tau_max = self.pcmci_config.get("tau_max", 2)
        self.alpha_level = self.pcmci_config.get("significance_level", 0.05)
        self.pc_alpha = self.pcmci_config.get("pc_alpha", None)

        # For storing observations or models, if needed
        self.loaded_models = {}
        self.storage = {
            "observations": {}
        }

        # Callback for RL training
        self.reward_callback = RewardTrackingCallback()

        # Directory structure
        self.results_path = os.path.join("results", self.environment)
        self.csv_path = os.path.join(self.results_path, "csv")
        self.plots_path = os.path.join(self.results_path, "plots", "baseline")
        self.models_path = os.path.join(self.results_path, "models")
        self.pcmci_path = os.path.join(self.results_path, "pcmci")

        # Create directories
        os.makedirs(self.csv_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.pcmci_path, exist_ok=True)

    def gather_observations(self, env_id, num_steps=1000, policy='random'):
        """
        Gather observations from a single environment instance.
        """
        import gymnasium as gym
        env = gym.make(env_id)

        observations = []
        obs, _ = env.reset()
        for _ in range(num_steps):
            if policy == 'random':
                action = env.action_space.sample()
            elif policy == 'trained':
                model = self.loaded_models.get(env_id, None)
                if model is None:
                    model = self._load_model_from_disk(env_id)
                if model is None:
                    logging.warning(f"No trained model found for {env_id}. Using random actions.")
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
            else:
                logging.warning(f"Unrecognized policy='{policy}'. Defaulting to random actions.")
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            observations.append(obs)

            if done or truncated:
                obs, _ = env.reset()

        env.close()
        observations = np.array(observations)

        # Optionally store in class-level storage
        self.storage.setdefault('observations', {})[f"{env_id}_{policy}"] = observations
        logging.info(f"Gathered {len(observations)} observations for env={env_id}, policy={policy}.")
        return observations

    def _load_model_from_disk(self, env_id):
        """
        Attempt to load the first .zip model from disk, if available.
        """
        candidate_files = [
            f for f in os.listdir(self.models_path) if f.endswith(".zip")
        ]
        if not candidate_files:
            return None

        model_path = os.path.join(self.models_path, candidate_files[0])
        logging.info(f"Loading model from {model_path} for env_id={env_id}.")
        model = RPPO.load(model_path)
        self.loaded_models[env_id] = model
        return model

    def fishers_method(self, pvals, epsilon=1e-15):
        """
        Combine multiple p-values using Fisher's method.

        Parameters:
        -----------
        pvals : array-like
            1D array or list of p-values from independent tests.
        epsilon : float
            A small constant to prevent log(0).

        Returns:
        --------
        float
            The combined p-value from Fisher's method.
        """
        # Convert to NumPy array
        pvals = np.array(pvals, dtype=float)
        # Clip p-values so none are exactly 0 or 1 to avoid numerical issues
        pvals = np.clip(pvals, epsilon, 1 - epsilon)

        # Fisher’s statistic
        statistic = -2.0 * np.sum(np.log(pvals))
        # Degrees of freedom = 2 * number_of_pvals
        df = 2 * len(pvals)

        # Combined p-value
        combined_pval = 1.0 - chi2.cdf(statistic, df)
        return combined_pval

    def run_ci_tests_fisher(self, n_times=5, policy='random', label=None):
        """
        Gathers observations multiple times (with the specified policy),
        runs PCMCI, then:
          - Averages val_matrix across runs (element-wise mean).
          - Combines p_matrix across runs using Fisher’s method (element-wise).

        Finally prints and saves results, including an aggregated Markov violation score.

        label (str): e.g. 'random' or 'trained' - appended to filenames & folder.
        """
        if label is None:
            label = policy  # fallback to the same name

        # Subdir for storing results for this particular label
        run_dir = os.path.join(self.pcmci_path, label)
        os.makedirs(run_dir, exist_ok=True)

        cit = ConditionalIndependenceTest()
        val_matrices = []
        p_matrices = []

        # Collect multiple runs
        for i in range(n_times):
            obs = self.gather_observations(
                env_id=self.environment,
                num_steps=1000,
                policy=policy
            )

            # Run PCMCI for this rollout
            results = cit.run_pcmci(
                observations=obs,
                tau_min=self.tau_min,
                tau_max=self.tau_max,
                alpha_level=self.alpha_level,
                pc_alpha=self.pc_alpha,
                env_id=self.environment,
                label=f"{label}_run_{i}",
                results_dir=run_dir
            )

            val_matrix = results['val_matrix']
            p_matrix = results['p_matrix']

            val_matrices.append(val_matrix)
            p_matrices.append(p_matrix)

        # Convert collected matrices to arrays of shape (n_times, N, N, lag_count)
        val_matrices = np.stack(val_matrices, axis=0)  # shape = (n_times, N, N, L)
        p_matrices = np.stack(p_matrices, axis=0)  # shape = (n_times, N, N, L)

        # 1) Average val_matrix across runs
        avg_val_matrix = np.mean(val_matrices, axis=0)  # shape = (N, N, L)

        # 2) Combine p_matrices across runs using Fisher’s method (element-wise)
        n_times, N, _, L = p_matrices.shape
        combined_p_matrix = np.zeros((N, N, L), dtype=float)

        for i in range(N):
            for j in range(N):
                for k in range(L):
                    # Extract the p-values across runs for (i, j, k)
                    pvals_for_link = p_matrices[:, i, j, k]
                    # Combine them with Fisher’s method
                    combined_p_matrix[i, j, k] = self.fishers_method(pvals_for_link)

        # Print "significant links" for the combined matrices
        dummy_df = DataFrame(data=obs)
        cond_ind_test = ParCorr()
        dummy_pcmci = PCMCI(dataframe=dummy_df, cond_ind_test=cond_ind_test)

        significant_txt_file = os.path.join(run_dir, f"fisher_significant_links_{label}.txt")
        old_stdout = sys.stdout
        try:
            with open(significant_txt_file, "w") as f:
                sys.stdout = f
                dummy_pcmci.print_significant_links(
                    p_matrix=combined_p_matrix,
                    val_matrix=avg_val_matrix,
                    alpha_level=self.alpha_level,
                )
        finally:
            sys.stdout = old_stdout

        # Compute Markov violation with the combined p_matrix & average val_matrix
        markovian_violation_score = get_markov_violation_score(
            p_matrix=combined_p_matrix,
            val_matrix=avg_val_matrix,
            alpha_level=self.alpha_level
        )

        # Save the combined results
        combined_npz_file = os.path.join(run_dir, f"fisher_val_and_p_matrices_{label}.npz")
        np.savez_compressed(
            combined_npz_file,
            val_matrix=avg_val_matrix,
            p_matrix=combined_p_matrix,
            markovian_violation_score=markovian_violation_score
        )

        logging.info(f"[{label}] Combined PCMCI results saved to {combined_npz_file}")
        logging.info(f"[{label}] Significant links saved to {significant_txt_file}")
        logging.info(f"[{label}] Markovian violation score: {markovian_violation_score:.4f}")

        # Also store the Markov violation score in a text file
        markov_txt_file = os.path.join(run_dir, f"markovian_violation_score_{label}.txt")
        with open(markov_txt_file, "w") as f:
            f.write(f"{markovian_violation_score:.6f}\n")

        logging.info(f"[{label}] Markovian violation score saved to {markov_txt_file}")

    def run_ci_tests(self, n_times=5, policy='random', label=None):
        """
        Gathers observations multiple times (with the specified policy),
        runs PCMCI, averages val/p_matrix, prints & saves the results.

        label (str): e.g. 'random' or 'trained' - appended to filenames & folder.
        """
        if label is None:
            label = policy  # fallback to the same name

        # Subdir for storing results for this particular label
        run_dir = os.path.join(self.pcmci_path, label)
        os.makedirs(run_dir, exist_ok=True)

        cit = ConditionalIndependenceTest()
        sum_val_matrix = None
        sum_p_matrix = None

        for i in range(n_times):
            obs = self.gather_observations(
                env_id=self.environment,
                num_steps=1000,
                policy=policy
            )

            # Run PCMCI for this rollout
            results = cit.run_pcmci(
                observations=obs,
                tau_min=self.tau_min,
                tau_max=self.tau_max,
                alpha_level=self.alpha_level,
                pc_alpha=self.pc_alpha,
                env_id=self.environment,
                label=f"{label}_run_{i}",
                results_dir=run_dir
            )

            val_matrix = results['val_matrix']
            p_matrix = results['p_matrix']

            if sum_val_matrix is None:
                sum_val_matrix = val_matrix.copy()
                sum_p_matrix = p_matrix.copy()
            else:
                sum_val_matrix += val_matrix
                sum_p_matrix += p_matrix

        # Average over all rollouts
        avg_val_matrix = sum_val_matrix / n_times
        avg_p_matrix = sum_p_matrix / n_times

        # Print significant links for the averaged matrices
        dummy_df = DataFrame(data=obs)
        cond_ind_test = ParCorr()
        dummy_pcmci = PCMCI(dataframe=dummy_df, cond_ind_test=cond_ind_test)

        significant_txt_file = os.path.join(run_dir, f"avg_significant_links_{label}.txt")
        with open(significant_txt_file, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            dummy_pcmci.print_significant_links(
                p_matrix=avg_p_matrix,
                val_matrix=avg_val_matrix,
                alpha_level=self.alpha_level,
            )
            sys.stdout = old_stdout

        # Compute Markov violation
        markovian_violation_score = get_markov_violation_score(
            p_matrix=avg_p_matrix,
            val_matrix=avg_val_matrix,
            alpha_level=self.alpha_level
        )

        avg_npz_file = os.path.join(run_dir, f"avg_val_and_p_matrices_{label}.npz")
        np.savez_compressed(
            avg_npz_file,
            val_matrix=avg_val_matrix,
            p_matrix=avg_p_matrix,
            markovian_violation_score=markovian_violation_score
        )

        logging.info(f"[{label}] Averaged PCMCI results saved to {avg_npz_file}")
        logging.info(f"[{label}] Significant links saved to {significant_txt_file}")
        logging.info(f"[{label}] Markovian violation score: {markovian_violation_score:.4f}")

        # Also store the Markov score in a text file
        markov_txt_file = os.path.join(run_dir, f"markovian_violation_score_{label}.txt")
        with open(markov_txt_file, "w") as f:
            f.write(f"{markovian_violation_score:.6f}\n")

        logging.info(f"[{label}] Markovian violation score saved to {markov_txt_file}")

    def plot_learning_curve(self, df, output_file, smoothing_window=10):
        """
        Plots the learning curve (episode vs. total reward) with smoothing.
        """
        df['SmoothedReward'] = df['TotalReward'].rolling(window=smoothing_window, min_periods=1).mean()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Episode", y="SmoothedReward",
                     label="Smoothed Reward", color="blue")
        sns.lineplot(data=df, x="Episode", y="TotalReward",
                     label="Raw Reward", color="red", alpha=0.3)

        plt.title(f"Learning Curve (with smoothing) for {self.environment}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def train(self):
        """
        Perform a single training run for the specified environment.
        """
        self.last_seed = random.randint(0, 1000)
        env_uid = str(uuid.uuid4())

        logging.info(f"Creating environment: {self.environment}, UID: {env_uid}, Seed: {self.last_seed}")

        # Create environment
        vec_env = make_vec_env(self.environment, n_envs=self.num_envs, seed=self.last_seed)

        # Initialize and train the RL model
        model = RPPO("MlpPolicy", vec_env, verbose=0, noise=None, learning_rate=3e-4)
        model.learn(total_timesteps=self.time_steps, callback=self.reward_callback)

        # Save model
        model_path = os.path.join(self.models_path, f"{env_uid}.zip")
        model.save(model_path)
        logging.info(f"Model saved to: {model_path}")

        # Extract final episode rewards
        episode_rewards = self.reward_callback.get_rewards()
        results = []
        for i, reward_val in enumerate(episode_rewards, start=1):
            results.append({
                "Environment": self.environment,
                "UID": env_uid,
                "Seed": self.last_seed,
                "Episode": i,
                "TotalReward": reward_val
            })

        # Save to CSV
        df = pd.DataFrame(results)
        csv_file_name = f"baseline_learning_curve.csv"
        csv_full_path = os.path.join(self.csv_path, csv_file_name)
        df.to_csv(csv_full_path, index=False)
        logging.info(f"Saved learning curve CSV to: {csv_full_path}")

        # Plot smoothed vs raw
        plot_file = os.path.join(self.plots_path, f"{self.environment}_{env_uid}_learning_curve.png")
        self.plot_learning_curve(df, plot_file, smoothing_window=10)
        logging.info(f"Saved learning curve plot to: {plot_file}")

    def run_experiments(self):
        """
        Orchestrates the order of experiments:
          1) Run PCMCI with 'random' policy (pre-training)
          2) Train the RL model
          3) Run PCMCI with 'trained' policy (post-training)
        """
        # 1) PCMCI with random policy
        #self.run_ci_tests(n_times=5, policy='random', label='random')
        self.run_ci_tests_fisher(n_times=15, policy='random', label='random')

        # 2) Train RL
        self.train()

        # 3) PCMCI with trained policy
        #self.run_ci_tests(n_times=5, policy='trained', label='trained')
        self.run_ci_tests_fisher(n_times=15, policy='random', label='random')

        return self.last_seed


def run(config_path="config.json", env_name=None):
    """
    Loads a config file, picks the environment,
    and runs random-policy CI tests, training, and trained-policy CI tests.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    if env_name is None:
        environment_config = config["environments"][0]
    else:
        environment_config = None
        for env_item in config["environments"]:
            if env_item["name"] == env_name:
                environment_config = env_item
                break
        if environment_config is None:
            raise ValueError(f"Could not find environment '{env_name}' in {config_path}")

    start_time = time.perf_counter()
    baseline = BaselineExperiments(environment_config["name"], environment_config)

    # Single call to orchestrate everything
    used_seed = baseline.run_experiments()

    end_time = time.perf_counter()
    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")
    return used_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline RL Experiments")
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Name of the environment to use from config.json.")
    args = parser.parse_args()

    run(config_path=args.config_path, env_name=args.env)