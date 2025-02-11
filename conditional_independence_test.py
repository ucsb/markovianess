import os
import time
import numpy as np

from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

def get_markov_violation_score(p_matrix, val_matrix, alpha_level=0.05):
    """
    Computes a dimension-aware Markov violation score by summing
        |val_matrix[i,j,k]| * ( -ln( p_matrix[i,j,k] ) )
    over all significant links whose lag >= 2 (k >= 2),
    then normalizing by N^2 * (tau_max - 1).

    The shape of p_matrix and val_matrix is [N, N, tau_max+1].
      - N = number of variables
      - tau_max+1 = # of lags from 0..tau_max

    Specifically:
    score = sum_{i,j,k >=2, p_value <= alpha_level} (|val[i,j,k]| * -ln(p[i,j,k]))
            / [N^2 * (tau_max - 1)]

    If tau_max <= 1, we return 0.0 (no "beyond first-order" lags).
    """

    N, _, lag_count = p_matrix.shape
    # tau_max is lag_count - 1
    tau_max = lag_count - 1

    if tau_max <= 1:
        return 0.0

    raw_sum = 0.0
    for i in range(N):
        for j in range(N):
            for k in range(2, lag_count):  # lags 2..tau_max
                if p_matrix[i, j, k] <= alpha_level:
                    raw_sum += abs(val_matrix[i, j, k]) * -np.log(p_matrix[i, j, k])

    denom = (N**2) * (tau_max - 1)
    if denom == 0:
        return 0.0

    return raw_sum / denom


class ConditionalIndependenceTest:
    """
    A class to run conditional independence tests (via PCMCI, using ParCorr)
    on time-series observations, and automatically save results.
    """
    def __init__(self):
        """
        Always using ParCorr for simplicity.
        """
        self.results = None  # Will store the PCMCI results dictionary

    def run_pcmci(
        self,
        observations: np.ndarray,
        tau_min: int = 0,
        tau_max: int = 5,
        alpha_level: float = 0.05,
        pc_alpha = None,
        env_id: str = None,
        label: str = None,
        results_dir: str = "results/pcmci"
    ):
        """
        Runs PCMCI with ParCorr, prints significant links, and saves the results to an .npz file.

        Args:
            observations (np.ndarray): 2D array of shape (T, num_variables).
            tau_min (int): Minimum lag for PCMCI.
            tau_max (int): Maximum lag for PCMCI.
            alpha_level (float): Significance level for printing links.
            pc_alpha (float or None): alpha cutoff for the PC step (optional).
            env_id (str): Optional environment ID for naming output files.
            label (str): Optional string label to include in filenames.
            results_dir (str): Directory to store the .npz result file.

        Returns:
            dict : The PCMCI results dictionary (with val_matrix, p_matrix, etc.).
        """
        if not isinstance(observations, np.ndarray):
            raise TypeError("Observations must be a NumPy array.")
        if observations.ndim != 2:
            raise ValueError("Observations array should be 2D: (time, variables).")

        dataframe = DataFrame(data=observations)

        cond_ind_test = ParCorr()
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test
        )

        start_time = time.time()
        self.results = pcmci.run_pcmci(
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha
        )
        end_time = time.time()

        val_matrix = self.results['val_matrix']
        p_matrix = self.results['p_matrix']

        # Print significant links for these particular observations
        num_vars = observations.shape[1]
        var_names = [f"X{i}" for i in range(num_vars)]

        print(f"\n[ConditionalIndependenceTest] PCMCI results with alpha={alpha_level}:")
        print(f"Variable names: {var_names}")
        pcmci.print_significant_links(
            p_matrix=p_matrix,
            val_matrix=val_matrix,
            alpha_level=alpha_level,
        )

        # Ensure the results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Construct a filename with env_id, label, tau range, etc.
        filename_parts = []
        if env_id:
            filename_parts.append(env_id)
        if label:
            filename_parts.append(label)
        filename_parts.append("ParCorr")
        tau_str = f"tau_{tau_min}-{tau_max}"
        filename_parts.append(tau_str)

        filename = "_".join(filename_parts) + ".npz"
        filepath = os.path.join(results_dir, filename)

        # Save the intermediate PCMCI results for these observations
        np.savez_compressed(
            filepath,
            val_matrix=val_matrix,
            p_matrix=p_matrix,
            graph=self.results.get('graph'),
            conf_matrix=self.results.get('conf_matrix'),
            tau_min=tau_min,
            tau_max=tau_max,
            alpha_level=alpha_level,
            pc_alpha=pc_alpha
        )
        print(f"[ConditionalIndependenceTest] Results saved to '{filepath}'")
        print(f"Time taken: {end_time - start_time:.2f}s\n")

        return self.results