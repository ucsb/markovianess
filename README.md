# Markovianess Experiments

This repository contains a collection of experiments and utilities to investigate the “Markovian” property of Reinforcement Learning (RL) environments. Specifically, the project measures how modifications to the observations and actions (like dropping dimensions, adding noise, or artificially introducing dependencies) affect the Markov property. We quantify these effects using [PCMCI](https://github.com/jakobrunge/tigramite) to check for conditional independence at various time lags.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
   - [Top-Level Scripts](#top-level-scripts)
   - [Experiments Package](#experiments-package)
   - [Conditional Independence (CI) Utilities](#conditional-independence-ci-utilities)
   - [Tests](#tests)
   - [Other Files](#other-files)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Configuration File](#configuration-file)
   - [Command-Line Examples](#command-line-examples)
5. [Key Experiments](#key-experiments)
   - [Baseline](#baseline)
   - [Dimension Dropping](#dimension-dropping)
   - [Noised Observations (Gaussian, AR, etc.)](#noised-observations-gaussian-ar-etc)
   - [Noised Actions (Gaussian, AR, etc.)](#noised-actions-gaussian-ar-etc)
6. [Results Directory Structure](#results-directory-structure)
7. [License](#license)
8. [References](#references)

---

## Overview

This project aims to:
- Train RL agents (with [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)) on standard environments like `CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`.
- Apply various modifications (noise injection, dimension dropping, etc.) to see if and how the environment stops being fully Markovian.
- Collect rollouts and run the [Tigramite PCMCI](https://github.com/jakobrunge/tigramite) algorithm to detect conditional independences across different time lags.
- Compute a *Markov violation score* based on the partial correlations and p-values from PCMCI.

---
### Top-Level Scripts

- **`main.py`**  
  Orchestrates all experiments by reading a configuration file (`config.json`) to determine which environments to run, how many timesteps, which types of noise, etc. 
  - Moves old results into `backup/<timestamp>` automatically.
  - Runs each environment sequentially for baseline, then noise injections, etc.

### Experiments Package

Inside `markovianess/experiments/` you will find Python modules that implement different experiment types:

- **`baseline.py`**  
  Trains a standard PPO model with no modifications and computes a Markov violation score.

- **`dropped.py`**  
  Drops one observation dimension at a time, retrains an agent, and measures how performance and Markov property are affected.

- **`noised_gaussian.py`** / **`noised_gaussian_action.py`**  
  Adds Gaussian noise to observations (or to actions) and measures changes in performance and Markov violation.

- **`noised_auto_regressive.py`** / **`noised_action_auto_regressive.py`**  
  Introduces AR(p) noise into observations (or actions). Optionally uses a learned transition model to partially override the environment’s next state.

Each experiment script has a main function to be called either standalone or from `main.py`.

### Conditional Independence (CI) Utilities

- **`ci/conditional_independence_test.py`**  
  - Contains the `ConditionalIndependenceTest` class, which wraps Tigramite’s PCMCI + ParCorr.  
  - Provides a helper function `get_markov_violation_score()` that processes the output of PCMCI (partial correlations & p-values) to produce a single numeric “violation” measure.  

### Tests

Under `tests/`, there are example scripts (`pendulum_drop_experiment.py`, `cartpole_drop_experiment.py`, `acrobot_drop_experiment.py`) showing how to systematically drop observation dimensions and measure the effect on performance.

### Other Files

- **`utils.py`**  
  Contains shared logging logic, smoothing functions, or any other global utility.  
- **`temp.py`**  
  A placeholder or scratch file, you may ignore or remove if not needed.  

---

## Installation

1. **Clone** this repository:

   ```bash
   git clone https://github.com/your_username/markovianess.git
   cd markovianess
    ```
2. **Create a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```
4. Run main.py to check if everything is working:

    ```bash
    python main.py
    ```