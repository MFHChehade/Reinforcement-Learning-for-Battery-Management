# Reinforcement Learning for Energy Management

This repository contains a collection of components designed for reinforcement learning (RL) experiments in an energy management domain. The project is structured to facilitate the development, training, and evaluation of RL agents within a custom energy management environment. Below is an overview of the primary components and their functionalities.

## Data Inputs

- `Data_input.csv` & `Data_input_v2.csv`: Core datasets designed for reinforcement learning experiments within the energy management domain. These files contain time-series data essential for simulating and navigating energy system complexities. Each dataset includes the following columns:

  - **Demand**
  - **Price**
  - **Day**
  - **Month**
  - **Hour**

This structured input supports the dynamic interaction between RL agents and the custom energy management environment, enabling the development of optimized energy usage and cost strategies.

## RL Monitoring Utilities

- `plotting.py`: A utility script for visualizing training metrics, including rewards, losses, and other relevant performance indicators over time.
- `reward_callback.py`: Contains callback functions designed to compute or adjust rewards based on specific conditions during the training process.
- `vec_monitor.py`: Implements monitoring and logging for vectorized environments, facilitating detailed analysis of agent performance and environment states.

## Agents

- `agent.py`: Defines the RL agent's architecture, including the policy network, learning algorithms, and any additional logic for decision-making.
- `agents_main.py`: The main script used to kickstart the training or evaluation process. It orchestrates the interaction between agents and the environment, handling the training loop, logging, and any necessary pre/post-processing.

## Environment

- `energy_management_env.py`: Implements the custom energy management environment. It simulates the dynamics of energy systems, handling state transitions, reward calculations, and any interactions an agent can have within this domain.
- `env_registration.py`: Facilitates the registration of the custom environment with the RL framework being used, ensuring compatibility and ease of use within standard training routines.

## Problem Formulations

- `problem_formulations/RL Formulation - Battery Problem.pdf`: This document provides a detailed mathematical formulation of the battery optimization problem addressed by the RL agents in this project. It includes the definition of the state space, action space, reward function, transition dynamics, and constraints, offering a theoretical foundation for understanding and improving the RL algorithms used for energy management.

## Getting Started

To get started with training an agent in the energy management environment, ensure you have the required dependencies installed. Then, you can run the `agents_main.py` script to begin the training process. Make sure to adjust the data input paths and any hyperparameters according to your experimental setup.

### Installation

Briefly outline the installation process, including installing any dependencies or necessary libraries.

```bash
pip install -r requirements.txt
