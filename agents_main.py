# Import necessary libraries
from environments.energy_management_env import EnergyManagementEnv
from rl_monitoring_utils.plotting import plot_results
from agents.agent import Agent
from environments.env_registration import register_env
from agents.agent import Agent

# Register the custom environment
env_params = {
    'SOC_min': 0.2,
    'SOC_max': 0.8,
    'E': 1000,
    'lambda_val': 0.1,
    'data_path': 'data/Data_input.csv',
    'initial_SOC': 0.5  # Set to None if not using an initial_SOC
}
register_env('EnergyManagement-v0', 'environments.env_registration:environment_creator', {'environment_class': EnergyManagementEnv, **env_params})

# Define environment ID
env_id = 'EnergyManagement-v0'

# Define total timesteps and number of environments for training
total_timesteps = 2.5e6
num_envs = 8

# Create a dictionary to store agents with their names
agents = {
    'ppo_agent': Agent(env_id, 'ppo', num_envs=num_envs),
    'trpo_agent': Agent(env_id, 'trpo', num_envs=num_envs),
    'dqn_agent': Agent(env_id, 'dqn', num_envs=num_envs)
}

# Train agents
for agent_name, agent in agents.items():
    agent.train(total_timesteps=total_timesteps)

# Plot results for all agents
log_dirs = [agent.get_log_dir() for agent in agents.values()]
plot_results(log_dirs)
