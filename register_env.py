from gym.envs.registration import register
from energy_management_env import EnergyManagementEnv

# Register the environment with Gym
register(
    id='EnergyManagement-v0',
    entry_point='energy_management_env:energy_management_env_creator',
    kwargs={
        'SOC_min': 0.2,
        'SOC_max': 0.8,
        'E': 1000,
        'lambda_val': 0.1,
        'data_path': 'Data_input.xlsx',
        'initial_SOC': 0.5  # Set to None if not using an initial_SOC
    },
)
