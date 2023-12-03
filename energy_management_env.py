import gym
from gym import spaces
import pandas as pd
import numpy as np

class EnergyManagementEnv(gym.Env):
    def __init__(self, SOC_min, SOC_max, E, lambda_val, data_path, initial_SOC=None):
        super(EnergyManagementEnv, self).__init__()

        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.E = E
        self.lambda_val = lambda_val
        self.data = self.load_data("Data_input_v2.csv")  # Use the provided data_path
        self.current_index = 0  # Initialize time step index
        self.initial_SOC = initial_SOC
        self.bt_old = 0.0  # Initialize bt_old

        # Define the action space as discrete: 0.1, 0, -0.1
        self.action_space = spaces.Discrete(3)

        # Define the state space (electricity demand, state of charge, price)
        self.observation_space = spaces.Box(
            low=np.array([0, SOC_min, 0], dtype=np.float32),
            high=np.array([np.inf, SOC_max, np.inf], dtype=np.float32)
        )

        # Set initial state
        if initial_SOC is not None:
            initial_demand = self.data['Demand'].iloc[0]
            initial_price = self.data['Price'].iloc[0]
            self.state = np.array([initial_demand, initial_SOC, initial_price], dtype=np.float32)
        else:
            self.state = self.get_initial_state()

    def load_data(self, data_path):
        # Load data from provided CSV file
        data = pd.read_csv("Data_input_v2.csv")
        return data

    def get_initial_state(self):
        # Get initial state from the first row of the dataframe
        initial_demand = self.data['Demand'].iloc[0]
        initial_price = self.data['Price'].iloc[0]
        initial_SOC = self.SOC_min  # Assuming SOC starts at the minimum value
        return np.array([initial_demand, initial_SOC, initial_price])

    def step(self, action):
        # Map the discrete actions to actual values: 0 -> 0, 1 -> 0.1, 2 -> -0.1
        discrete_actions = [-0.1, 0, 0.1]
        bt = discrete_actions[action]

        # Unpack state variables
        dt, SOC_t, pt = self.state

        # Calculate grid energy and new state of charge
        gt = dt + bt
        SOC_next = SOC_t + bt

        # Calculate reward
        if self.SOC_min <= SOC_next <= self.SOC_max:
            reward = -(pt * 1e-2 * (dt + bt * self.E)) 
        elif self.SOC_min > SOC_next:
            reward = -50e4 * (self.SOC_min - SOC_next) ** 2 - 1e4
        else:
            reward = -50e4 * (self.SOC_max - SOC_next) ** 2 - 1e4

        # Increment time step index
        self.current_index += 1

        if self.current_index == len(self.data):
            self.current_index = 0

        # Extract demand and price from the dataset for the next step
        dt = self.data['Demand'].iloc[self.current_index]
        pt = self.data['Price'].iloc[self.current_index]

        # Update state
        self.state = np.array([dt, SOC_next, pt])
        self.bt_old = bt  # Store bt as bt_old for the next step

        return self.state, reward, False, {}

    def reset(self):
        # Set the initial state based on the first time step in the dataframe
        if self.initial_SOC is not None:
            initial_demand = self.data['Demand'].iloc[0]
            initial_price = self.data['Price'].iloc[0]
            self.state = np.array([initial_demand, self.initial_SOC, initial_price])
        else:
            self.state = self.get_initial_state()
        self.current_index = 0  # Reset time step index
        self.bt_old = 0.0  # Reset bt_old
        return self.state

    def render(self, mode='human'):
        # Implement rendering if needed
        pass

    def close(self):
        # Implement any cleanup if needed
        pass

def energy_management_env_creator(SOC_min, SOC_max, E, lambda_val, data_path, initial_SOC=None):
    return EnergyManagementEnv(SOC_min, SOC_max, E, lambda_val, data_path, initial_SOC)
