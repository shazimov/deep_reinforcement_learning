import numpy as np
import matplotlib.pyplot as plt
import torch

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np

from agent import Agent
from ddpg_learning import ddpg

# Initialize the Environment
unity_env = UnityEnvironment(file_name="Visual3DBall\\UnityEnvironment.app")
env = UnityToGymWrapper(unity_env)

# Get the action size
action_size = 2

# Get the state size
state_shape = (84,84,12)

# Get number of agents
num_agents = 1


#Initialize the Agent with given hyperparameters

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64       # batch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 5e-4         # learning rate of the actor
LR_CRITIC = 5e-3        # learning rate of the critic
UPDATE_EVERY = 20        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Torch device to use

agent = Agent(state_shape=state_shape,
              action_size=action_size,
              num_agents=num_agents,
              buffer_size=BUFFER_SIZE,
              batch_size=BATCH_SIZE,
              gamma=GAMMA,
              tau=TAU,
              learning_rate_actor=LR_ACTOR,
              learning_rate_critic=LR_CRITIC,
              device=device,
              update_every=UPDATE_EVERY,
              random_seed=42)


# Train the agent

AVERAGE_SCORE_SOLVED=100

scores, num_episodes_solved = ddpg(env=env,
                                   agent=agent,
                                   num_agents=num_agents,
                                   average_score_solved=AVERAGE_SCORE_SOLVED)
