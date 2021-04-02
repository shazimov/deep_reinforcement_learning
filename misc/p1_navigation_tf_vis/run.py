from unityagents import UnityEnvironment
from agent import Agent
from deep_q_learning import dqn

# Initialize the Environment
env = UnityEnvironment(file_name="VisualBanana.app")

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Get the action size
action_size = brain.vector_action_space_size

# Get the state size
state_shape = env_info.visual_observations[0]

#Initialize the Agent with given hyperparameters

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.999           # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
seed = 42               # random seed

agent = Agent(state_shape=state_shape,
              action_size=action_size,
              buffer_size=BUFFER_SIZE,
              batch_size=BATCH_SIZE,
              gamma=GAMMA,
              tau=TAU,
              learning_rate=LR,
              update_every=UPDATE_EVERY,
              seed=seed)

# Train the agent with given epsilon hyperparameters

EPSILON_START = 0.9           #starting value of epsilon, for epsilon-greedy action selection
EPSILON_MIN = 0.01            #minimum value of epsilon
EPSILON_DECAY = 0.9           #epsilon decay factor
EPSILON_DECAY_DELAY = 10      #used to delay the decay of epsilon by a given number of episodes
AVERAGE_SCORE_SOLVED = 13.0   #average score needed (over 100 last episodes) to consider the environment as solved

scores, num_episodes_solved = dqn(env=env,
                                  agent=agent,
                                  average_score_solved=AVERAGE_SCORE_SOLVED,
                                  epsilon_start=EPSILON_START,
                                  epsilon_min=EPSILON_MIN,
                                  epsilon_decay=EPSILON_DECAY,
                                  epsilon_decay_delay=EPSILON_DECAY_DELAY)
