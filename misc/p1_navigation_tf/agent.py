import numpy as np
import random
import tensorflow as tf

from model import QNetwork
from replay_buffer import ReplayBuffer

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, tau, learning_rate, update_every, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): used for soft update of target parameters
            learning_rate (float): learning rate
            update_every (int): how many steps between network updates
            device (torch.Device): pytorch device
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(action_size)
        self.qnetwork_target = QNetwork(action_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Adds new experience to the replay buffer and learns from a subset of memories.

        Params
        ======
        state (array_like): initial state
        action (int): chosen action
        reward (int): reward given
        next_state (array_like): next state
        done (bool): True if the episode is finished
        """

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()

                self.learn(experiences)

    def act(self, state, epsilon=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """
        state = tf.expand_dims(state, axis=0)
        action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states)
        Q_targets_next = tf.math.reduce_max(Q_targets_next, axis=1)
        Q_targets_next = tf.expand_dims(Q_targets_next, axis=1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        with tf.GradientTape() as tape:
            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states)
            Q_expected = tf.gather(Q_expected, indices=actions, axis=1, batch_dims=1)
            loss = self.loss_fn(y_true=Q_targets, y_pred=Q_expected)

        gradients = tape.gradient(loss, self.qnetwork_local.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.qnetwork_local.trainable_weights))

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """

        list_of_weights = []
        for i in range(len(target_model.trainable_weights)):
            list_of_weights.append(self.tau * local_model.trainable_weights[i] + (1.0 - self.tau) * target_model.trainable_weights[i])

        target_model.set_weights(list_of_weights)
