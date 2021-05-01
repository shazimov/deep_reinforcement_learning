import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, buffer_size, batch_size, gamma, tau, beta, beta_increment, learning_rate_actor, learning_rate_critic, device, update_every=1, random_seed=42):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents acting in the environment
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): used for soft update of target parameters
            learning_rate_actor (float): learning rate for the actor
            learning_rate_critic (float): learning rate for the critic
            device (torch.Device): pytorch device
            update_every (int): how many time steps between network updates
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.update_every = update_every
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=learning_rate_critic, weight_decay=0)

        # Noise process
        self.noise = OUNoise(size=action_size, seed=random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, beta, beta_increment, device=device, seed=random_seed)

        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        next_state_torch = torch.from_numpy(next_state).float().to(self.device)
        reward_torch = torch.from_numpy(np.array(reward)).float().to(self.device)
        done_torch = torch.from_numpy(np.array(done).astype(np.uint8)).float().to(self.device)
        state_torch = torch.from_numpy(state).float().to(self.device)
        action_torch = torch.from_numpy(action).float().to(self.device)
        
        self.actor_target.eval()
        self.critic_target.eval()
        self.critic_local.eval()
        with torch.no_grad():
            action_next = self.actor_target(next_state_torch)
            Q_target_next = self.critic_target(next_state_torch, action_next)
            Q_target = reward_torch + (self.gamma * Q_target_next * (1 - done_torch))
            Q_expected = self.critic_local(state_torch, action_torch)
        self.actor_local.train()
        self.critic_target.train()
        self.critic_local.train()
        
        #Error used in prioritized replay buffer
        error = (Q_expected - Q_target).squeeze().cpu().data.numpy()
        
        #Adding experiences to prioritized replay buffer
        #for i in np.arange(len(reward)):
        self.memory.add(error, state, action, reward, next_state, done)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        self.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences, idxs, is_weights = self.memory.sample()
                self.learn(experiences, idxs, is_weights)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        #print("a: " + str(action))

        if add_noise:
            action += self.noise.sample()
            #print("a+n: " + str(action))
            #print("a+n+c: " + str(np.clip(action, -1, 1)))
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, idxs, is_weights):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = (torch.from_numpy(is_weights).float().to(self.device) * F.mse_loss(Q_expected, Q_targets)).mean()

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        #gradient clipping
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        #.......................update priorities in prioritized replay buffer.......#
        #Calculate errors used in prioritized replay buffer
        errors = (Q_expected-Q_targets).squeeze().cpu().data.numpy()
        
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
