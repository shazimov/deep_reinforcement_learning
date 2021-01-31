import time
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise

UPDATES_PER_TIME_STEP = 1

class MultiAgent():
    def __init__(self, state_size, action_size, num_agents, buffer_size, batch_size, gamma, tau, learning_rate_actor, learning_rate_critic, weight_decay, device, update_every=1, random_seed=42):
        """Initialize an Multi Agent object (MADDPG).

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (list): number of agents acting in the environment
            buffer_size (int): size of the replay buffer
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): used for soft update of target parameters
            learning_rate_actor (float): learning rate for the actor
            learning_rate_critic (float): learning rate for the critic
            weight_decay (float): weight decay for the optimizers
            device (torch.Device): pytorch device
            update_every (int): how many time steps between network updates
            random_seed (int): random seed
        """
        self.action_size = action_size
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = ReplayBuffer(num_agents, action_size, buffer_size, batch_size, device=device, seed=random_seed)
        self.agents = []
        for i in range(num_agents):
            self.agents.append(
                Agent(num_agents=num_agents,
                state_size=state_size,
                action_size=action_size,
                gamma=gamma,
                tau=tau,
                learning_rate_actor=learning_rate_actor,
                learning_rate_critic=learning_rate_critic,
                weight_decay=weight_decay,
                device=device,
                random_seed=random_seed))

        self.update_every = update_every

        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                #for z in range(UPDATES_PER_TIME_STEP):
                experiences = self.memory.sample()
                for i,agent in enumerate(self.agents):
                    states, actions, rewards, next_states, dones = experiences
                    agent.learn(i, experiences, self.gamma)

    def act(self, states, epsilon=0, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = np.zeros([self.num_agents, self.action_size])
        for i,agent in enumerate(self.agents):
            actions[i,:] = agent.act(states[i], epsilon, add_noise = add_noise)
        return actions

    def reset(self):
        """Resets the noise"""
        for agent in self.agents:
            agent.reset()

    def save_checkpoints(self):
        """Save the actor and critic parameters as checkpoints"""
        for i,agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent' + str(i) + 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'agent' + str(i) + 'checkpoint_critic.pth')

    def load_actor_checkpoints(self):
        """Load the checkpoints for the actor parameters"""
        for i,agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('agent' + str(i) + 'checkpoint_actor.pth'))


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size, gamma, tau, learning_rate_actor, learning_rate_critic, weight_decay, device, random_seed=42):
        """Initialize an Agent object (used my MultiAgent for MADDPG).

        Params
        ======
            num_agents (list): number of agents acting in the environment
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            gamma (float): discount factor
            tau (float): used for soft update of target parameters
            learning_rate_actor (float): learning rate for the actor
            learning_rate_critic (float): learning rate for the critic
            weight_decay (float): weight decay for the optimizers
            device (torch.Device): pytorch device
            random_seed (int): random seed
        """

        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate_actor, weight_decay=weight_decay)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(num_agents, state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(num_agents, state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=learning_rate_critic, weight_decay=weight_decay)#0.0001

        # Noise process
        self.noise = OUNoise(size=action_size, seed=random_seed)

        self.timestep = 0

    def act(self, state, epsilon=1, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * epsilon
        return np.clip(action, -1, 1)

    def reset(self):
        """Resets the noise"""
        self.noise.reset()

    def learn(self, index, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            index (int): Index of the current agent
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        all_states = torch.cat(states, dim=1).to(self.device)
        all_next_states = torch.cat(next_states, dim=1).to(self.device)
        all_actions = torch.cat(actions, dim=1).to(self.device)

        actions_next = actions.copy()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next[index] = self.actor_target(next_states[index])
        all_actions_next = torch.cat(actions_next, dim=1).to(self.device)
        Q_targets_next = self.critic_target(all_next_states, all_actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards[index] + (gamma * Q_targets_next * (1 - dones[index]))
        # Compute critic loss
        Q_expected = self.critic_local(all_states, all_actions)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(Q_expected, Q_targets.detach())
        #critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = actions.copy()
        actions_pred[index] = self.actor_local(states[index])
        all_actions_pred = torch.cat(actions_pred, dim=1).to(self.device)
        actor_loss = -self.critic_local(all_states, all_actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

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
