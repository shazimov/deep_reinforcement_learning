import torch
import numpy as np

from collections import deque

def ddpg(env, agent, num_agents, average_score_solved=30.0, n_episodes=1000):
    """Deep Deterministic Policy Gradient Algorithm. Exits once the environment is considered solved.
    Returns the scores per episode and total number of episodes to solve the environment.

    Params
    ======
        env: environment to use
        agent: agent to use
        num_agents (int): number of agents in the environment
        average_score_solved (float): average score needed (over last 100 episodes) to consider the environment solved
        n_episodes (int): maximum number of training episodes
    """

    num_episodes_solved = 0
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=100) # last 100 scores
    mean_scores_per_episode_list = [] # list containing average scores from each episode
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations            # get the current state

        agent.reset()
        scores = np.zeros(num_agents)
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones) #take a step/train the agent
            states = next_states
            scores += rewards
            if np.any(dones): # exit the loop when at least one agent is done
                break

        mean_scores = np.mean(scores)
        mean_scores_per_episode_list.append(mean_scores) # save most recent score
        scores_deque.append(mean_scores)
        mean_scores_deque = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, mean_scores_deque, mean_scores), end="")
        if i_episode % 50 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_scores_deque))

        if mean_scores_deque >= average_score_solved:
            torch.save(agent.actor_local.state_dict(), 'solved_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'solved_checkpoint_critic.pth')
            num_episodes_solved = i_episode - 100
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(num_episodes_solved, mean_scores_deque))
            break

    return mean_scores_per_episode_list, num_episodes_solved
