import torch
import numpy as np

from collections import deque

def dqn(env, agent, n_episodes=10000, average_score_solved=13.0, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995, epsilon_decay_delay=0):
    """Deep Q-Learning algorithm. Exits once the environment is considered solved.
    Returns the scores per episode and total number of episodes to solve the environment.

    Params
    ======
        env: environment to use
        agent: agent to use
        n_episodes (int): maximum number of training episodes
        average_score_solved (float): average score needed (over last 100 episodes) to consider the environment solved
        epsilon_start (float): starting value of epsilon, for epsilon-greedy action selection
        epsilon_min (float): minimum value of epsilon
        epsilon_decay (float): multiplicative factor (per episode) for decreasing epsilon
        epsilon_decay_delay (int): used to delay the decay of epsilon by a given number of episodes
    """

    num_episodes_solved = 0
    brain_name = env.brain_names[0]
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    epsilon = epsilon_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state

        score = 0
        while True:
            action = agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        if i_episode > epsilon_decay_delay:
            epsilon = max(epsilon_min, epsilon_decay*epsilon) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=average_score_solved:
            num_episodes_solved = i_episode - 100
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(num_episodes_solved, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores, num_episodes_solved
