import time
import torch
import numpy as np

from collections import deque

def maddpg(env, agent, num_agents, average_score_solved=0.5, n_episodes=10000, epsilon=1.0, epsilon_decay=1.0):
    """Multi Agent Deep Deterministic Policy Gradient Algorithm. Exits once the environment is considered solved.
    Returns the scores per episode, rolling average of the scores and total number of episodes to solve the environment.

    Params
    ======
        env: environment to use
        agent: agent to use
        num_agents (int): number of agents in the environment
        average_score_solved (float): average score needed (over last 100 episodes) to consider the environment solved
        n_episodes (int): maximum number of training episodes
        epsilon (float): for scaling the noise added to the actions (exploration)
        epsilon_decay (float): for decaying the noise during training
    """
    total_timesteps = 0

    num_episodes_solved = 0
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=100) # last 100 scores
    max_score_per_episode_list = [] # list containing average scores from each episode
    rolling_average = []

    start_time = time.perf_counter()

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations            # get the current state

        agent.reset()
        scores = np.zeros(num_agents)

        timesteps = 0

        while True:
            timesteps+=1
            actions = agent.act(states, epsilon)
            epsilon *= epsilon_decay
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones) #take a step/train the agent
            states = next_states
            scores += rewards
            if np.any(dones): # exit the loop when at least one agent is done
                break

        max_score = np.max(scores)
        max_score_per_episode_list.append(max_score) # save most recent score
        scores_deque.append(max_score)
        mean_scores_deque = np.mean(scores_deque)
        rolling_average.append(mean_scores_deque)

        total_timesteps += timesteps

        print('\rEpisode {}\tAverage Score: {:.4f}\tScores: [ {:.2f} | {:.2f} ]'.format(i_episode, mean_scores_deque, scores[0], scores[1]), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_scores_deque))
            print("total timesteps: " + str(total_timesteps))
            print("epsilon: " + str(epsilon))

        if mean_scores_deque >= average_score_solved:
            elapsed_time = time.perf_counter() - start_time
            agent.save_checkpoints()
            num_episodes_solved = i_episode - 100
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTotal Time: {:d} minutes {:d} seconds'.format(num_episodes_solved, mean_scores_deque, int(elapsed_time/60), int(elapsed_time%60)))
            break

    return max_score_per_episode_list, rolling_average, num_episodes_solved
