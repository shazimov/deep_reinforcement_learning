import torch
import numpy as np
import time

import sys

from collections import deque

def ddpg(env, agent, num_agents, average_score_solved=2000.0, n_episodes=50000):
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
    scores_deque = deque(maxlen=100) # last 100 scores
    score_per_episode_list = [] # list containing average scores from each episode

    total_timesteps = 0

    start_time = time.perf_counter()

    max_score = -1000
    max_score_timesteps = 0

    for i_episode in range(1, n_episodes+1):
        state = env.reset() # reset the environment
        agent.reset()
        score = 0
        timesteps = 0

        while timesteps < 1000:
            timesteps+=1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done) #take a step/train the agent
            state = next_state
            score += reward
            if done:
                break
        
        if score > max_score:
            max_score = score
            max_score_timesteps = timesteps

        total_timesteps += timesteps
        score_per_episode_list.append(score) # save most recent score
        scores_deque.append(score)
        mean_scores_deque = np.mean(scores_deque)
        #max_score = np.max(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tTimesteps: {:d}               '.format(i_episode, mean_scores_deque, score, timesteps), end='')
        if i_episode % 100 == 0:
            elapsed_time = time.perf_counter() - start_time
            print('\rEpisode {}\tAverage Score: \033[1m{:.2f}\033[0m\tMax: {:.2f}\tMax Timesteps: {:d}'.format(i_episode, mean_scores_deque, max_score, max_score_timesteps))
            print('Elapsed Time: {:d} minutes {:d} seconds\tTotal Timesteps: {:d}'.format(int(elapsed_time/60), int(elapsed_time%60), total_timesteps))
            max_score = -1000
            max_score_timesteps = 0

        if i_episode % 1000 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoints\\checkpoint_actor_local_'+str(i_episode)+'.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoints\\checkpoint_critic_local_'+str(i_episode)+'.pth')
            torch.save(agent.actor_target.state_dict(), 'checkpoints\\checkpoint_actor_target_'+str(i_episode)+'.pth')
            torch.save(agent.critic_target.state_dict(), 'checkpoints\\checkpoint_critic_target_'+str(i_episode)+'.pth')

        if mean_scores_deque >= average_score_solved:
            torch.save(agent.actor_local.state_dict(), 'checkpoints\\solved_checkpoint_actor_local.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoints\\solved_checkpoint_critic_local.pth')
            torch.save(agent.actor_target.state_dict(), 'checkpoints\\solved_checkpoint_actor_target.pth')
            torch.save(agent.critic_target.state_dict(), 'checkpoints\\solved_checkpoint_critic_target.pth')
            num_episodes_solved = i_episode - 100
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(num_episodes_solved, mean_scores_deque))
            break

    return score_per_episode_list, num_episodes_solved
