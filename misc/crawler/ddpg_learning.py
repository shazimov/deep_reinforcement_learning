import torch
import numpy as np

from collections import deque

def ddpg(env, agent, num_agents, average_score_solved=2000.0, n_episodes=10000, epsilon=1.0, epsilon_decay=1.0, epsilon_min=0.01):
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

    timesteps = 0

    for i_episode in range(1, n_episodes+1):
        state = env.reset() # reset the environment
        agent.reset()
        score = 0
        while True:
            timesteps+=1
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done) #take a step/train the agent
            state = next_state
            score += reward
            if done:
                break

        epsilon = max(epsilon_min, epsilon_decay*epsilon)

        score_per_episode_list.append(score) # save most recent score
        scores_deque.append(score)
        mean_scores_deque = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, mean_scores_deque, score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_scores_deque))
            print("total timesteps: " + str(timesteps))
            print("epsilon: " + str(epsilon))

        if mean_scores_deque >= average_score_solved:
            torch.save(agent.actor_local.state_dict(), 'solved_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'solved_checkpoint_critic.pth')
            num_episodes_solved = i_episode - 100
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(num_episodes_solved, mean_scores_deque))
            break

    return score_per_episode_list, num_episodes_solved
