import torch
import numpy as np

from collections import deque

def ddpg(env, agent, num_agents, average_score_solved=0.5, n_episodes=10000):
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

    total_timesteps = 0

    num_episodes_solved = 0
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=100) # last 100 scores
    max_score_per_episode_list = [] # list containing average scores from each episode
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations            # get the current state

        agent.reset()
        scores = np.zeros(num_agents)

        timesteps = 0

        while True:
            timesteps+=1
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

        max_score = np.max(scores)
        max_score_per_episode_list.append(max_score) # save most recent score
        scores_deque.append(max_score)
        mean_scores_deque = np.mean(scores_deque)

        total_timesteps += timesteps

        print('\rEpisode {}\tAverage Score: {:.2f}\tMax Score: {:.2f}'.format(i_episode, mean_scores_deque, max_score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_scores_deque))
            print("total timesteps: " + str(total_timesteps))


        #print("\ttimesteps: " + str(timesteps) + "/" +str(total_timesteps))
        #print("\tscores: " + str(scores))


        if mean_scores_deque >= average_score_solved:
            torch.save(agent.actor_local.state_dict(), 'solved_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'solved_checkpoint_critic.pth')
            num_episodes_solved = i_episode - 100
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(num_episodes_solved, mean_scores_deque))
            break

    return max_score_per_episode_list, num_episodes_solved
