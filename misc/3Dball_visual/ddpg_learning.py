import torch
import numpy as np
import sys
import time 

from collections import deque

from collections import Mapping, Container
from sys import getsizeof

def ddpg(env, agent, num_agents, average_score_solved=2000.0, n_episodes=10000):
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
        state = process_frames(state)

        agent.reset()
        score = 0
        timesteps = 0

        while True:
            timesteps+=1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = process_frames(next_state)
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

def scale_lumininance(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

def normalize(img):
    return img / 255

def process_frames(frames):

    out = []
    for i in range(4):
        a = i*3
        b = 3+i*3
        f = scale_lumininance(frames[:, :, a:b])
        f = normalize(f)
        out.append(f)

    out = np.stack(out, axis=0)
    out = np.expand_dims(out, axis=0)

    return out
 
def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
 
    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.
 
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
 
    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0
 
    r = getsizeof(o)
    ids.add(id(o))
 
    if isinstance(o, str):
        return r
 
    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
 
    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)
 
    return r 