import numpy as np
import random

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
    step_count = 0

    update_target_every = 5000
    frames_per_state = 4

    num_episodes_solved = 0
    brain_name = env.brain_names[0]
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    epsilon = epsilon_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        action = random.choice(np.arange(4))

        state_buffer = deque(maxlen=frames_per_state)
        next_state_buffer = deque(maxlen=frames_per_state)
        for i in range(frames_per_state):
            env_info = env.step(action)[brain_name]
            cur_state = env_info.visual_observations[0]  # get the current state
            state_buffer.append(cur_state)
            next_state_buffer.append(cur_state)

        state = process_frames(state_buffer)

        score = 0
        while True:
            step_count += 1
            action = agent.act(state, epsilon)

            env_info = env.step(action)[brain_name]
            cur_state = env_info.visual_observations[0]   # get the next state

            next_state_buffer.append(cur_state)
            next_state = process_frames(next_state_buffer)

            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)

            if step_count % update_target_every == 0:
                agent.hard_update()

            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        if i_episode > epsilon_decay_delay:
            epsilon = max(epsilon_min, epsilon_decay*epsilon) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:2f}'.format(i_episode, np.mean(scores_window), epsilon), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=average_score_solved:
            num_episodes_solved = i_episode - 100
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(num_episodes_solved, np.mean(scores_window)))
            agent.qnetwork_local.save_weights('./checkpoints/qnetwork_local.ckpt')
            break
    return scores, num_episodes_solved


def scale_lumininance(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

def normalize(img):
    return img / 255

def process_frames(frames):
    #x = scale_lumininance(frames)
    #x = normalize(x)
    #x = tf.expand_dims(x, axis=-1)
    f = frames.copy()

    for i in range(len(f)):
        f[i] = scale_lumininance(f[i])
        f[i] = normalize(f[i])

    out = np.stack(f, axis=-1)
    return out
