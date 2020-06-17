import argh
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from argh import arg

from unityagents import UnityEnvironment
from agent import Agent


def plot_figure(scores):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def maddpg(agent, env, brain_name, n_episodes=1000, max_t=1000, train=True,
           print_every=100, n_agents=2):
    """DDPG

    Params
    ======
    n_episodes (int): maximum number of training episodes
    max_t (int): maximum number of timesteps per episode
    brain_name (str): brain from where to fetch env details
    agent (obj): instance of a maddpg agent
    """
    max_scores = []
    max_score_window = deque(maxlen=print_every)
    best_score = -np.Inf

    pbar = tqdm(total=n_episodes)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(n_agents)
        agent.reset()
        for t in range(max_t):
            actions = np.array([np.squeeze(agent.act(states[s]))
                                for s in range(n_agents)])
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            if train:
                for i in range(n_agents):
                    agent.step(states[i], actions[i], rewards[i],
                               next_states[i], dones[i])
            scores += rewards
            states = next_states
            if np.any(dones):
                break

        max_score = np.max(scores)
        max_score_window.append(max_score)
        max_scores.append(max_score)

        if max_score > best_score:
            best_score = max_score

        if not train:
            pbar.write('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(max_score_window)))

        if i_episode % print_every == 0:
            pbar.write('Episode {}\tAverage Score: {:.2f}\tMax: {:.1f}'.format(
                i_episode, np.mean(max_score_window),
                np.max(max_score_window)))

        if train and np.mean(max_score_window) > 0.50:
            print('\nEnv solved in {:d} episodes!\tAverage Score: {:.2f}'
                  .format(i_episode-100, np.mean(max_score_window)))
            torch.save(agent.actor_local.state_dict(),
                       'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       'checkpoint_critic.pth')

        pbar.update()

    plot_figure(scores)

    return scores


def play(agent, env):

    state = env.reset()
    for t in range(200):
        action = agent.act(state, add_noise=False)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break


@arg('--chkp_actor', '--chkp_critic',
     help='Checkpoint file. If present, agent will run in eval mode.')
def runner(chkp_actor=None, chkp_critic=None):
    '''
    This function loads the environment and the agent. By default runs in
    training mode, but if a checkpoint file is passed it runs in eval mode.
    Params
    ======
        chkp_actor (None|file):
            file containing an actor checkpoint saved during training.
        chkp_critic (None|file):
            file containing a critic checkpoint saved during training.
    '''
    # instantiate Unity environment
    env = UnityEnvironment(file_name='./Tennis_Linux/Tennis.x86_64')
    # get first brain
    brain_name = env.brain_names[0]
    # get action size
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    # get state size
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    state_size = states.shape[1]
    # instantiate the Agent
    agent = Agent(state_size=state_size,
                  action_size=action_size, random_seed=2)

    if chkp_actor:
        cp_actor = torch.load(chkp_actor)
        cp_critic = torch.load(chkp_critic)
        agent.actor_local.load_state_dict(cp_actor)
        agent.critic_local.load_state_dict(cp_critic)
        maddpg(agent, env, brain_name, n_episodes=100, train=False)

    else:
        maddpg(agent, env, brain_name, train=True)

    env.close()


if __name__ == '__main__':
    argh.dispatch_command(runner)
