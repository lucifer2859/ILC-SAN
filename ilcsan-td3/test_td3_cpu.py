import numpy as np
import torch
import torch.nn as nn
import gym
import pickle
import math
import sys

from replay_buffer_norm import ReplayBuffer
from core_cuda import MLPActor


def test_mujoco_render(pi_model_file, mean_var_file, env_fn,
                       hidden_sizes=(256, 256), norm_clip_limit=3,
                       activation=nn.ReLU):
    # Set device
    device = torch.device("cpu")
    # Set environment
    test_env = env_fn()
    obs_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.shape[0]
    act_limit = test_env.action_space.high[0]

    # Replay buffer for running z-score norm
    b_mean_var = pickle.load(open(mean_var_file, "rb"))
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=1,
                                 clip_limit=norm_clip_limit, norm_update_every=1)
    replay_buffer.mean = b_mean_var[0]
    replay_buffer.var = b_mean_var[1]

    pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
    pi.load_state_dict(torch.load(pi_model_file))

    def get_action(o):
        a = pi(torch.as_tensor(o, dtype=torch.float32, device=device)).numpy()
        return np.clip(a, -act_limit, act_limit)

    # Start testing
    o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    while not (d or (ep_len == 1000)):
        # test_env.render()
        with torch.no_grad():
            o, r, d, _ = test_env.step(get_action(replay_buffer.normalize_obs(o)))
        ep_ret += r
        ep_len += 1

    return ep_ret


if __name__ == '__main__':
    import math
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--num_model', type=int, default=10)
    args = parser.parse_args()

    reward_list = []

    for model_idx in range(args.num_model):

        file_dir = './params/td3_td3-pi-' + args.env
        
        # Best epoch reward during training
        test_reward, _ = pickle.load(open(file_dir + '/model' + str(model_idx) + '_test_rewards.p', 'rb'))
        best_epoch_idx = 0
        best_epoch_reward = 0
        for idx in range(20):
            if test_reward[(idx + 1) * 5 - 1] > best_epoch_reward:
                best_epoch_reward = test_reward[(idx + 1) * 5 - 1]
                best_epoch_idx = (idx + 1) * 5

        best_rewards = [best_epoch_reward]

        # Test Reward (last epoch idx) 
        model_file_dir = file_dir + '/model' + str(model_idx) + '_e100.pt'
        buffer_file_dir = file_dir + '/model' + str(model_idx) + '_e100_mean_var.p'

        reward = test_mujoco_render(model_file_dir, 
                                    buffer_file_dir, 
                                    lambda : gym.make(args.env))

        best_rewards.append(reward)

        # Test Reward (best epoch idx) 
        model_file_dir = file_dir + '/model' + str(model_idx) + '_e' + str(best_epoch_idx) + '.pt'
        buffer_file_dir = file_dir + '/model' + str(model_idx) + '_e' + str(best_epoch_idx) + '_mean_var.p'

        reward = test_mujoco_render(model_file_dir, 
                                    buffer_file_dir, 
                                    lambda : gym.make(args.env))

        best_rewards.append(reward)

        print("Model", model_idx, ", Reward:", best_rewards)

        reward_list.append(max(best_rewards))

    print('------------------------------------------------')
    print('mean: ', np.mean(reward_list))
    print('std: ', np.std(reward_list, ddof=1))