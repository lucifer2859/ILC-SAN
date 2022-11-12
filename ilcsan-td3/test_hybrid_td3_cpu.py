import numpy as np
import torch
import torch.nn as nn
import gym
import pickle
import math
import sys

from replay_buffer_norm import ReplayBuffer
from popsan import PopSpikeActor
from mdcsan import MDCSpikeActor


def test_mujoco_render(popsan_model_file, mean_var_file, env_fn,
                       encoder_pop_dim, decoder_pop_dim, mean_range, std, 
                       spike_ts, encode, decode, v_decay, neurons, connections,
                       hidden_sizes=(256, 256), norm_clip_limit=3):
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

    # SAN
    if 'DN' in neurons:
        popsan = MDCSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
                               mean_range, std, spike_ts, act_limit, device, encode, decode, v_decay, 
                               neurons, connections)
    else:
        popsan = PopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
                               mean_range, std, spike_ts, act_limit, device, encode, decode, v_decay,
                               neurons, connections)
    
    popsan.load_state_dict(torch.load(popsan_model_file))

    def get_action(o):
        a = popsan(torch.as_tensor(o, dtype=torch.float32, device=device), 1).numpy()
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
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    parser.add_argument('--num_model', type=int, default=10)
    parser.add_argument('--encode', type=str, default='pop-det', choices=['pop-det', 'pop', 'layer'])
    parser.add_argument('--decode', type=str, default='last-mem', choices=['fr-mlp', 'last-mem', 'max-mem', 'mean-mem'])
    parser.add_argument('--v_decay', type=float, default=1.0)
    parser.add_argument('--neurons', type=str, default='LIF', choices=['LIF', 'DN'])
    parser.add_argument('--connections', default='intra', choices=['none', 'intra', 'no-bias', 'no-self', 'no-lateral', 'bias', 'self', 'lateral'])
    args = parser.parse_args()
    
    mean_range = (-3, 3)
    std = math.sqrt(args.encoder_var)
    spike_ts = 5

    if 'pop' in args.encode:
        file_dir = './params/hybrid-td3_td3-ilcsan-' + args.env + '-encoder-dim-' + str(args.encoder_pop_dim) + '-decoder-dim-' + \
                   str(args.decoder_pop_dim) + '-' + args.encode + '_' + args.decode + '_' + args.neurons
    else:
        file_dir = './params/hybrid-td3_td3-ilcsan-' + args.env + '-decoder-dim-' + str(args.decoder_pop_dim) + \
                   '-' + args.encode + '_' + args.decode + '_' + args.neurons

    if args.v_decay < 1.0:
        file_dir += '_' + str(args.v_decay)

    if args.connections != 'none':
        file_dir += '_' + args.connections

    reward_list = []

    for model_idx in range(args.num_model):
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
                                    lambda : gym.make(args.env), 
                                    args.encoder_pop_dim, 
                                    args.decoder_pop_dim, 
                                    mean_range, std, spike_ts,
                                    args.encode, args.decode,
                                    args.v_decay, args.neurons, 
                                    args.connections)

        best_rewards.append(reward)

        # Test Reward (best epoch idx) 
        model_file_dir = file_dir + '/model' + str(model_idx) + '_e' + str(best_epoch_idx) + '.pt'
        buffer_file_dir = file_dir + '/model' + str(model_idx) + '_e' + str(best_epoch_idx) + '_mean_var.p'

        reward = test_mujoco_render(model_file_dir, 
                                    buffer_file_dir, 
                                    lambda : gym.make(args.env), 
                                    args.encoder_pop_dim, 
                                    args.decoder_pop_dim, 
                                    mean_range, std, spike_ts,
                                    args.encode, args.decode,
                                    args.v_decay, args.neurons, 
                                    args.connections)

        best_rewards.append(reward)

        print("Model", model_idx, ", Reward:", best_rewards)

        reward_list.append(max(best_rewards))

    print('------------------------------------------------')
    print('mean: ', np.mean(reward_list))
    print('std: ', np.std(reward_list, ddof=1))
