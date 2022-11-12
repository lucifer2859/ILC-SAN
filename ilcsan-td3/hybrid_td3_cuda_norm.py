from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
import pickle
import os
import sys

from replay_buffer_norm import ReplayBuffer
from popsan import PopSpikeActor
from mdcsan import MDCSpikeActor
from core_cuda import MLPQFunction


class SpikeActorDeepCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts, 
                 device, encode, decode, v_decay, neurons, connections,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        # build policy and value functions
        if 'DN' in neurons:
            self.popsan = MDCSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
                                        mean_range, std, spike_ts, act_limit, device, encode, decode, v_decay, 
                                        neurons, connections)
        else:
            self.popsan = PopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
                                        mean_range, std, spike_ts, act_limit, device, encode, decode, v_decay, 
                                        neurons, connections)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, batch_size):
        with torch.no_grad():
            return self.popsan(obs, batch_size).cpu().numpy()


def spike_td3(env_fn, actor_critic=SpikeActorDeepCritic, ac_kwargs=dict(), seed=0,
              steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
              polyak=0.995, popsan_lr=1e-4, q_lr=1e-3, batch_size=100, start_steps=10000,
              update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
              noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
              save_freq=5, norm_clip_limit=3, norm_update=50, tb_comment='', model_idx=0, use_cuda=True):
    # Set device
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 clip_limit=norm_clip_limit, norm_update_every=norm_update)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            popsan_targ = ac_targ.popsan(o2, batch_size)

            # Target policy smoothing
            epsilon = torch.randn_like(popsan_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = popsan_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_popsan_targ = ac_targ.q1(o2, a2)
            q2_popsan_targ = ac_targ.q2(o2, a2)
            q_popsan_targ = torch.min(q1_popsan_targ, q2_popsan_targ)
            backup = r + gamma * (1 - d) * q_popsan_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                         Q2Vals=q2.cpu().detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 popsan loss
    def compute_loss_popsan(data):
        o = data['obs']
        q1_popsan = ac.q1(o, ac.popsan(o, batch_size))
        return -q1_popsan.mean()

    # Set up optimizers for policy and q-function
    popsan_optimizer = Adam(ac.popsan.parameters(), lr=popsan_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for popsan.
            popsan_optimizer.zero_grad()
            loss_popsan = compute_loss_popsan(data)
            loss_popsan.backward()
            popsan_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), 1)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        ###
        # compuate the return mean test reward
        ###
        test_reward_sum = 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(replay_buffer.normalize_obs(o), 0))
                ep_ret += r
                ep_len += 1
            test_reward_sum += ep_ret
        return test_reward_sum / num_test_episodes

    ###
    # Add tensorboard support and save rewards
    # also create dir for saving parameters
    ###
    # writer = SummaryWriter(comment="_" + tb_comment + "_" + str(model_idx))
    save_test_reward = []
    save_test_reward_steps = []
    try:
        os.mkdir("./params")
        print("Directory params Created")
    except FileExistsError:
        print("Directory params already exists")
    model_dir = "./params/hybrid-td3_" + tb_comment
    try:
        os.mkdir(model_dir)
        print("Directory ", model_dir, " Created")
    except FileExistsError:
        print("Directory ", model_dir, " already exists")

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(replay_buffer.normalize_obs(o), act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # writer.add_scalar(tb_comment + '/Train-Reward', ep_ret, t + 1)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(device, batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                ac.popsan.to('cpu')
                torch.save(ac.popsan.state_dict(),
                           model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '.pt')
                ac.popsan.to(device)
                pickle.dump([replay_buffer.mean, replay_buffer.var],
                            open(model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '_mean_var.p', "wb+"))
                print("Weights saved in ", model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '.pt')

            # Test the performance of the deterministic version of the agent.
            test_mean_reward = test_agent()
            save_test_reward.append(test_mean_reward)
            save_test_reward_steps.append(t + 1)
            print("Model: ", model_idx, " Steps: ", t + 1, " Mean Reward: ", test_mean_reward)

    # Save Test Reward List
    pickle.dump([save_test_reward, save_test_reward_steps],
                open(model_dir + '/' + "model" + str(model_idx) + "_test_rewards.p", "wb+"))


if __name__ == '__main__':
    import math
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    parser.add_argument('--start_model_idx', type=int, default=0)
    parser.add_argument('--num_model', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--encode', type=str, default='pop-det', choices=['pop-det', 'pop', 'layer'])
    parser.add_argument('--decode', type=str, default='last-mem', choices=['fr-mlp', 'last-mem', 'max-mem', 'mean-mem'])
    parser.add_argument('--v_decay', type=float, default=1.0)
    parser.add_argument('--neurons', type=str, default='LIF', choices=['LIF', 'DN'])
    parser.add_argument('--connections', default='intra', choices=['none', 'intra', 'no-bias', 'no-self', 'no-lateral', 'bias', 'self', 'lateral'])
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device_id)

    START_MODEL = args.start_model_idx
    NUM_MODEL = args.num_model
    AC_KWARGS = dict(hidden_sizes=[256, 256],
                     encoder_pop_dim=args.encoder_pop_dim,
                     decoder_pop_dim=args.decoder_pop_dim,
                     mean_range=(-3, 3),
                     std=math.sqrt(args.encoder_var),
                     spike_ts=5,
                     device=torch.device('cuda'),
                     encode=args.encode,
                     decode=args.decode,
                     v_decay=args.v_decay,
                     neurons=args.neurons,
                     connections=args.connections)

    if 'pop' in args.encode:
        COMMENT = 'td3-ilcsan-' + args.env + '-encoder-dim-' + str(args.encoder_pop_dim) + \
                  '-decoder-dim-' + str(args.decoder_pop_dim) + '-' + args.encode + '_' + \
                  args.decode + '_' + args.neurons
    elif args.encode == 'none':
        COMMENT = 'td3-ilcsan-' + args.env + '-decoder-dim-' + str(args.decoder_pop_dim) + \
                  '-' + args.encode + '_' + args.decode + '_' + args.neurons
    else:
        print('Error: undefined encode')

    if args.v_decay < 1.0:
        COMMENT += '_' + str(args.v_decay)

    if args.connections != 'none':
        COMMENT += '_' + args.connections
    
    for num in range(START_MODEL, START_MODEL + NUM_MODEL):
        seed = num * 10
        spike_td3(lambda : gym.make(args.env), actor_critic=SpikeActorDeepCritic, ac_kwargs=AC_KWARGS,
                  popsan_lr=1e-4, gamma=0.99, seed=seed, epochs=args.epochs,
                  norm_clip_limit=3.0, tb_comment=COMMENT, model_idx=num)
