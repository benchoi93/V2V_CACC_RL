import argparse
import torch
import numpy as np
import gym
import os
from itertools import count
from pathlib import Path
from ddpg.DDPG import DDPG
from cacc_env.multiCACCenv import multiCACC
from cacc_env.state_type import state_minmax_lookup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
    parser.add_argument('--tau',  default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--test_iteration', default=10, type=int)

    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size
    parser.add_argument('--batch_size', default=100, type=int)  # mini batch size
    parser.add_argument('--seed', default=False, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)
    # optional parameters

    parser.add_argument('--sample_frequency', default=2000, type=int)
    parser.add_argument('--render', default=False, type=bool)  # show UI or not
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--load', default=False, type=bool)  # load model
    parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--max_episode', default=100000, type=int)  # num of games
    parser.add_argument('--print_log', default=5, type=int)
    parser.add_argument('--update_iteration', default=200, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)

    parser.add_argument("--adj-amp", default=1, type=float)
    parser.add_argument("--speed-reward-coef", default=1, type=float)
    parser.add_argument("--safe-reward-coef", default=1, type=float)
    parser.add_argument("--jerk-reward-coef", default=1, type=float)
    parser.add_argument("--acc-reward-coef", default=1, type=float)
    parser.add_argument("--energy-reward-coef", default=0.01, type=float)

    parser.add_argument("--shared-reward", default=True, action='store_true')
    parser.add_argument("--enable-communication", default=False, action='store_true')

    parser.add_argument("--num-agents", default=20, type=int)
    parser.add_argument("--init_spacing", default=50, type=float)
    parser.add_argument("--init_speed", default=30, type=float)
    parser.add_argument("--max-speed", default=120/3.6, type=float)
    parser.add_argument("--acc-bound", default=5, type=float)
    parser.add_argument("--keep-duration", default=100, type=int)
    parser.add_argument("--track-length", default=3000, type=float)

    config = parser.parse_args()

    args = parser.parse_args()

    return args


def main(args, device, directory):
    state_dim = env.observation_space.shape[0] // args.num_agents
    action_dim = env.action_space.shape[0] // args.num_agents
    max_action = float(env.action_space.high[0])
    hidden_dim = args.hidden_dim

    agent = DDPG(state_dim, action_dim, hidden_dim, max_action, device, directory, args)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.load:
            agent.load()
        total_step = 0
        for i in range(args.max_episode):
            total_reward = 0
            step = 0
            state = env.reset()
            print(f"Episode: {i} started - virtual leader info = (keep_duration ={env.virtual_leader.keep_duration} , min_speed = {env.virtual_leader.reach_speed})")
            for t in count():
                action = agent.select_action(state)
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

                next_state, reward, done, info = env.step(action)
                if args.render and i >= args.render_interval:
                    env.render()
                agent.replay_buffer.push((state, next_state, action, np.array(reward), np.array(done, bool)))

                state = next_state
                if all(done):
                    break
                step += 1
                total_reward += np.mean(reward)
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f} Avg Reward: \t{:0.2f}".format(total_step, i, total_reward, total_reward/step))
            agent.update()
           # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")


if __name__ == "__main__":
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_name = os.path.basename(__file__)

    kwargs = {"num_agents": args.num_agents,
              "initial_position": np.cumsum(np.ones(args.num_agents)) * args.init_spacing,
              "initial_speed": np.ones(args.num_agents) * args.init_speed,
              "dt": 0.1,
              "acc_bound": (-1 * args.acc_bound, args.acc_bound),
              "adj_amp": args.adj_amp,
              "track_length": args.track_length,
              "max_speed": args.max_speed,
              "coefs": [args.speed_reward_coef,
                        args.safe_reward_coef,
                        args.jerk_reward_coef,
                        args.acc_reward_coef,
                        args.energy_reward_coef],
              "shared_reward": args.shared_reward,
              "config": args,
              "state_minmax_lookup": state_minmax_lookup,
              "enable_communication": args.enable_communication,
              #   "enable_communication": True,
              }

    env = multiCACC(**kwargs)

    if args.seed:
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    min_Val = torch.tensor(1e-7).float().to(device)  # min value

    model_dir = Path('./models')
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    directory = model_dir/curr_run / 'logs'

    main(args, device, directory)
