import time
import argparse
import torch
import numpy as np
import gym
import os
from itertools import count
from pathlib import Path
from ddpg.DDPG import DDPG
from ddpg.TD3 import TD3
from cacc_env.multiCACCenv import multiCACC
from cacc_env.state_type import state_minmax_lookup
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--test_iteration', default=10, type=int)

    parser.add_argument('--learning_rate', default=1e-1, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size
    parser.add_argument('--batch_size', default=512, type=int)  # mini batch size
    parser.add_argument('--seed', default=False, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)
    # optional parameters

    parser.add_argument('--sample_frequency', default=2000, type=int)
    parser.add_argument('--render', action="store_true", default=False)  # show UI or not
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--load', default=False, type=bool)  # load model
    parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--max_episode', default=100000, type=int)  # num of games
    parser.add_argument('--print_log', default=5, type=int)
    parser.add_argument('--update_iteration', default=200, type=int)
    parser.add_argument('--hidden-dim', default=64, type=int)
    parser.add_argument('--max_children', default=1, type=int)
    parser.add_argument('--msg_dim', default=32, type=int)

    parser.add_argument("--adj-amp", default=5, type=float)
    parser.add_argument("--speed-reward-coef", default=1, type=float)
    parser.add_argument("--safe-reward-coef", default=1, type=float)
    parser.add_argument("--jerk-reward-coef", default=1, type=float)
    parser.add_argument("--acc-reward-coef", default=0, type=float)
    parser.add_argument("--energy-reward-coef", default=0, type=float)

    parser.add_argument("--shared-reward", default=False, action='store_true')
    parser.add_argument("--enable-communication", default=False, action='store_true')
    parser.add_argument("--episode_length", default=1000, type=int)

    parser.add_argument("--num-agents", default=20, type=int)
    parser.add_argument("--init_spacing", default=50, type=float)
    parser.add_argument("--init_speed", default=30, type=float)
    parser.add_argument("--max-speed", default=120 / 3.6, type=float)
    parser.add_argument("--acc-bound", default=5, type=float)
    parser.add_argument("--keep-duration", default=100, type=int)
    parser.add_argument("--track-length", default=3000, type=float)
    parser.add_argument("--num_processes", default=2, type=int)
    parser.add_argument("--max_steps", default=1000, type=int)

    parser.add_argument("--td", default=False, action="store_true")
    parser.add_argument("--bu", default=False, action="store_true")
    parser.add_argument("--model", choices=["TD3", "DDPG"], default="TD3")

    args = parser.parse_args()

    return args


def main(args, device, directory):
    state_dim = env.observation_space.shape[0] // args.num_agents
    action_dim = env.action_space.shape[0] // args.num_agents
    max_action = float(env.action_space.high[0])
    hidden_dim = args.hidden_dim

    if args.model == "TD3":
        model = TD3
    elif args.model == "DDPG":
        model = DDPG
    else:
        raise NotImplementedError

    agent = model(state_dim=state_dim,
                  action_dim=action_dim,
                  hidden_dim=hidden_dim,
                  msg_dim=args.msg_dim,
                  batch_size=args.batch_size,
                  max_action=max_action,
                  max_children=1,
                  disable_fold=True,
                  td=args.td,
                  bu=args.bu,
                  directory=directory,
                  device=device,
                  args=args)
    # (self, state_dim, action_dim, hidden_dim, msg_dim, batch_size, max_action, max_children, disable_fold, td, bu, device, directory, args):
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
            # total_reward = 0
            step = 0
            state = env.reset()

            envs = env.get_envs()

            for k in range(args.num_processes):
                print(
                    f"Episode: {i} Process {k} started - virtual leader info = (keep_duration ={envs[k][0].virtual_leader.keep_duration} , min_speed = {envs[k][0].virtual_leader.reach_speed})")

            now = time.time()

            done_list = [False for i in range(args.num_processes)]
            total_reward_list = [0 for i in range(args.num_processes)]
            for t in count():
                action = agent.select_action(state)

                if args.render and i % args.render_interval == 0:
                    # for k in range(args.num_processes):
                    if t == args.episode_length - 1:
                        figs = env.render(display=True, save=True)
                        for k in range(len(figs)):
                            fig = figs[k][0]
                            agent.writer.add_figure('episode', fig, global_step=i*args.num_processes + k)
                    else:
                        env.render(display=False)

                else:
                    action = (action + np.random.normal(0, args.exploration_noise,
                                                        size=env.action_space.shape[0])).clip(
                        env.action_space.low, env.action_space.high)

                # if i == 0:
                #     action = np.zeros_like(action)

                next_state, reward, done, info = env.step(action)

                if args.shared_reward:
                    reward = reward + np.broadcast_to(np.expand_dims(reward.mean(1), 1), reward.shape)

                agent.replay_buffer.push((state, next_state, action, np.array(reward), np.array(done, bool)))

                state = next_state

                done_list = [done_list[i] or done.all(1)[i] for i in range(args.num_processes)]

                # if all(done_list):
                # break
                if t >= args.max_steps:
                    break
                step += 1

                sum_reward = np.mean(reward, 1)
                total_reward_list = [x+sum_reward[i] for i, x in enumerate(total_reward_list)]

            for k in range(args.num_processes):
                total_step += (step + 1)
                print(
                    f"Total T:{total_step} || ReplayBuffer {len(agent.replay_buffer.X_storage)} || Episode: {i} || Total Reward {total_reward_list[k]} || Avg Reward {total_reward_list[k]/step}")
                agent.writer.add_scalar('episode_reward', total_reward_list[k], i * args.num_processes + k)
                agent.writer.add_scalar('avg_reward', total_reward_list[k] / step, i * args.num_processes + k)
            print(f"Rollout Time : {time.time() - now :.2f}")

            now = time.time()
            agent.update()
            print(f"Update Time : {time.time() - now :.2f}")
            print("-----------------------------------------\n")
            # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")


if __name__ == "__main__":
    args = parse_args()

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

    # env = multiCACC(**kwargs)

    def make_env(kwargs):
        def _init():
            env = multiCACC(**kwargs)
            return env
        return _init

    env = [make_env(kwargs) for i in range(args.num_processes)]
    env = SubprocVecEnv(env)

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
    directory = model_dir / curr_run / 'logs'

    main(args, device, directory)
