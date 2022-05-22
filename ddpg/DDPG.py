import numpy as np
import time
from collections import defaultdict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ActorNetwork import Actor, ActorUpAction, ActorGraphPolicy
from .CriticNetwork import Critic, CriticUpAction, CriticGraphPolicy
from .ReplayBuffer import Replay_buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG(object):
    # def __init__(self, state_dim, action_dim, hidden_dim, max_action, device, directory, args):
    #
    #     self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
    #     self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)

    #     self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
    #     self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
    def __init__(self, state_dim, action_dim, hidden_dim, msg_dim, batch_size, max_action, max_children, disable_fold,
                 td, bu, directory, device, args):
        self.actor = ActorGraphPolicy(state_dim, action_dim, hidden_dim,
                                      msg_dim, batch_size,
                                      max_action, max_children,
                                      disable_fold, td, bu).to(device)

        self.actor_target = ActorGraphPolicy(state_dim, action_dim, hidden_dim,
                                             msg_dim, batch_size,
                                             max_action, max_children,
                                             disable_fold, td, bu).to(device)

        self.critic = CriticGraphPolicy(state_dim, action_dim, hidden_dim,
                                        msg_dim, batch_size,
                                        max_children,
                                        disable_fold, td, bu).to(device)

        self.critic_target = CriticGraphPolicy(state_dim, action_dim, hidden_dim,
                                               msg_dim, batch_size,
                                               max_children,
                                               disable_fold, td, bu).to(device)
        # self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        # self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.args = args
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.directory = directory

        self._cnt = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state = state.view(-1, self.args.num_agents, self.state_dim)
        return self.actor(state, mode="inference").cpu().data.numpy().flatten()

    def update(self):
        self._cnt += 1

        time_func = defaultdict(list)

        for it in range(self.args.update_iteration):
            now = time.time()
            # Sample replay buffer
            b = self.args.batch_size
            n = self.args.num_agents

            x, y, u, r, d = self.replay_buffer.sample(self.args.batch_size)
            state = torch.FloatTensor(x).to(self.device).view(b, n, x.shape[-1] // n)  # [B x N x S]
            action = torch.FloatTensor(u).to(self.device).view(b, n, u.shape[-1] // n)  # [B x N x A]
            next_state = torch.FloatTensor(y).to(self.device).view(b, n, y.shape[-1] // n)  # [B x N x S]
            done = torch.FloatTensor(1 - d).to(self.device).view(b, n, d.shape[-1] // n)  # [B x N x 1]
            reward = torch.FloatTensor(r).to(self.device).view(b, n, r.shape[-1] // n)  # [B x N x 1]

            time_func["replay_buffer"].append(time.time() - now)
            now = time.time()

            # Compute the target Q value
            next_action = self.actor_target(next_state).view(b, n, self.action_dim)
            target_Q = self.critic_target(next_state, next_action)  # [B x N x 1]
            target_Q = reward + (done * self.args.gamma * target_Q).detach()

            time_func["critic_target"].append(time.time() - now)
            now = time.time()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            time_func["critic"].append(time.time() - now)
            now = time.time()

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            time_func["critic_optimizer"].append(time.time() - now)
            now = time.time()

            # Compute actor loss
            cur_pol_action = self.actor(state).view(b, n, self.action_dim)
            actor_loss = -self.critic(state, cur_pol_action).mean()
            actor_loss += ((cur_pol_action) ** 2).mean() * 0.001
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            time_func["actor_loss"].append(time.time() - now)
            now = time.time()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            time_func["actor_optimizer"].append(time.time() - now)
            now = time.time()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

        print(f"time_func")
        print(f"replay_buffer={np.mean(time_func['replay_buffer']):.5f}  \t  critic={np.mean(time_func['critic']):.5f}")
        print(f"critic_target={np.mean(time_func['critic_target']):.5f}  \t  actor_loss={np.mean(time_func['actor_loss']):.5f}  ")
        print(f"actor_optimizer={np.mean(time_func['actor_optimizer']):.5f}  \t  critic_optimizer={np.mean(time_func['critic_optimizer']):.5f}  ")

    def save(self):
        torch.save(self.actor.state_dict(), self.directory / f'actor_{self._cnt}.pth')
        torch.save(self.critic.state_dict(), self.directory / 'critic_{self._cnt}.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(self.directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(self.directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
