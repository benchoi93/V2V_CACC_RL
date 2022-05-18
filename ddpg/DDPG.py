from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ReplayBuffer import Replay_buffer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], -1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, device, directory, args):
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.device = device
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.directory = directory

        self._cnt = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state = state.view(-1, self.args.num_agents, self.state_dim)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        self._cnt += 1

        for it in range(self.args.update_iteration):
            # Sample replay buffer
            b = self.args.batch_size
            n = self.args.num_agents

            x, y, u, r, d = self.replay_buffer.sample(self.args.batch_size)
            state = torch.FloatTensor(x).to(self.device).view(b, n, x.shape[-1]//n)
            action = torch.FloatTensor(u).to(self.device).view(b, n, u.shape[-1]//n)
            next_state = torch.FloatTensor(y).to(self.device).view(b, n, y.shape[-1]//n)
            done = torch.FloatTensor(1-d).to(self.device).view(b, n, d.shape[-1]//n)
            reward = torch.FloatTensor(r).to(self.device).view(b, n, r.shape[-1]//n)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            cur_pol_action = self.actor(state)
            actor_loss = -self.critic(state, cur_pol_action).mean()
            actor_loss += ((cur_pol_action)**2).mean() * 0.001
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

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
