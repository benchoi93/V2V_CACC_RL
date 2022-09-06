from __future__ import print_function
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Network import FCRelu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.base = FCRelu(state_dim, action_dim, hidden_dim)

    def forward(self, x):
        x = self.max_action * torch.tanh(self.base(x))
        return x


class ActorUp(nn.Module):
    """a bottom-up module used in bothway message passing that only passes message to its parent"""

    def __init__(self, state_dim, msg_dim, max_children):
        super(ActorUp, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64 + msg_dim, 64)
        self.fc3 = nn.Linear(64, msg_dim)

    def forward(self, x, *m):
        m = torch.cat(m, dim=-1)
        x = self.fc1(x)
        x = F.normalize(x, dim=-1)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        xm = self.fc2(xm)
        xm = torch.tanh(xm)
        xm = self.fc3(xm)
        xm = F.normalize(xm, dim=-1)
        msg_up = xm

        return msg_up


class ActorUpAction(nn.Module):
    """a bottom-up module used in bottom-up-only message passing that passes message to its parent and outputs action"""

    def __init__(self, state_dim, msg_dim, max_children, action_dim, hidden_dim, max_action):
        super(ActorUpAction, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64 + msg_dim * max_children, 64)
        self.fc3 = nn.Linear(64, msg_dim)
        self.action_base = FCRelu(state_dim + msg_dim * max_children, action_dim, hidden_dim, )
        self.max_action = max_action

    def forward(self, x, *m):
        m = torch.cat(m, dim=0)
        xm = torch.cat((x, m), dim=-1)
        xm = torch.tanh(xm)
        action = self.max_action * torch.tanh(self.action_base(xm))

        x = self.fc1(x)
        x = F.normalize(x, dim=-1)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        xm = self.fc2(xm)
        xm = torch.tanh(xm)
        xm = self.fc3(xm)
        xm = F.normalize(xm, dim=-1)
        msg_up = xm

        return msg_up, action


class ActorDownAction(nn.Module):
    """a top-down module used in bothway message passing that passes messages to children and outputs action"""

    # input dim is state dim if only using top down message passing
    # if using bottom up and then top down, it is the node's outgoing message dim
    def __init__(self, self_input_dim, action_dim, hidden_dim, msg_dim, max_action, max_children):
        super(ActorDownAction, self).__init__()
        self.max_action = max_action
        self.action_base = FCRelu(self_input_dim + msg_dim, action_dim, hidden_dim)
        self.msg_base = FCRelu(self_input_dim + msg_dim, msg_dim * max_children, hidden_dim)

    def forward(self, x, m):
        xm = torch.cat((x, m), dim=-1)
        xm = torch.tanh(xm)
        action = self.max_action * torch.tanh(self.action_base(xm))
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)
        return action, msg_down


class ActorGraphPolicy(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(self, state_dim, action_dim, hidden_dim, msg_dim, batch_size, max_action, max_children, disable_fold,
                 td, bu, num_processes):
        super(ActorGraphPolicy, self).__init__()
        self.num_limbs = 20
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_processes = num_processes

        assert self.action_dim == 1
        self.td = td
        self.bu = bu
        if self.bu:
            # bottom-up then top-down
            if self.td:
                self.sNet = nn.ModuleList([ActorUp(state_dim, msg_dim, max_children)] * self.num_limbs).to(device)
            # bottom-up only
            else:
                self.sNet = nn.ModuleList(
                    [ActorUpAction(state_dim, msg_dim, max_children, action_dim, hidden_dim,
                                   max_action)] * self.num_limbs).to(
                    device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
        # we pass msg_dim as first argument because in both-way message-passing, each node takes in its passed-up message as 'state'
        if self.td:
            # bottom-up then top-down
            if self.bu:
                self.actor = nn.ModuleList(
                    [ActorDownAction(msg_dim, action_dim, hidden_dim, msg_dim, max_action,
                                     max_children)] * self.num_limbs).to(
                    device)
            # top-down only
            else:
                self.actor = nn.ModuleList(
                    [ActorDownAction(state_dim, action_dim, hidden_dim, msg_dim, max_action,
                                     max_children)] * self.num_limbs).to(
                    device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "actor" + str(i).zfill(3), self.actor[i])

        # no message passing
        if not self.bu and not self.td:
            self.actor = nn.ModuleList([Actor(state_dim, action_dim, hidden_dim, max_action)] * self.num_limbs).to(
                device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "actor" + str(i).zfill(3), self.actor[i])

        if not self.disable_fold:
            for i in range(self.max_children):
                setattr(self, 'get_{}'.format(i), self.addFunction(i))

    def forward(self, state, mode='train'):
        # input
        # state : [batch_size, num_limbs, state_dim]

        self.clear_buffer()
        if mode == 'inference':
            temp = self.batch_size
            self.batch_size = self.num_processes

        for i in range(self.num_limbs):
            # self.input_state[i] = state[0][i]
            self.input_state[i] = state[:, i, :]

        if self.bu:
            # bottom up transmission by recursion
            # for i in range(self.num_limbs):
            # self.bottom_up_transmission(i)

            # since we know where is the "child-est" or "parent-est" node, we can call this function once
            # all other process is done by recursion
            self.bottom_up_transmission(0)

        if self.td:
            # top down transmission by recursion
            # for i in range(self.num_limbs):
            #     self.top_down_transmission(i)
            self.top_down_transmission(len(self.msg_down) - 1)

        if not self.bu and not self.td:
            for i in range(self.num_limbs):
                if self.action[i] is None:
                    #     self.action[i] = torch.zeros(1)
                    # else:
                    self.action[i] = self.actor[i](self.input_state[i])

        for i in range(self.num_limbs):
            if self.action[i] is None:
                self.action[i] = torch.zeros(1, requires_grad=True)

            if self.msg_down[i] is None:
                self.msg_down[i] = torch.zeros(1, requires_grad=True)

        self.action = torch.stack(self.action, dim=-1)
        self.msg_down = torch.stack(self.msg_down, dim=-1)

        if mode == 'inference':
            self.batch_size = temp

        return torch.squeeze(self.action)

    def bottom_up_transmission(self, node):

        if node < 0:
            return torch.zeros((self.batch_size, self.msg_dim), requires_grad=True).to(device)

        if node > len(self.msg_up)-1:
            return torch.zeros((self.batch_size, self.msg_dim), requires_grad=True).to(device)

        if self.msg_up[node] is None:
            self.msg_up[node] = torch.zeros((self.batch_size, self.msg_dim), requires_grad=True).to(device)
        else:
            return self.msg_up[node]

        state = self.input_state[node]

        msg_in = [self.bottom_up_transmission(node + 1)]
        # msg_in = torch.zeros((self.msg_dim, 1), requires_grad=True).to(device)

        if self.td:
            self.msg_up[node] = self.sNet[node](state, *msg_in)
        else:
            self.msg_up[node], self.action[node] = self.sNet[node](state, *msg_in)

        return self.msg_up[node]

    def top_down_transmission(self, node):
        # if node < 0:
        #     if not self.disable_fold:
        #         return self.zeroFold_td
        #     else:
        #         return torch.zeros((self.batch_size, self.msg_dim * self.max_children), requires_grad=True)

        if node < 0:
            return torch.zeros((self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

        if node > len(self.msg_down)-1:
            return torch.zeros((self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

        if self.msg_down[node] is None:
            self.msg_down[node] = torch.zeros((self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)
        else:
            return self.msg_down[node]

        # in both-way message-passing, each node takes in its passed-up message as 'state'
        if self.bu:
            state = self.msg_up[node]
        else:
            state = self.input_state[node]
        # parent_msg = self.top_down_transmission(self.parents[node])

        # find self children index (first child of parent, second child of parent, etc)
        # by finding the number of previous occurences of parent index in the list
        # self_children_idx = self.parents[:node].count(self.parents[node])

        # if the structure is flipped, flip message order at the root
        # if self.parents[0] == -2 and node == 1:
        #     self_children_idx = (self.max_children - 1) - self_children_idx
        #
        # if not self.disable_fold:
        #     msg_in = self.fold.add('get_{}'.format(self_children_idx), parent_msg)
        # else:
        #     msg_in = self.msg_slice(parent_msg, self_children_idx)

        msg_in = [self.top_down_transmission(node - 1)]

        self.action[node], self.msg_down[node] = self.actor[node](state, *msg_in)

        return self.msg_down[node]

    def zero_func_td(self):
        return torch.zeros((1, self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

    def zero_func_bu(self):
        return torch.zeros((1, self.batch_size, self.msg_dim), requires_grad=True).to(device)

    # an ugly way to define functions in a for loop (for torchfold only)
    def addFunction(self, n):
        def f(x):
            return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[n]

        return f

    def msg_slice(self, x, idx):
        return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[idx]

    def clear_buffer(self):
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents):
        self.parents = parents
        if not self.disable_fold:
            if self.bu:
                for i in range(1, self.num_limbs):
                    delattr(self, "sNet" + str(i).zfill(3))
            if not (self.bu and not self.td):
                for i in range(1, self.num_limbs):
                    delattr(self, "actor" + str(i).zfill(3))
        self.num_limbs = 20
        self.msg_down = [0] * self.num_limbs
        self.msg_up = [0] * self.num_limbs
        self.action = [0] * self.num_limbs
        self.input_state = [0] * self.num_limbs
        if self.bu:
            self.sNet = nn.ModuleList([self.sNet[0]] * self.num_limbs)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
        if not (self.bu and not self.td):
            self.actor = nn.ModuleList([self.actor[0]] * self.num_limbs)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "actor" + str(i).zfill(3), self.actor[i])

