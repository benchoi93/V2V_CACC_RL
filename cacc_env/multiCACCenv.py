# from .multiCACCenv import multiCACC
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import gym
from .vehicle import Vehicle, Virtual_Leader
from .utils import min_max_normalizer, min_max_normalizer_action
from .rendering import Viewer, Figure
from .state_type import state_function

from gym.spaces import Box


class multiCACC(gym.Env):
    '''def __init__(self,
                 num_agents: int,
                 initial_position: npt.NDArray[np.float32],
                 initial_speed: npt.NDArray[np.float32],
                 dt: float = 0.1,
                 acc_bound: tuple[float, float] = (-5, 5),
                 track_length: float = 1000.0,
                 max_speed: float = 120.0 / 3.6,  # m/s
                 state_mins: npt.NDArray[np.float32] = np.array([0, 0, -5, 0, -120]),
                 state_maxs: npt.NDArray[np.float32] = np.array([1000, 120, 5, 1000, 120]),
                 shared_rewad: bool = False
                 ):'''

    def __init__(self, num_agents,
                 initial_position,
                 initial_speed, dt=0.1,
                 acc_bound=(-1, 1),
                 adj_amp=1,
                 track_length=2000.0,
                 max_speed=120.0,
                 #  state_type=["speed", "acceleration", "spacing", "relative_speed", "message"],
                 state_type=["speed", "spacing", "relative_speed", "acceleration"],
                 #  state_mins=np.array([0, 0, -5, 0, -120/3.6]),
                 #  state_maxs=np.array([1000, 120/3.6, 5, 1000, 120/3.6]),
                 shared_reward=False,
                 coefs=[1, 1, 1, 1],
                 config=None,
                 state_minmax_lookup=None,
                 enable_communication=False,):

        assert num_agents > 0
        assert initial_position.shape[0] == num_agents
        assert initial_speed.shape[0] == num_agents

        self.num_agents = num_agents
        self.initial_position = initial_position
        self.initial_speed = initial_speed
        self.dt = dt
        self.acc_bound = acc_bound
        self.track_length = track_length
        self.max_speed = max_speed
        self.max_steps = 1000

        self.state_type = state_type
        self.state_minmax_lookup = state_minmax_lookup
        minmax = [state_minmax_lookup[s_name] for s_name in state_type]
        self.state_mins = np.array([x[0] for x in minmax])
        self.state_maxs = np.array([x[1] for x in minmax])

        self.shared_reward = shared_reward
        self.config = config

        self.adj_amp = adj_amp

        self.coefs = coefs

        self._step_count = 0
        self.viewer = None
        # state of the environment
        # ego_position, ego_speed, ego_acceleration, spacing, relative_speed

        # self.observation_space = Box(low=0, high=1, shape=(num_agents, 2))
        # self.action_space = Box(low=-1, high=1, shape=(num_agents,))  # acceleration

        # self.observation_space = [Box(low=0, high=1, shape=(len(state_type),)) for _ in range(num_agents)]
        self.observation_space = Box(low=0, high=1, shape=(len(state_type) * num_agents,))
        # self.observation_space = [Box(low=0, high=1, shape=(5,)) for _ in range(num_agents)]

        self.action_space = Box(low=-1, high=1, shape=(num_agents,))

        self.action_normalizer = min_max_normalizer_action(-1 * adj_amp, adj_amp)

        self.state_normalizer = min_max_normalizer(self.state_mins, self.state_maxs)

        self.virtual_leader = Virtual_Leader(self.initial_position[num_agents-1]+self.config.init_spacing,
                                             self.initial_speed[num_agents-1],
                                             self.dt,
                                             self.acc_bound,
                                             self.max_speed,
                                             keep_duration=config.keep_duration)

        self.agents = [Vehicle(self.initial_position[-1-i],
                               self.initial_speed[i],
                               self.dt,
                               self.acc_bound,
                               self.max_speed
                               ) for i in range(num_agents)]

        for i in range(len(self.agents)):
            if i == 0:
                self.agents[i].set_leader(self.virtual_leader)
            else:
                self.agents[i].set_leader(self.agents[i-1])

        self.reset()

    def get_state(self, norm=True, i=None) -> npt.NDArray[np.float32]:
        if i is not None:
            states = np.zeros(shape=(len(self.state_type)))
            for j, s_name in enumerate(self.state_type):
                states[j] = state_function[s_name](self.agents[i])

        else:
            states = np.zeros(shape=(self.num_agents, len(self.state_type)))
            for i in range(self.num_agents):
                for j, s_name in enumerate(self.state_type):
                    states[i, j] = state_function[s_name](self.agents[i])

            # ego position
            # states[i, 0] = self.agents[i].x
            # # ego speed
            # states[i, 1] = self.agents[i].v
            # # ego acceleration
            # states[i, 2] = self.agents[i].a
            # # spacing
            # prev_x = self.agents[i-1].x if i > 0 else self.virtual_leader.x
            # states[i, 3] = prev_x - self.agents[i].x
            # # relative speed
            # prev_v = self.agents[i-1].v if i > 0 else self.virtual_leader.v
            # states[i, 4] = self.agents[i].v - prev_v

        if norm:
            return self.state_normalizer.normalize(states)
        else:
            return states

    def get_agent_obs(self, i):
        # TODO : implement partial observation by agent i
        return self.get_state(i=i)

    def check_collision(self, i):
        is_collision = False

        # positions = [self.virtual_leader.x] + list(self.get_state(norm=False)[:, 0])
        positions = [self.virtual_leader.x] + [a0.x for a0 in self.agents]

        for j in range(len(positions)):
            if positions[j] <= positions[i+1]:
                break

        is_collision = i+1 != j

        return is_collision

    def get_agent_reward(self, i, action, coefs=[1, 1, 1, 1]) -> float:
        # state = self.get_state(norm=False)
        # norm_state = self.state_normalizer.normalize(state)

        spd_reward = - np.abs(self.agents[i].v - self.agents[i].max_speed) / self.agents[i].max_speed
        is_collision = self.check_collision(i)
        spacing = state_function['spacing'](self.agents[i])
        rel_spd = state_function['relative_speed'](self.agents[i])

        if rel_spd == 0:
            safe_reward = 0
        else:
            if is_collision:
                TTC = 1e-10
            else:
                # TTC = state[i, 3] / state[i, 4]
                TTC = spacing / rel_spd

            if 1e-10 < TTC < 4:
                safe_reward = np.log(TTC/4)
            else:
                safe_reward = 0

        mu = 0.422618
        sigma = 0.43659
        if spacing <= 0:
            gap_reward = -1
        else:
            gap_reward = (np.exp(-(np.log(spacing) - mu) ** 2 / (2 * sigma ** 2)) / (spacing * sigma * np.sqrt(2 * np.pi)))

        jerk_reward = - (self.agents[i]._jerk)**2 / (self.acc_bound[1] - self.acc_bound[0])**2

        energy_reward = - max(self.agents[i].get_energy_consumption(), 0)

        acc_reward = - (self.agents[i].a)**2 / self.acc_bound[1]**2

        reward = coefs[0] * gap_reward + \
            coefs[1] * safe_reward + \
            coefs[2] * jerk_reward + \
            coefs[3] * acc_reward + \
            coefs[4] * energy_reward

        # action_diff = np.abs(self.agents[i].a - action)

        if is_collision:
            reward += -10  # collision_penalty

        self.agents[i].reward_record['speed'] = gap_reward
        self.agents[i].reward_record['safe'] = safe_reward
        self.agents[i].reward_record['acc'] = acc_reward
        self.agents[i].reward_record['jerk'] = jerk_reward
        self.agents[i].reward_record['energy'] = energy_reward
        self.agents[i].reward_record['total'] = reward

        return reward

    def get_reward(self) -> float:
        # TODO: Not implemented yet
        return 0

    def get_shared_reward(self) -> float:
        # TODO: Not implemented yet
        return 0

    def get_done(self, i) -> bool:
        return (self.agents[i].x > self.track_length) and (self._step_count > self.max_steps)

    def clip_acc(self, acc):
        # if acc < self.acc_bound[0]:
        #     return self.acc_bound[0]
        # el
        if acc <= self.acc_bound[1]:
            return acc
        else:
            return self.acc_bound[1]

    def step(self, action_n: npt.NDArray[np.float32]) -> Tuple[List[npt.NDArray[np.float32]], List[float], List[bool], Dict[Any, Any]]:
        # assert action_n.shape[0] == self.num_agents

        self._step_count += 1

        obs_n = []
        reward_n = []
        done_n = []
        info_n: Dict[Any, Any] = {}  # TODO : implement info
        self.virtual_leader.update()

        if self.config.enable_communication:
            communication_n = [x[1] for x in action_n]
            action_n = [x[0] for x in action_n]

        for i in range(self.num_agents):
            idm_acc = self.agents[i].get_idm_acc()

            adj = float(self.action_normalizer.denormalize(action_n[i]))

            # acc = idm_acc + adj
            acc = adj
            acc = self.clip_acc(acc)

            self.agents[i].action_record = adj

            self.agents[i].update(acc)
            if self.config.enable_communication:
                self.agents[i]._outgoing_message = communication_n[i]
                self.agents[i].leader.broadcast(self.agents[i], self.agents[i].leader._outgoing_message)

            # obs_n += [self.get_agent_obs(i)]
            reward_n += [self.get_agent_reward(i, action_n[i], coefs=self.coefs)]
            done_n += [self.get_done(i)]

        if self.shared_reward:
            shared_reward = np.mean(reward_n)
            # reward_n = [np.mean(reward_n)] * self.num_agents
            reward_n = [x+shared_reward for x in reward_n]

        obs_n = self.get_state().flatten()

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self._step_count = 0

        for i in range(self.num_agents):
            self.agents[i].reset()
        self.virtual_leader.reset()

        # obs_n = []
        # for i in range(self.num_agents):
        #     obs_n += [self.get_agent_obs(i)]

        if isinstance(self.viewer, Viewer):
            if self.viewer is not None:
                self.viewer.close()

        self.viewer = None

        obs_n = self.get_state().flatten()
        return obs_n

    def render(self, mode="human", display=True, save=False, viewer=False, save_path=None):
        screen_width = 1500
        screen_height = 500
        # clearance_x = 80
        # clearance_y = 10
        # zero_x = 0.25 * screen_width
        # visible_track_length = 1000
        # scale_x = screen_width / visible_track_length

        if self.viewer is None:
            import matplotlib.pyplot as plt
            # import seaborn as sns
            if viewer:
                self.viewer = Viewer(width=screen_width,
                                     height=screen_height)
            else:
                self.viewer = NULLViewer()

            for i in range(self.num_agents):
                if i == 0:
                    self.viewer.history['time_cnt'] = defaultdict(list)
                    self.viewer.history['position'] = defaultdict(list)
                    self.viewer.history['speed'] = defaultdict(list)
                    self.viewer.history['jerk_value'] = defaultdict(list)
                    self.viewer.history['TTC_value'] = defaultdict(list)
                    self.viewer.history['acceleration'] = defaultdict(list)
                    self.viewer.history['adjustment'] = defaultdict(list)
                    self.viewer.history['spacing'] = defaultdict(list)
                    self.viewer.history['relative_speed'] = defaultdict(list)
                    self.viewer.history['total_reward'] = defaultdict(list)
                    self.viewer.history['speed_reward'] = defaultdict(list)
                    self.viewer.history['safe_reward'] = defaultdict(list)
                    self.viewer.history['jerk_reward'] = defaultdict(list)
                    self.viewer.history['acc_reward'] = defaultdict(list)
                    self.viewer.history['energy_reward'] = defaultdict(list)
                    self.viewer.history['message'] = defaultdict(list)

                self.viewer.history['time_cnt'][str(i)] = []
                self.viewer.history['position'][str(i)] = []
                self.viewer.history['speed'][str(i)] = []
                self.viewer.history['jerk_value'][str(i)] = []
                self.viewer.history['TTC_value'][str(i)] = []
                self.viewer.history['acceleration'][str(i)] = []
                self.viewer.history['adjustment'][str(i)] = []
                self.viewer.history['spacing'][str(i)] = []
                self.viewer.history['relative_speed'][str(i)] = []
                self.viewer.history['total_reward'][str(i)] = []
                self.viewer.history['speed_reward'][str(i)] = []
                self.viewer.history['safe_reward'][str(i)] = []
                self.viewer.history['jerk_reward'][str(i)] = []
                self.viewer.history['acc_reward'][str(i)] = []
                self.viewer.history['energy_reward'][str(i)] = []
                self.viewer.history['message'][str(i)] = []

            # self.fig = plt.Figure((640 / 80, 200 / 80), dpi=80)
            self.fig = plt.Figure((screen_width/80, screen_height/80), dpi=80)
            info = Figure(self.fig,
                          rel_anchor_x=0,
                          rel_anchor_y=0,
                          batch=self.viewer.batch,
                          group=self.viewer.background)
            # info.position = (clearance_x - 40, 225 + clearance_y)
            self.viewer.components['info'] = info

        # state_tmp = self.get_state(norm=False)
        # state = np.zeros((self.num_agents+1, len(self.state_type)))
        # state[1:, :] = state_tmp
        # state[0, :] = [self.virtual_leader.x, self.virtual_leader.v, self.virtual_leader.a] + [0] * (len(self.state_type)-3)

        # state = self.state_normalizer.denormalize(state)

        for i in range(self.num_agents+1):
            if i == 0:
                self.viewer.history['time_cnt'][str(i)].append(self._step_count)
                self.viewer.history['position'][str(i)].append(self.virtual_leader.x)
                self.viewer.history['speed'][str(i)].append(self.virtual_leader.v)
                self.viewer.history['jerk_value'][str(i)].append(0)
                self.viewer.history['TTC_value'][str(i)].append(0)
                self.viewer.history['acceleration'][str(i)].append(self.virtual_leader.a)
                self.viewer.history['spacing'][str(i)].append(0)
                self.viewer.history['relative_speed'][str(i)].append(0)
                self.viewer.history['message'][str(i)].append(0)
            else:
                self.viewer.history['time_cnt'][str(i)].append(self._step_count)
                self.viewer.history['position'][str(i)].append(self.agents[i-1].x)
                self.viewer.history['speed'][str(i)].append(self.agents[i-1].v)
                self.viewer.history['jerk_value'][str(i)].append(self.agents[i-1].jerk)
                self.viewer.history['TTC_value'][str(i)].append(abs(state_function['spacing'](
                    self.agents[i-1])/(state_function['relative_speed'](self.agents[i-1]) + 0.00001)))
                self.viewer.history['acceleration'][str(i)].append(self.agents[i-1].a)
                self.viewer.history['spacing'][str(i)].append(state_function['spacing'](self.agents[i-1]))
                self.viewer.history['relative_speed'][str(i)].append(state_function['relative_speed'](self.agents[i-1]))
                self.viewer.history['message'][str(i)].append(state_function['message'](self.agents[i-1]))

            if i == 0:
                self.viewer.history['adjustment'][str(i)].append(0)
            else:
                self.viewer.history['adjustment'][str(i)].append(self.agents[i-1].action_record)

            # record rewards
            if i == 0:
                reward_record = {"total": 0,
                                 "speed": 0,
                                 "safe": 0,
                                 "jerk": 0,
                                 "acc": 0,
                                 "energy": 0}
            else:
                reward_record = self.agents[i-1].reward_record
            self.viewer.history['total_reward'][str(i)].append(reward_record['total'])
            self.viewer.history['speed_reward'][str(i)].append(reward_record['speed'])
            self.viewer.history['safe_reward'][str(i)].append(reward_record['safe'])
            self.viewer.history['jerk_reward'][str(i)].append(reward_record['jerk'])
            self.viewer.history['acc_reward'][str(i)].append(reward_record['acc'])
            self.viewer.history['energy_reward'][str(i)].append(reward_record['energy'])

        if display:
            from matplotlib import cm
            cmap = cm.get_cmap('inferno', self.num_agents + 1)

            self.fig.clf()
            ax = self.fig.add_subplot(251)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['position'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('Position in m')
            ax.set_xlim(0, max(self._step_count, 100))

            ax = self.fig.add_subplot(252)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['speed'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('Speed in m/s')
            ax.set_xlim(0, max(self._step_count, 100))
            ax.set_ylim(0, self.max_speed*1.2)

            ax = self.fig.add_subplot(253)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['acceleration'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('Acceleration in m/s2')
            ax.set_xlim(0, max(self._step_count, 100))
            ax.set_ylim(self.acc_bound[0]*1.1, self.acc_bound[1]*1.1)

            ax = self.fig.add_subplot(254)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['adjustment'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('Adjustment in m/s2')
            ax.set_xlim(0, max(self._step_count, 100))
            ax.set_ylim(-1 * self.adj_amp * 1.1, self.adj_amp * 1.1)

            # ax = self.fig.add_subplot(255)
            # for i in range(self.num_agents + 1):
            #     ax.plot(self.viewer.history['time_cnt'][str(i)],
            #             self.viewer.history['message'][str(i)],
            #             lw=2,
            #             color=cmap(i))
            # ax.set_xlabel('Time in 0.1 s')
            # ax.set_ylabel('Message')
            # ax.set_xlim(0, max(self._step_count, 100))
            # ax.set_ylim(-1 * 1.1,  1.1)

            ax = self.fig.add_subplot(2, 5, 5)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['acc_reward'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('acc_reward')
            ax.set_xlim(0, max(self._step_count, 100))

            ax = self.fig.add_subplot(256)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['total_reward'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('total_reward')
            ax.set_xlim(0, max(self._step_count, 100))

            ax = self.fig.add_subplot(257)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['speed_reward'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('speed_reward')
            ax.set_xlim(0, max(self._step_count, 100))

            ax = self.fig.add_subplot(258)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['safe_reward'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('safe_reward')
            ax.set_xlim(0, max(self._step_count, 100))

            ax = self.fig.add_subplot(259)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['jerk_reward'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('jerk_reward')
            ax.set_xlim(0, max(self._step_count, 100))

            # ax = self.fig.add_subplot(2, 5, 9)
            # for i in range(self.num_agents + 1):
            #     ax.plot(self.viewer.history['time_cnt'][str(i)],
            #             self.viewer.history['gap_reward'][str(i)],
            #             lw=2,
            #             color=cmap(i))
            # ax.set_xlabel('Time in 0.1 s')
            # ax.set_ylabel('gap_reward')
            # ax.set_xlim(0, max(self._step_count, 100))

            ax = self.fig.add_subplot(2, 5, 10)
            for i in range(self.num_agents + 1):
                ax.plot(self.viewer.history['time_cnt'][str(i)],
                        self.viewer.history['energy_reward'][str(i)],
                        lw=2,
                        color=cmap(i))
            ax.set_xlabel('Time in 0.1 s')
            ax.set_ylabel('energy_reward')
            ax.set_xlim(0, max(self._step_count, 100))

            # ax.set_ylim(self.acc_bound[0], self.acc_bound[1])

            self.fig.tight_layout()
            self.viewer.components['info'].figure = self.fig
            if save:
                # self.fig.savefig(save_path)
                return self.fig
            else:
                return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class NULLViewer(object):
    def __init__(self):
        self.history = {}
        self.components = {}
        self.batch = None
        self.background = None
