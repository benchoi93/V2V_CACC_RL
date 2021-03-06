from .utils import min_max_normalizer
from typing import Tuple
import numpy as np
import random


class queue():
    def __init__(self, length):
        self.queue = []
        self.length = length

    def enqueue(self, value):
        if len(self.queue) < self.length:
            self.queue.append(value)
        else:
            self.queue.pop(0)
            self.queue.append(value)


class Vehicle():
    def __init__(self,
                 initial_position: float,
                 initial_speed: float,
                 dt: float = 0.1,
                 acc_bound: Tuple[float, float] = (-5, 5),
                 max_speed: float = 120.0 / 3.6,  # m/s
                 max_dec: float = -5.0,  # m/s2
                 reaction_time=1,
                 ):

        self.args = {"initial_position": initial_position,
                     "initial_speed": initial_speed,
                     "dt": dt}

        self.dt = dt
        self.acc_bound = acc_bound
        self.max_speed = max_speed
        self.max_dec = max_dec

        self.reaction_time = reaction_time

        self.reward_record = {"total": 0,
                              "speed": 0,
                              "gap": 0,
                              "safe": 0,
                              "jerk": 0,
                              "acc": 0,
                              "energy": 0,
                              "ss": 0}

        self.leader = None
        self.reset()

    @property
    def x(self):
        return self._x

    @property
    def v(self):
        return self._v

    @property
    def a(self):
        return self._a

    @property
    def jerk(self):
        return self._jerk

    @property
    def incoming_message(self):
        return self._incoming_message

    def set_leader(self, leader):
        self.leader = leader

    def update(self, acc: float, jamgap=1):
        """
        Update the vehicle's position and speed according to the action.
        """
        d_safe = self._v * self.reaction_time + (self.v)**2/(2*abs(self.acc_bound[0]))
        d_safe -= (self.leader.v)**2/(2*abs(self.acc_bound[0]))
        spacing = self.leader.x - self.x
        if d_safe > spacing:
            acc = self.acc_bound[0]  # severe deceleration

        if self.v + acc * self.dt < 0:
            acc = -self.v / self.dt

        if self.v + acc * self.dt > self.max_speed:
            acc = (self.max_speed - self.v) / self.dt

        self._jerk = (acc - self._a) / self.dt
        self._a = acc
        self._x = self.x + self.v * self.dt + 0.5 * self.a * self.dt ** 2
        self._v = self.v + self.a * self.dt

    def reset(self, force_x=None, force_v=None):
        if force_x is None:
            self._x = self.args["initial_position"]
        else:
            self._x = force_x

        if force_v is None:
            self._v = self.args["initial_speed"]
        else:
            self._v = force_v

        self._a = 0.0
        self._jerk = 0.0
        self.action_record = 0
        self._outgoing_message = 0
        self._incoming_message = 0
        self._reaction_queue = queue(11)

    def get_energy_consumption(self):
        power = 0

        M = 1200      # mass of average sized vehicle (kg)
        g = 9.81      # gravitational acceleration (m/s^2)
        Cr = 0.005    # rolling resistance coefficient
        Ca = 0.3      # aerodynamic drag coefficient
        rho = 1.225   # air density (kg/m^3)
        A = 2.6       # vehicle cross sectional area (m^2)

        speed = self.v
        accel = self.a
        power += M * speed * accel + M * g * Cr * speed + 0.5 * rho * A * Ca * speed ** 3
        return power * 0.001  # kilo Watts (KW)

    def broadcast(self, target, message):
        target._incoming_message = message

    def get_gibbs_acc(self):
        v_lead = self.leader.v
        v_ego = self.v

        x_lead = self.leader.x
        x_ego = self.x

        a = 3
        b = 2
        del_t = 2.1
        s0 = 3

        s = x_lead - x_ego

        inside = (b*del_t)**2 + v_lead**2 + 2*b*(s-s0-v_ego*del_t)
        inside = max(inside, 0)

        v_safe = -b * del_t + (inside) ** 0.5
        self._reaction_queue.enqueue(v_safe)

        v_next = min(v_ego + a * 0.1, self.max_speed, self._reaction_queue.queue[0])

        acc = min((v_next - v_ego) / 0.1, a)
        # acc = max(acc, -3)

        return acc

    def get_GHR_acc(self):
        v_lead = self.leader.v
        v_ego = self.v

        x_lead = self.leader.x
        x_ego = self.x

        del_t = 11
        alpha = 1
        beta = 1
        gamma = 1

        self._reaction_queue.enqueue(v_lead)

        acc = alpha * (self._reaction_queue.queue[0] ** beta) * (v_lead - v_ego) / (x_lead - x_ego)**gamma
        # acc = (v_new - v_ego) / del_t

        return acc

    def get_idm_acc(self):
        desired_speed = self.max_speed
        timegap = 1.5
        a = self.acc_bound[1]
        b = -self.acc_bound[0]

        delta = 4
        jamgap = 2

        v_lead = self.leader.v
        v_ego = self.v

        x_lead = self.leader.x
        x_ego = self.x

        v_rel = v_ego - v_lead
        spacing = x_lead - x_ego

        s_star = jamgap + v_ego * timegap + (v_ego * v_rel) / (2 * (a * b)**0.5)

        acc = a * (1 - (v_ego / desired_speed) ** delta - (s_star/spacing) ** 2)

        return acc

    def get_eidm_acc(self):
        a = self.acc_bound[1]
        b = -self.acc_bound[0]
        c = 0.99

        acc_iidm = self.get_idm_acc()
        acc_cah = self.get_acc_cah()

        if acc_iidm > acc_cah:
            acc = acc_iidm
        else:
            acc = (1-c) * acc_iidm + c * (acc_cah + b * np.tanh((acc_iidm - acc_cah) / b))

        return acc

    def get_acc_cah(self):
        a = self.acc_bound[1]

        v_lead = self.leader.v
        v_ego = self.v

        x_lead = self.leader.x
        x_ego = self.x

        v_rel = v_ego - v_lead
        spacing = x_lead - x_ego

        a_restricted = a
        dvp = max(v_rel, 0)
        v_lead2 = v_ego - dvp
        denomCAH = v_lead2 * v_lead2 - 2 * spacing * a_restricted

        if (v_lead2 * dvp < -2 * spacing * a_restricted) and (denomCAH != 0):
            acc_cah = v_ego * v_ego * a_restricted / denomCAH
        else:
            acc_cah = a_restricted - 0.5 * dvp * dvp / max(spacing, 0.0001)

        return acc_cah


class NGSIM_Virtual_Leader(Vehicle):
    def __init__(self,
                 initial_position: float,
                 initial_speed: float,
                 dt: float = 0.1,
                 acc_bound: Tuple[float, float] = (-5, 5),
                 max_speed: float = 100.0 / 3.6,  # m/s
                 reaction_time=1.0,

                 ):
        self.timecnt = 0
        self.mode = 0

        super().__init__(initial_position, initial_speed, dt, acc_bound, max_speed, reaction_time)

    def reset(self, force_x=None, force_v=None, ref_traj=None):
        self.ref_traj = ref_traj

        if force_x is None:
            self._x = self.args["initial_position"]
        else:
            self._x = force_x

        if force_v is None:
            self._v = self.args["initial_speed"]
        else:
            self._v = force_v

        self._a = 0.0
        self._jerk = 0.0
        self.action_record = 0
        self.timecnt = 0

    def update(self):
        self.timecnt += 1
        leader_v = self.ref_traj[self.timecnt, 3]

        new_a = (self.v - leader_v) / self.dt
        self._jerk = (new_a - self._a) / self.dt
        self._a = new_a

        self._x = self.x + leader_v * self.dt
        self._v = leader_v


class Virtual_Leader(Vehicle):
    def __init__(self,
                 initial_position: float,
                 initial_speed: float,
                 dt: float = 0.1,
                 acc_bound: Tuple[float, float] = (-5, 5),
                 max_speed: float = 100.0 / 3.6,  # m/s
                 reaction_time=1.0,
                 keep_duration=300,
                 reach_speed=10,
                 ):
        self.timecnt = 0
        self.mode = 0
        self.keep_cnt = 0
        # self.keep_duration = keep_duration
        # randomly select keep_duration from range [0,keep_duration]
        self.keep_duration_max = keep_duration
        self.reach_speed_max = reach_speed

        super().__init__(initial_position, initial_speed, dt, acc_bound, max_speed, reaction_time)

    def update(self):
        self.timecnt += 1

        if self.mode == 0:
            self._v = self.max_speed
            self._x = self.x + self.v * self.dt
            if self.timecnt > 300:
                self.mode = 1
        elif self.mode == 1:
            if self._v <= self.reach_speed:
                self.keep_cnt += 1
                if self.keep_cnt > self.keep_duration:
                    self.mode = 2
                self._a = 0

            else:
                self._a = -2.5

            prev_v = self._v
            self._v = max(self._v + self._a * self.dt, 0)
            self._a = (self._v - prev_v) / self.dt
            self._x = self.x + self.v * self.dt

        elif self.mode == 2:
            self._a = 2.5
            prev_v = self._v
            self._v = min(self._v + self._a * self.dt, self.max_speed)
            self._a = (self._v - prev_v) / self.dt
            self._x = self.x + self.v * self.dt

    def reset(self):
        self.timecnt = 0
        self.mode = 0
        self.keep_cnt = 0

        self._x = self.args["initial_position"]
        self._v = self.args["initial_speed"]
        self._a = 0.0
        self._jerk = 0.0
        self.action_record = 0
        self._outgoing_message = 0
        self._incoming_message = 0

        self.keep_duration = np.random.randint(0, self.keep_duration_max)
        self.reach_speed = np.random.randint(0, self.reach_speed_max)

        # self.keep_duration = 100
        # self.reach_speed = 5
