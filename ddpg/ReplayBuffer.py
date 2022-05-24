import numpy as np


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, max_size=1000000):
        self.X_storage = []
        self.Y_storage = []
        self.U_storage = []
        self.R_storage = []
        self.D_storage = []

        self.max_size = max_size
        self.ptr = 0
        self.cnt = 0

    def push(self, data):
        for i in range(data[0].shape[0]):
            if len(self.X_storage) == self.max_size:
                self.X_storage[int(self.ptr)] = data[0][i]
                self.Y_storage[int(self.ptr)] = data[1][i]
                self.U_storage[int(self.ptr)] = data[2][i]
                self.R_storage[int(self.ptr)] = data[3][i]
                self.D_storage[int(self.ptr)] = data[4][i]

                self.ptr = (self.ptr + 1) % self.max_size
                self.cnt += 1
            else:
                self.X_storage.append(data[0][i])
                self.Y_storage.append(data[1][i])
                self.U_storage.append(data[2][i])
                self.R_storage.append(data[3][i])
                self.D_storage.append(data[4][i])

    def sample(self, batch_size, norm_rews=True):
        ind = np.random.randint(0, len(self.X_storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            # X, Y, U, R, D = self.storage[i]
            X = self.X_storage[i]
            Y = self.Y_storage[i]
            U = self.U_storage[i]
            R = self.R_storage[i]
            D = self.D_storage[i]

            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        if norm_rews:
            self.mu = np.mean(self.R_storage)
            self.std = np.mean(self.R_storage)
            r = (r - self.mu) / (self.std + 1e-8)

        return np.array(x), np.array(y), np.array(u), np.array(r), np.array(d)
