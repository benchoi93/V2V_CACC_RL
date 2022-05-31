import numpy as np


class StateQueue():
    def __init__(self, size, template):
        self.size = size
        self.state = [-1 * np.ones_like(template)] * size

    def put(self, input):
        self.state = self.state[1:] + [input]

    def get_state(self):
        return np.stack(self.state, axis=1)
