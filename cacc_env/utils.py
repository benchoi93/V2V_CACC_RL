import numpy as np
import numpy.typing as npt


class min_max_normalizer():
    def __init__(self,
                 min: npt.NDArray[np.float32],
                 max: npt.NDArray[np.float32]):

        self.min = min
        self.max = max

    def normalize(self, value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.array(np.divide(value - self.min, self.max - self.min))

    def denormalize(self, value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.array(value * (self.max - self.min) + self.min)


class min_max_normalizer_action():
    def __init__(self,
                 min: npt.NDArray[np.float32],
                 max: npt.NDArray[np.float32]):

        self.min = min
        self.max = max

    def normalize(self, value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.array(np.divide(2 * value - (self.min+self.max), self.max - self.min))

    def denormalize(self, value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 0.5 * np.array(value * (self.max - self.min) - (self.max + self.min))
