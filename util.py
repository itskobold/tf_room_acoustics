import config as cfg
from scipy.stats.qmc import LatinHypercube


# Class of utility functions
class Util:
    def __init__(self,
                 manager):
        self.manager = manager

    # Convert real position in meters to position in discretized space
    def real_to_sample_pos(self,
                           xy_pos_real,
                           x_len_samples,
                           y_len_samples):
        x_pos = int(((xy_pos_real[0] + self.manager.metadata["x_len"] / 2)
                     / self.manager.metadata["x_len"]) * (x_len_samples - 1))
        y_pos = int(((xy_pos_real[1] + self.manager.metadata["y_len"] / 2)
                     / self.manager.metadata["y_len"]) * (y_len_samples - 1))
        return x_pos, y_pos

    # Convert relative position between (0, 1) to position in discretized space
    @staticmethod
    def relative_to_sample_pos(xy_pos_relative,
                               x_len_samples,
                               y_len_samples):
        x_pos = int((x_len_samples - 1) * xy_pos_relative[0])
        y_pos = int((y_len_samples - 1) * xy_pos_relative[1])
        return x_pos, y_pos

    # Sample a set of (x, y) collocation points using LHC sampling
    def sample_collocation_points(self,
                                  n=cfg.FDTD_NUM_IC_POSITIONS):
        # Sample points
        lhc = LatinHypercube(d=2)
        samples = lhc.random(n)

        # Map to real space and return
        x_len = self.manager.metadata["x_len"]
        y_len = self.manager.metadata["y_len"]
        samples[:, 0] = samples[:, 0] * x_len - x_len / 2
        samples[:, 1] = samples[:, 1] * y_len - y_len / 2
        return samples
