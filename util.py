import config as cfg
from scipy.stats.qmc import LatinHypercube
from datetime import datetime


# Class of utility functions
class Util:
    def __init__(self,
                 manager):
        self.manager = manager

    # Convert real position in meters to position in discretized space
    def real_to_sample_pos(self,
                           xy_pos_real,
                           dim_lengths_samples):
        x_len = self.manager.metadata["dim_lengths"][0]
        y_len = self.manager.metadata["dim_lengths"][1]
        x_len_samples = dim_lengths_samples[0]
        y_len_samples = dim_lengths_samples[1]
        x_pos = int(((xy_pos_real[0] + x_len / 2) / x_len) * (x_len_samples - 1))
        y_pos = int(((xy_pos_real[1] + y_len / 2) / y_len) * (y_len_samples - 1))
        return x_pos, y_pos

    # Convert relative position between (0, 1) to position in discretized space
    @staticmethod
    def relative_to_sample_pos(xy_pos_relative,
                               dim_lengths_samples):
        x_len_samples = dim_lengths_samples[0]
        y_len_samples = dim_lengths_samples[1]
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
        x_len = self.manager.metadata["dim_lengths"][0]
        y_len = self.manager.metadata["dim_lengths"][1]
        samples[:, 0] = samples[:, 0] * x_len - x_len / 2
        samples[:, 1] = samples[:, 1] * y_len - y_len / 2
        return samples

    # Sample a set of boundary absorption coefficients using LHC sampling
    @staticmethod
    def sample_boundary_absorption_coeffs(n=cfg.FDTD_NUM_BC_ABSORPTION_COEFFS):
        # Sample points and return
        lhc = LatinHypercube(d=4)
        return lhc.random(n)

    # Create time string from timedelta object
    @staticmethod
    def timedelta_to_str(timedelta):
        hours, rem = divmod(timedelta.seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours} hours, {minutes} mins, {seconds} secs"
