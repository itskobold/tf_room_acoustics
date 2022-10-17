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
        x_pos = int(((xy_pos_real[0] + self.manager.x_len / 2)
                     / self.manager.x_len) * (x_len_samples - 1))
        y_pos = int(((xy_pos_real[1] + self.manager.y_len / 2)
                     / self.manager.y_len) * (y_len_samples - 1))
        return x_pos, y_pos

    # Convert relative position between (0, 1) to position in discretized space
    @staticmethod
    def relative_to_sample_pos(xy_pos_relative,
                               x_len_samples,
                               y_len_samples):
        x_pos = int((x_len_samples - 1) * xy_pos_relative[0])
        y_pos = int((y_len_samples - 1) * xy_pos_relative[1])
        return x_pos, y_pos
