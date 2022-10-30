import numpy as np
import config as cfg
import json
import pickle

# Constants
SIDE_LEFT = 0
SIDE_RIGHT = 1
SIDE_BOTTOM = 2
SIDE_TOP = 3


# Class of utility functions.
class Util:
    def __init__(self,
                 manager):
        self.manager = manager

    # Convert real position in meters to position in discretized space.
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

    # Convert relative position between (0, 1) to position in discretized space.
    @staticmethod
    def relative_to_sample_pos(xy_pos_relative,
                               dim_lengths_samples):
        x_len_samples = dim_lengths_samples[0]
        y_len_samples = dim_lengths_samples[1]
        x_pos = int((x_len_samples - 1) * xy_pos_relative[0])
        y_pos = int((y_len_samples - 1) * xy_pos_relative[1])
        return x_pos, y_pos


# Create time string from timedelta object.
def timedelta_to_str(timedelta):
    hours, rem = divmod(timedelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours} hours, {minutes} mins, {seconds} secs"


# Get neural network input shape from data shape (X, Y, T).
def input_shape_from_data_shape(data_shape,
                                t_lookback=cfg.FNO_T_LOOKBACK):
    return data_shape[:2] + [t_lookback + 2]  # +2 for the additional (x, y) coordinate encoded into layer depth


# Get neural network output shape from data shape (X, Y, T).
def output_shape_from_data_shape(data_shape):
    return data_shape[:2] + [1]


# Mean relative error.
def mre(true_data,
        pred_data):
    return (pred_data - true_data).mean()


# Mean absolute error.
def mae(true_data,
        pred_data):
    return np.abs(pred_data - true_data).mean()


# Mean squared error.
def mse(true_data,
        pred_data):
    return ((pred_data - true_data) ** 2).mean()


# Root mean squared error.
def rmse(true_data,
         pred_data):
    return np.sqrt((pred_data - true_data) ** 2).mean()


# Returns the error for each (x, y) measured position.
# True and predicted datasets provided should have identical (X, Y) shapes (axes 0 and 1).
def calc_error_heatmap(true_data,
                       pred_data,
                       error_mode):
    # True and predicted data should be the same shapes
    pred_shape = np.shape(pred_data)
    td_diff = np.shape(true_data)[-1] - pred_shape[-1]
    true_data = true_data[:, :, td_diff:]
    true_shape = np.shape(true_data)
    assert true_shape == pred_shape

    # Get error func
    if error_mode == "mae":
        error_func = mae
    elif error_mode == "mre":
        error_func = mre
    elif error_mode == "mse":
        error_func = mse
    elif error_mode == "rmse":
        error_func = rmse
    else:
        print(f"WARNING: error mode {error_mode} unrecognized, no error heatmap will be generated.")
        return

    # Calculate error
    error_grid = np.zeros(pred_shape[:2])
    for x in range(pred_shape[0]):
        for y in range(pred_shape[1]):
            error_grid[x, y] = error_func(true_data=true_data[x, y, :],
                                          pred_data=pred_data[x, y, :])
    return error_grid


def array_to_formatted_str(data):
    ret = ""
    for i, entry in enumerate(data):
        ret += str(entry)
        if i + 1 < len(data):
            ret += ", "
    return ret


# Load .json from file.
def load_json(file_path):
    with open(file_path, "r") as inp:
        return json.load(inp)


# Save .json to file.
def save_json(file_path, data):
    with open(file_path, "w") as outp:
        json.dump(data, outp)


# Load .pkl from file.
def load_data(file_path):
    with open(file_path, "rb") as inp:
        return pickle.load(inp)


# Save .pkl to file.
def save_data(file_path, data):
    with open(file_path, "wb") as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
