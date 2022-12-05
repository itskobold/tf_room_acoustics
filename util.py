import numpy as np
import config as cfg
import json
import pickle
from pathlib import Path
from sklearn.utils import shuffle

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

    # Loop through FDTD data, output unshuffled data prepared for neural network input
    def create_raw_input_data(self,
                              fdtd_dir,
                              num_blocks,
                              t_lookback=cfg.FNO_T_LOOKBACK,
                              t_out_size=cfg.NN_OUTPUT_T_LEN,
                              pad_data=cfg.NN_PAD_DATA):
        print(f"Creating raw input data for FDTD dataset '{fdtd_dir}'...")
        fdtd_path = f"{self.manager.get_proj_path()}fdtd/{fdtd_dir}/"
        input_path = f"{fdtd_path}input/raw/"
        X_path = f"{input_path}X/"
        mesh_X_path = f"{input_path}mesh_X/"
        y_path = f"{input_path}y/"
        create_folder(X_path)
        create_folder(mesh_X_path)
        create_folder(y_path)
        meta = load_json(f"{fdtd_path}meta.json")
        sims_per_block = meta["sims_per_file"]

        # Loop through all blocks created by FDTD simulation
        for block_id in range(num_blocks):
            # Load block
            block = load_data(f"{fdtd_path}{block_id}.pkl")

            # Pad/trim data
            if pad_data:
                padding_amt = t_out_size - (block.shape[-1] % t_out_size)
                block = np.pad(block, [[0, 0], [0, 0], [0, 0], [0, padding_amt]])
            elif t_out_size > 1:
                block = block[..., :-(block.shape[-1] % t_out_size)]

            # Readability
            x_len_samples = block.shape[-3]
            y_len_samples = block.shape[-2]
            t_len_samples = block.shape[-1]

            # Create data buffers
            steps_to_predict = int(t_len_samples - t_lookback)
            if t_out_size > 1:
                steps_to_predict /= t_out_size
            a = np.zeros([sims_per_block * steps_to_predict, x_len_samples, y_len_samples, t_lookback])
            a_mesh = np.zeros([sims_per_block * steps_to_predict, 1])
            u = np.zeros([sims_per_block * steps_to_predict, x_len_samples, y_len_samples, t_out_size])

            # Loop through all solutions in dataset
            for i, fdtd_data in enumerate(block):
                # Split data into chunks and add to buffers (a, a_mesh, u)
                for step in range(steps_to_predict):
                    t = step * t_out_size
                    step_id = i * steps_to_predict + step
                    a[step_id] = fdtd_data[..., t:t + t_lookback]
                    a_mesh[step_id] = meta[f"{block_id * sims_per_block + i}"]["i"]
                    u[step_id] = fdtd_data[..., t + t_lookback:t + t_lookback + t_out_size]

            # Save data
            save_data(f"{X_path}{block_id}.pkl", a)
            save_data(f"{mesh_X_path}{block_id}.pkl", a_mesh)
            save_data(f"{y_path}{block_id}.pkl", u)
            print(f"Raw data created for block ID {block_id}.")
        print("Finished producing raw data from FDTD simulations.\n")

    # Process raw input data by shuffling it together
    def prepare_raw_data_for_network(self,
                                     fdtd_dir,
                                     num_blocks,
                                     shuffle_passes=cfg.NN_SHUFFLE_PASSES):
        print(f"Shuffling raw datasets together for FDTD dataset '{fdtd_dir}'...")

        # Shuffle through entire dataset multiple times
        for i in range(shuffle_passes):
            print(f"Shuffling raw data blocks: pass {i + 1}/{shuffle_passes}...")

            # Handle directories
            fdtd_path = f"{self.manager.get_proj_path()}fdtd/{fdtd_dir}/"
            input_path = f"{fdtd_path}input/"
            raw_path = f"{input_path}raw/"
            X_path = f"{input_path}X/"
            mesh_X_path = f"{input_path}mesh_X/"
            y_path = f"{input_path}y/"
            create_folder(X_path)
            create_folder(mesh_X_path)
            create_folder(y_path)

            # Shuffle order of blocks
            block_ids = np.arange(num_blocks)
            block_ids = shuffle(block_ids)

            # Loop through all blocks
            for j in range(num_blocks - 1):
                # Get 2 block IDs from shuffled list
                id_0 = block_ids[j]
                id_1 = block_ids[j + 1]

                # Load raw X, mesh_X and y data for adjacent blocks from block IDs
                load_path = raw_path if i == 0 else input_path
                b0_X = load_data(f"{load_path}X/{id_0}.pkl")
                b0_mesh_X = load_data(f"{load_path}mesh_X/{id_0}.pkl")
                b0_y = load_data(f"{load_path}y/{id_0}.pkl")
                b1_X = load_data(f"{load_path}X/{id_1}.pkl")
                b1_mesh_X = load_data(f"{load_path}mesh_X/{id_1}.pkl")
                b1_y = load_data(f"{load_path}y/{id_1}.pkl")

                # Concatenate adjacent block X, mesh_X and y data and shuffle together
                fb_X = np.concatenate([b0_X, b1_X], axis=0)
                fb_mesh_X = np.concatenate([b0_mesh_X, b1_mesh_X], axis=0)
                fb_y = np.concatenate([b0_y, b1_y], axis=0)
                fb_X, fb_mesh_X, fb_y = shuffle(fb_X, fb_mesh_X, fb_y)

                # Save to input data directory
                split = b0_X.shape[0]
                save_data(f"{X_path}{id_0}.pkl", fb_X[split:])
                save_data(f"{X_path}{id_1}.pkl", fb_X[:split])
                save_data(f"{mesh_X_path}{id_0}.pkl", fb_mesh_X[split:])
                save_data(f"{mesh_X_path}{id_1}.pkl", fb_mesh_X[:split])
                save_data(f"{y_path}{id_0}.pkl", fb_y[split:])
                save_data(f"{y_path}{id_1}.pkl", fb_y[:split])
        print("Finished shuffling raw datasets.\n")


# Create time string from timedelta object.
def timedelta_to_str(timedelta):
    hours, rem = divmod(timedelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours} hours, {minutes} mins, {seconds} secs"


# Get network input shape from data shape (X, Y, T).
def input_shape_from_data_shape(data_shape,
                                t_lookback=cfg.FNO_T_LOOKBACK):
    return data_shape[:2] + [t_lookback]


# Get network output shape from data shape (X, Y, T).
def output_shape_from_data_shape(data_shape,
                                 t_out_size=cfg.NN_OUTPUT_T_LEN):
    return data_shape[:2] + [t_out_size]


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


# Create comma-separated string from array data
def array_to_formatted_str(data):
    ret = ""
    for i, entry in enumerate(data):
        ret += str(entry)
        if i + 1 < len(data):
            ret += ", "
    return ret


# Create folder at path
def create_folder(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)


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
