import config as cfg
import numpy as np
from datetime import datetime
import pickle
import json


# Class for managing FDTD simulation
class FDTD:
    # Init FDTD settings
    def __init__(self,
                 manager,
                 f_max=cfg.FDTD_F_MAX,
                 ppw=cfg.FDTD_PPW):
        self.manager = manager
        self.f_max = f_max
        self.ppw = ppw

        # Placeholder variables
        self.data = None
        self.metadata = {}

    # Load FDTD data from file
    def load_data(self, file_name):
        with open(f"{self.manager.get_proj_path()}/{file_name}.pkl", "rb") as inp:
            self.data = pickle.load(inp)
        with open(f"{self.manager.get_proj_path()}/{file_name}_meta.json", "r") as inp:
            self.metadata = json.load(inp)

    # Saves FDTD data and metadata
    def save_data(self,
                  file_name_out):
        # Save FDTD data
        file_path = f"{self.manager.get_proj_path()}/{file_name_out}.pkl"
        with open(file_path, "wb") as outp:
            pickle.dump(self.data, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Saved FDTD data as '{file_path}'.")

        # Save metadata
        file_path = f"{self.manager.get_proj_path()}/{file_name_out}_meta.json"
        with open(file_path, "w") as outp:
            json.dump(self.metadata, outp)
        print(f"Saved FDTD metadata as '{file_path}'.\n")

    # Run acoustic finite-difference time domain simulation
    def run(self,
            ic_positions,
            bc_abs_coeffs):
        # Generate solutions for each IC and BC passed
        num_ic_positions = np.shape(ic_positions)[0]
        num_bc_coeffs = np.shape(bc_abs_coeffs)[0]
        num_solutions = num_ic_positions * num_bc_coeffs

        # Get grid spacing and system sample rate
        grid_spacing = self.manager.metadata["c"] / self.f_max / self.ppw
        dt = np.sqrt(0.5) * grid_spacing / self.manager.metadata["c"]

        # Readability
        x_len = self.manager.metadata["dim_lengths"][0]
        y_len = self.manager.metadata["dim_lengths"][1]
        t_len = self.manager.metadata["dim_lengths"][2]

        # Work out the dimensions of the FDTD grid
        # Add 2 to each spatial dimension to include boundary nodes
        x_len_samples = int(np.ceil(x_len / grid_spacing)) + 2
        y_len_samples = int(np.ceil(y_len / grid_spacing)) + 2
        t_len_samples = int(np.ceil(t_len / dt))
        self.metadata["dim_lengths_samples"] = x_len_samples, y_len_samples, t_len_samples

        # Init empty array
        self.data = np.zeros([num_solutions,
                              x_len_samples,
                              y_len_samples,
                              t_len_samples])

        # Init constants for finite difference calculation
        lmb = np.sqrt(0.5)
        a = 2 - (4 * lmb ** 2)
        b = lmb ** 2
        c = -1

        # Start timer for entire computation
        t0_all = datetime.now()

        # Loop for number of solutions to be generated
        for ic_index in range(num_ic_positions):
            for bc_index in range(num_bc_coeffs):
                solution_index = ic_index * num_ic_positions + bc_index
                pressure = self.data[solution_index]

                # Apply initial condition to pressure matrix at time step 0
                for x_ in range(x_len_samples):
                    for y_ in range(y_len_samples):
                        x_val = x_ / x_len_samples * x_len
                        y_val = y_ / y_len_samples * y_len
                        x_val -= x_len / 2
                        y_val -= y_len / 2
                        pressure[x_, y_, 0] = self.manager.icbc.gaussian_ic(x_val, y_val,
                                                                            impulse_xy=ic_positions[ic_index])
                # Run simulation
                print(f"Running FDTD simulation: IC {ic_index + 1}/{num_ic_positions},"
                      f" BC = {bc_index + 1}/{num_bc_coeffs}...", end='')

                # Start timer for this simulation
                t0 = datetime.now()
                for step in range(1, t_len_samples - 1):
                    if (step - 1) % cfg.ANIM_FPS == 0:
                        print(".", end='')

                    # Create matrices for readability
                    prev_p = pressure[:, :, step - 1]
                    current_p = pressure[:, :, step]
                    next_p = pressure[:, :, step + 1]

                    # FD process
                    # TODO: use slices to make this more efficient/tidier, GPU implementation
                    for x in range(x_len_samples):
                        for y in range(y_len_samples):
                            # Bottom left corner
                            if x == 0 and y == 0:
                                next_p[x, y] = 0
                            # Bottom right corner
                            elif x == x_len_samples - 1 and y == 0:
                                next_p[x, y] = 0
                            # Top left corner
                            elif x == 0 and y == y_len_samples - 1:
                                next_p[x, y] = 0
                            # Top right corner
                            elif x == x_len_samples - 1 and y == y_len_samples - 1:
                                next_p[x, y] = 0
                            # Left boundary
                            elif x == 0:
                                next_p[x, y] = self.manager.icbc.absorbing_bc("left",
                                                                              boundary_abs=bc_abs_coeffs[bc_index],
                                                                              current_p=current_p,
                                                                              prev_p=prev_p,
                                                                              x=x, y=y)
                            # Right boundary
                            elif x == x_len_samples - 1:
                                next_p[x, y] = self.manager.icbc.absorbing_bc("right",
                                                                              boundary_abs=bc_abs_coeffs[bc_index],
                                                                              current_p=current_p,
                                                                              prev_p=prev_p,
                                                                              x=x, y=y)
                            # Bottom boundary
                            elif y == 0:
                                next_p[x, y] = self.manager.icbc.absorbing_bc("bottom",
                                                                              boundary_abs=bc_abs_coeffs[bc_index],
                                                                              current_p=current_p,
                                                                              prev_p=prev_p,
                                                                              x=x, y=y)
                            # Top boundary
                            elif y == y_len_samples - 1:
                                next_p[x, y] = self.manager.icbc.absorbing_bc("top",
                                                                              boundary_abs=bc_abs_coeffs[bc_index],
                                                                              current_p=current_p,
                                                                              prev_p=prev_p,
                                                                              x=x, y=y)
                            # Within domain
                            if x != 0 and x != x_len_samples - 1 and y != 0 and y != y_len_samples - 1:
                                next_p[x, y] = a * current_p[x, y] + \
                                               b * (current_p[x + 1, y] + current_p[x - 1, y] +
                                                    current_p[x, y + 1] + current_p[x, y - 1]) + \
                                               c * prev_p[x, y]

                    # Set pressure values for next time step
                    pressure[:, :, step + 1] = next_p

                # Save pressure matrix and metadata
                self.data[solution_index] = pressure
                self.metadata[solution_index] = {"impulse_xy": ic_positions[ic_index].tolist(),
                                                 "boundary_abs": bc_abs_coeffs[bc_index].tolist()}

                # Done with this simulation - print time in ms.
                print(f"done. Took {round((datetime.now() - t0).total_seconds() * 1000)}ms.")

        # Stop timer for entire computation
        timedelta = datetime.now() - t0_all
        self.metadata["computation_time"] = timedelta.total_seconds() * 1000
        print(f"Total FDTD simulation time: "
              f"{self.manager.util.timedelta_to_str(timedelta)}.\n")
