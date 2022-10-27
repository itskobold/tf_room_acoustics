import config as cfg
import util
from util import *
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.stats.qmc import LatinHypercube


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

    # Run acoustic finite-difference time domain simulation
    def run(self,
            num_solutions,
            file_name_out,
            sims_per_file=cfg.FDTD_SOLUTIONS_PER_FILE,
            round_bc_coeffs=cfg.FDTD_ROUND_BC_COEFFS):
        print(f"Starting FDTD simulation '{file_name_out}'...")

        # Init metadata dict
        metadata = {}

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
        metadata["dim_lengths_samples"] = x_len_samples, y_len_samples, t_len_samples

        # Init constants for finite difference calculation
        lmb = np.sqrt(0.5)
        a = 2 - (4 * lmb ** 2)
        b = lmb ** 2
        c = -1

        # Init empty array
        def init_array():
            return np.zeros([sims_per_file,  # TODO: make sizing dynamic
                             x_len_samples,
                             y_len_samples,
                             t_len_samples])

        # Start timer for entire computation
        t0_all = datetime.now()

        # Loop for number of solutions to be generated
        data = init_array()
        solution_index = 0
        total_solution_index = 0
        file_num = 0
        for i in range(num_solutions):
            # Get IC/BC points
            ic_pos = self.sample_collocation_point()
            bc_abs = self.sample_boundary_absorption_coeff(round_bc_coeffs)

            print(f"Running FDTD simulation {i + 1}/{num_solutions}: "
                  f"IC {ic_pos}, BC = {bc_abs}", end='')

            # Apply initial condition to pressure matrix at time step 0
            for x_ in range(x_len_samples):
                for y_ in range(y_len_samples):
                    x_val = x_ / x_len_samples * x_len
                    y_val = y_ / y_len_samples * y_len
                    x_val -= x_len / 2
                    y_val -= y_len / 2
                    data[solution_index][x_, y_, 0] = self.manager.icbc.gaussian_ic(x_val, y_val,
                                                                                    impulse_xy=ic_pos)

            # Start timer for this simulation
            t0 = datetime.now()
            for step in range(1, t_len_samples - 1):
                if (step - 1) % cfg.ANIM_FPS == 0:
                    print(".", end='')

                # Create matrices for readability
                prev_p = data[solution_index][:, :, step - 1]
                current_p = data[solution_index][:, :, step]
                next_p = data[solution_index][:, :, step + 1]

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
                            next_p[x, y] = self.manager.icbc.absorbing_bc(SIDE_LEFT,
                                                                          boundary_abs=bc_abs,
                                                                          current_p=current_p,
                                                                          prev_p=prev_p,
                                                                          x=x, y=y)
                        # Right boundary
                        elif x == x_len_samples - 1:
                            next_p[x, y] = self.manager.icbc.absorbing_bc(SIDE_RIGHT,
                                                                          boundary_abs=bc_abs,
                                                                          current_p=current_p,
                                                                          prev_p=prev_p,
                                                                          x=x, y=y)
                        # Bottom boundary
                        elif y == 0:
                            next_p[x, y] = self.manager.icbc.absorbing_bc(SIDE_BOTTOM,
                                                                          boundary_abs=bc_abs,
                                                                          current_p=current_p,
                                                                          prev_p=prev_p,
                                                                          x=x, y=y)
                        # Top boundary
                        elif y == y_len_samples - 1:
                            next_p[x, y] = self.manager.icbc.absorbing_bc(SIDE_TOP,
                                                                          boundary_abs=bc_abs,
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
                data[solution_index][:, :, step + 1] = next_p

            # Done with this simulation - print time in ms.
            print(f"done. Took {round((datetime.now() - t0).total_seconds() * 1000)}ms.")

            # Increment solution index and handle saving
            solution_index += 1
            if solution_index >= sims_per_file or i >= num_solutions - 1:
                # Save FDTD data to file
                self.save_data(data,
                               file_name_out=file_name_out,
                               file_num=file_num)

                # Handle variables and clear pressure matrix
                file_num += 1
                solution_index = 0
                data = init_array()

            # Update simulation metadata
            metadata[total_solution_index] = {"impulse_xy": ic_pos.tolist(),
                                              "boundary_abs": bc_abs.tolist()}
            total_solution_index += 1

        # Stop timer for entire computation
        timedelta = datetime.now() - t0_all
        metadata["computation_time"] = timedelta.total_seconds() * 1000  # In MS
        metadata["num_files"] = file_num
        print(f"Total FDTD simulation time: "
              f"{util.timedelta_to_str(timedelta)}.\n")

        # Save metadata
        self.save_metadata(metadata,
                           file_name_out=file_name_out)

    # Saves FDTD data to file.
    def save_data(self,
                  data,
                  file_name_out,
                  file_num):
        # Make folder
        file_path = f"{self.manager.get_proj_path()}fdtd/{file_name_out}/"
        Path(file_path).mkdir(parents=True, exist_ok=True)

        # Save data
        full_path = f"{file_path}{file_num}.pkl"
        util.save_data(full_path, data)
        print(f"Block {file_num}: saved FDTD data to '{full_path}'.")

    # Save metadata as .json after all simulations are complete.
    def save_metadata(self,
                      metadata,
                      file_name_out):
        # Make folder
        file_path = f"{self.manager.get_proj_path()}fdtd/{file_name_out}/"
        Path(file_path).mkdir(parents=True, exist_ok=True)

        # Save metadata
        full_path = f"{file_path}meta.json"
        util.save_json(f"{file_path}meta.json", metadata)
        print(f"Saved FDTD metadata as '{full_path}'.\n")

    # Sample a set of (x, y) collocation points using LHC sampling.
    def sample_collocation_points(self, n):
        # Sample points
        lhc = LatinHypercube(d=2)
        samples = lhc.random(n)

        # Map to real space and return
        x_len = self.manager.metadata["dim_lengths"][0]
        y_len = self.manager.metadata["dim_lengths"][1]
        samples[:, 0] = samples[:, 0] * x_len - x_len / 2
        samples[:, 1] = samples[:, 1] * y_len - y_len / 2
        return samples

    # Sample just 1 (x, y) collocation point and return it.
    def sample_collocation_point(self):
        return self.sample_collocation_points(1)[0]

    # Sample a set of boundary absorption coefficients using LHC sampling.
    @staticmethod
    def sample_boundary_absorption_coeffs(n,
                                          round_coeffs=cfg.FDTD_ROUND_BC_COEFFS):
        # Sample points
        lhc = LatinHypercube(d=4)
        bc_coeffs = lhc.random(n)

        # Round sampled points
        if round_coeffs:
            bc_coeffs = np.around(bc_coeffs,
                                  decimals=2)
        return bc_coeffs

    # Sample just 1 set of boundary absorption coefficients.
    # Round to 2 decimal places if round is True.
    def sample_boundary_absorption_coeff(self,
                                         round_coeffs=cfg.FDTD_ROUND_BC_COEFFS):
        return self.sample_boundary_absorption_coeffs(1, round_coeffs)[0]
