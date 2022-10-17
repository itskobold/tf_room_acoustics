import config as cfg
import icbc
import numpy as np
import time
import pickle
import json


# Class for managing FDTD simulation
class FDTD:
    # Init FDTD settings
    def __init__(self,
                 manager,
                 boundary_abs=cfg.BOUNDARY_ABS,
                 f_max=cfg.FDTD_F_MAX,
                 ppw=cfg.FDTD_PPW,
                 impulse_x=cfg.IMPULSE_X,
                 impulse_y=cfg.IMPULSE_Y,
                 impulse_a=cfg.IMPULSE_A,
                 impulse_r=cfg.IMPULSE_R,
                 num_solutions=cfg.FDTD_NUM_SOLUTIONS):
        self.manager = manager
        self.boundary_abs = boundary_abs
        self.f_max = f_max
        self.ppw = ppw
        self.impulse_x = impulse_x
        self.impulse_y = impulse_y
        self.impulse_a = impulse_a
        self.impulse_r = impulse_r
        self.num_solutions = num_solutions

        # Placeholder variables
        self.data = None
        self.metadata = None
        self.x_len_samples = None
        self.y_len_samples = None
        self.td_len_samples = None
        self.computation_time = None

    # Load FDTD data from file
    def load_data(self, file_name):
        with open(f'{self.manager.get_proj_path()}/{file_name}.pkl', 'rb') as inp:
            self.data = pickle.load(inp)
        with open(f'{self.manager.get_proj_path()}/{file_name}_meta.json', 'r') as inp:
            self.metadata = json.load(inp)

    # Saves FDTD data and metadata
    def save_data(self,
                  file_name_out):
        # Save FDTD data
        file_path = f'{self.manager.get_proj_path()}/{file_name_out}.pkl'
        with open(file_path, 'wb') as outp:
            pickle.dump(self.data, outp, pickle.HIGHEST_PROTOCOL)
        print(f'Saved FDTD data to "{file_path}".')

        # Save metadata
        file_path = f'{self.manager.get_proj_path()}/{file_name_out}_meta.json'
        with open(file_path, 'w') as outp:
            json.dump(self.metadata, outp)
        print(f'Saved FDTD metadata to "{file_path}".\n')

    # Run acoustic finite-difference time domain simulation
    def run(self):
        # Get grid spacing and system sample rate
        grid_spacing = self.manager.c / self.f_max / self.ppw
        dt = np.sqrt(0.5) * grid_spacing / self.manager.c

        # Work out the dimensions of the FDTD grid
        # We add 2 to each spatial dimension to include boundary nodes
        self.x_len_samples = int(np.ceil(self.manager.x_len / grid_spacing)) + 2
        self.y_len_samples = int(np.ceil(self.manager.y_len / grid_spacing)) + 2
        self.td_len_samples = int(np.ceil(self.manager.td_len / dt))

        # Init empty array
        self.data = np.zeros([self.num_solutions,
                              self.x_len_samples,
                              self.y_len_samples,
                              self.td_len_samples])

        # Init constants for finite difference calculation
        lmb = np.sqrt(0.5)
        a = 2 - (4 * lmb ** 2)
        b = lmb ** 2
        c = -1

        # Start timer
        t0 = time.time()

        # Loop for number of solutions to be generated
        for solution_index in range(self.num_solutions):
            pressure = self.data[solution_index]

            # Apply initial condition to pressure matrix at time step 0
            for x_ in range(self.x_len_samples):
                for y_ in range(self.y_len_samples):
                    x_val = x_ / self.x_len_samples * self.manager.x_len
                    y_val = y_ / self.y_len_samples * self.manager.y_len
                    x_val -= self.manager.x_len / 2
                    y_val -= self.manager.y_len / 2
                    pressure[x_, y_, 0] = icbc.gaussian_ic(x_val, y_val,
                                                           impulse_x=self.impulse_x,
                                                           impulse_y=self.impulse_y,
                                                           impulse_r=self.impulse_r,
                                                           impulse_a=self.impulse_a)
            # Run simulation
            print('Running FDTD simulation', end='')
            for step in range(1, self.td_len_samples - 1):
                if (step - 1) % cfg.ANIM_FPS == 0:
                    print('.', end='')

                # Create matrices for readability
                prev_p = pressure[:, :, step - 1]
                current_p = pressure[:, :, step]
                next_p = pressure[:, :, step + 1]

                # FD process
                # TODO: use slices to make this more efficient/tidier, GPU implementation
                for x in range(self.x_len_samples):
                    for y in range(self.y_len_samples):
                        # Bottom left corner
                        if x == 0 and y == 0:
                            next_p[x, y] = 0
                        # Bottom right corner
                        elif x == self.x_len_samples - 1 and y == 0:
                            next_p[x, y] = 0
                        # Top left corner
                        elif x == 0 and y == self.y_len_samples - 1:
                            next_p[x, y] = 0
                        # Top right corner
                        elif x == self.x_len_samples - 1 and y == self.y_len_samples - 1:
                            next_p[x, y] = 0
                        # Left boundary
                        elif x == 0:
                            next_p[x, y] = icbc.absorbing_bc('left', self.boundary_abs, current_p, prev_p, x, y)
                        # Right boundary
                        elif x == self.x_len_samples - 1:
                            next_p[x, y] = icbc.absorbing_bc('right', self.boundary_abs, current_p, prev_p, x, y)
                        # Bottom boundary
                        elif y == 0:
                            next_p[x, y] = icbc.absorbing_bc('bottom', self.boundary_abs, current_p, prev_p, x, y)
                        # Top boundary
                        elif y == self.y_len_samples - 1:
                            next_p[x, y] = icbc.absorbing_bc('top', self.boundary_abs, current_p, prev_p, x, y)
                        # Within domain
                        if x != 0 and x != self.x_len_samples - 1 and y != 0 and y != self.y_len_samples - 1:
                            next_p[x, y] = a * current_p[x, y] + \
                                           b * (current_p[x + 1, y] + current_p[x - 1, y] +
                                                current_p[x, y + 1] + current_p[x, y - 1]) + \
                                           c * prev_p[x, y]

                # Set pressure values for next time step
                pressure[:, :, step + 1] = next_p

            # Save pressure matrix and metadata
            self.data[solution_index] = pressure
            self.metadata = self.get_metadata()

        # Stop timer
        self.computation_time = (time.time() - t0) * 1000
        print('done.')
        print(f'FDTD simulation time: {round(self.computation_time, 2)}ms.')

    # Return dictionary of metadata for the previous simulation
    def get_metadata(self):
        return {'x_len_samples': self.x_len_samples,
                'y_len_samples': self.y_len_samples,
                'td_len_samples': self.td_len_samples,
                'boundary_abs': self.boundary_abs,
                'impulse_x': self.impulse_x,
                'impulse_y': self.impulse_y}
