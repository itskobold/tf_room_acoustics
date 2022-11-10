import config as cfg
import util
import numpy as np
from datetime import datetime
from scipy.stats.qmc import LatinHypercube
from scipy.signal import ricker


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
            file_name_out,
            num_meshes=cfg.FDTD_NUM_MESHES,
            num_simulations=cfg.FDTD_NUM_SIMULATIONS,
            sims_per_file=cfg.FDTD_SOLUTIONS_PER_FILE,
            round_bc_coeffs=cfg.FDTD_ROUND_BC_COEFFS):
        print(f"Starting FDTD simulation '{file_name_out}'...")

        # Init metadata dict
        metadata = {}

        # Readability
        c = self.manager.metadata["c"]
        x_len = self.manager.metadata["dim_lengths"][0]
        y_len = self.manager.metadata["dim_lengths"][1]
        t_len = self.manager.metadata["dim_lengths"][2]

        # Get grid spacing and system sample rate
        lmb = np.sqrt(0.5)
        dx = c / self.f_max / self.ppw
        dt = lmb * dx / c

        # Work out the dimensions of the FDTD grid
        # Add 2 to each spatial dimension to include boundary nodes
        n_x = int(np.ceil(x_len / dx)) + 2
        n_y = int(np.ceil(y_len / dx)) + 2
        n_t = int(np.ceil(t_len / dt))
        metadata["dim_lengths_samples"] = n_x, n_y, n_t

        # Init empty array
        def init_array():
            return np.zeros([sims_per_file, n_x, n_y, n_t])

        data = init_array()

        # Prepare counters and start timer for entire computation
        solution_index = 0
        total_solution_index = 0
        file_num = 0
        t0_all = datetime.now()

        # Loop for number of meshes to be generated
        # TODO: better mesh generation
        for mesh_id in range(num_meshes):
            # Create mesh and save it
            mesh = self.create_mesh(metadata["dim_lengths_samples"],
                                    round_bc_coeffs=round_bc_coeffs)
            self.save_mesh(mesh,
                           file_name_out=file_name_out,
                           mesh_id=mesh_id)

            # Loop for number of solutions to be generated
            for sim_id in range(num_simulations):
                # Get IC point
                ic_pos = self.sample_collocation_point(mesh)
                ic_pos_samples = self.manager.util.real_to_sample_pos(ic_pos,
                                                                      metadata["dim_lengths_samples"])
                print(f"Mesh {mesh_id + 1}/{num_meshes}: running FDTD simulation {sim_id + 1}/{num_simulations} "
                      f"(IC = {util.array_to_formatted_str(ic_pos)})", end='')

                # Set sample rate
                metadata["sample_rate"] = 1 / dt

                # Create IC
                u_in = np.zeros(n_t)
                ricker_len = int(np.ceil(5 * metadata["sample_rate"] / self.f_max))
                u_in[:ricker_len] = ricker(points=ricker_len, a=4)
                u_in *= 1 / max(np.abs(u_in))

                # Create gamma and interior masks
                gam_mask = np.copy(mesh)
                gam_mask[gam_mask < 0] = 0
                in_mask = np.copy(gam_mask)
                in_mask[in_mask > 0] = 1
                in_mask = np.ones_like(in_mask) - in_mask

                # Create K map (interior neighbors)
                k_map = np.zeros_like(in_mask)
                k_map[1: n_x - 1, 1: n_y - 1] = k_map[1:n_x - 1, 1:n_y - 1] + in_mask[2:n_x, 1:n_y - 1]
                k_map[1: n_x - 1, 1: n_y - 1] = k_map[1:n_x - 1, 1:n_y - 1] + in_mask[0:n_x - 2, 1:n_y - 1]
                k_map[1: n_x - 1, 1: n_y - 1] = k_map[1:n_x - 1, 1:n_y - 1] + in_mask[1:n_x - 1, 2:n_y]
                k_map[1: n_x - 1, 1: n_y - 1] = k_map[1:n_x - 1, 1:n_y - 1] + in_mask[1:n_x - 1, 0:n_y - 2]
                k_map *= in_mask

                # Create gamma map (absorption)
                g_map = np.zeros_like(gam_mask)
                g_map[1: n_x - 1, 1: n_y - 1] = g_map[1:n_x - 1, 1:n_y - 1] + gam_mask[2:n_x, 1:n_y - 1]
                g_map[1: n_x - 1, 1: n_y - 1] = g_map[1:n_x - 1, 1:n_y - 1] + gam_mask[0:n_x - 2, 1:n_y - 1]
                g_map[1: n_x - 1, 1: n_y - 1] = g_map[1:n_x - 1, 1:n_y - 1] + gam_mask[1:n_x - 1, 2:n_y]
                g_map[1: n_x - 1, 1: n_y - 1] = g_map[1:n_x - 1, 1:n_y - 1] + gam_mask[1:n_x - 1, 0:n_y - 2]
                g_map /= np.full_like(g_map, 4) - k_map
                g_map[np.isnan(g_map)] = 0

                # TODO
                # g_map = np.divide(g_map, np.full_like(g_map, 4) - k_map,
                #                   out=np.zeros_like(g_map), where=g_map != 0)

                # Start timer for this simulation
                t0 = datetime.now()
                for step in range(1, n_t - 1):
                    if (step - 1) % 10 == 0:
                        print(".", end='')

                    # Readability
                    prev_p = data[solution_index][..., step - 1]
                    current_p = data[solution_index][..., step]
                    next_p = data[solution_index][..., step + 1]

                    # Update grid
                    for x in range(n_x):
                        for y in range(n_y):
                            if k_map[x, y] > 0:
                                rb = g_map[x, y] * (c * dt) / (2 * dx) * (4 - k_map[x, y])
                                next_p[x, y] = 1 / (1 + rb) * \
                                               ((2 - 0.5 * k_map[x, y]) * current_p[x, y] +
                                                0.5 * (current_p[x + 1, y] + current_p[x - 1, y] +
                                                       current_p[x, y + 1] + current_p[x, y - 1]) +
                                                (rb - 1) * prev_p[x, y])

                    # Inject source
                    next_p[ic_pos_samples[0], ic_pos_samples[1]] += u_in[step + 1]

                # Done with this simulation - print time in ms.
                print(f"done.\nTook {round((datetime.now() - t0).total_seconds() * 1000)}ms.")

                # Increment solution index and handle saving
                solution_index += 1
                if solution_index >= sims_per_file or (sim_id >= num_simulations - 1 and mesh_id + 1 == num_meshes):
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
                                                  "mesh_id": mesh_id}
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
        util.create_folder(file_path)

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
        util.create_folder(file_path)

        # Save metadata
        full_path = f"{file_path}meta.json"
        util.save_json(f"{file_path}meta.json", metadata)
        print(f"Saved FDTD metadata as '{full_path}'.\n")

    # Creates a mesh with randomly reflective surfaces
    # TODO: better mesh generation
    def create_mesh(self,
                    dim_lengths_samples,
                    mode="l",
                    round_bc_coeffs=cfg.FDTD_ROUND_BC_COEFFS):
        # Create empty domain
        x_l = dim_lengths_samples[0]
        y_l = dim_lengths_samples[1]
        mesh = np.full([x_l, y_l], -1,
                       dtype=self.manager.metadata["dtype"])

        # Create walls
        if mode == "square":
            for i in range(4):
                bc_abs = np.random.random()
                if round_bc_coeffs:
                    bc_abs = round(bc_abs, 2)

                if i == 0:
                    mesh[0, 1:y_l] = np.full([y_l - 1], bc_abs)
                elif i == 1:
                    mesh[x_l - 1, :y_l - 1] = np.full([y_l - 1], bc_abs)
                elif i == 2:
                    mesh[:x_l - 1, 0] = np.full([x_l - 1], bc_abs)
                elif i == 3:
                    mesh[1:x_l, y_l - 1] = np.full([x_l - 1], bc_abs)
        elif mode == "l":
            x_l2 = int(np.floor(x_l / 2))
            y_l2 = int(np.floor(y_l / 2))

            for i in range(6):
                bc_abs = np.random.random()
                if round_bc_coeffs:
                    bc_abs = round(bc_abs, 2)

                if i == 0:
                    mesh[0, 1:y_l] = np.full([y_l - 1], bc_abs)
                elif i == 1:
                    mesh[:x_l - 1, 0] = np.full([x_l - 1], bc_abs)
                elif i == 2:
                    mesh[x_l - 1, :y_l2] = np.full([y_l2], bc_abs)
                elif i == 3:
                    mesh[x_l2 + 1:, y_l2] = np.full([x_l2], bc_abs)
                elif i == 4:
                    mesh[x_l2, y_l2:y_l - 1] = np.full([y_l2], bc_abs)
                elif i == 5:
                    mesh[1:x_l2 + 1, y_l - 1] = np.full([x_l2], bc_abs)
            mesh[x_l2 + 1:, y_l2 + 1:] = np.zeros([x_l2, y_l2])

        return mesh

    # Save metadata as .json after all simulations are complete.
    def save_mesh(self,
                  mesh,
                  file_name_out,
                  mesh_id):
        # Make folder
        file_path = f"{self.manager.get_proj_path()}fdtd/{file_name_out}/mesh/"
        util.create_folder(file_path)

        # Save mesh data
        full_path = f"{file_path}{mesh_id}.pkl"
        util.save_data(full_path, mesh)
        print(f"Mesh saved to '{full_path}'.")

    # Sample an (x, y) collocation point in real-space using LHC sampling.
    # Ensures the sampled point isn't within a mesh boundary.
    def sample_collocation_point(self, mesh):
        lhc = LatinHypercube(d=2)
        while True:
            # Sample points
            xy_real = lhc.random()[0]

            # Map to real space
            x_len = self.manager.metadata["dim_lengths"][0]
            y_len = self.manager.metadata["dim_lengths"][1]
            xy_real[0] = xy_real[0] * x_len - x_len / 2
            xy_real[1] = xy_real[1] * y_len - y_len / 2

            # Check if in boundary, break if not
            xy_sample = self.manager.util.real_to_sample_pos(xy_pos_real=xy_real,
                                                             dim_lengths_samples=np.shape(mesh))
            if mesh[xy_sample] < 0:
                break

        return xy_real
