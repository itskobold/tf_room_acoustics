import config as cfg
import util
import numpy as np
from datetime import datetime
from scipy.stats.qmc import LatinHypercube
from scipy.signal import ricker
import glob


# Class for managing FDTD simulation
class FDTD:
    # Init FDTD settings
    def __init__(self, manager):
        self.manager = manager

    # Run acoustic finite-difference time domain simulation
    def run(self,
            file_name_out,
            mesh_dir,
            ic_points,
            sims_per_block=cfg.FDTD_SIMS_PER_BLOCK,
            f_max=cfg.FDTD_F_MAX,
            ppw=cfg.FDTD_PPW):
        num_meshes = self.count_meshes_in_dir(mesh_dir)
        num_sims = np.shape(ic_points)[1]
        assert num_meshes == np.shape(ic_points)[0]

        print(f"Starting FDTD simulation '{file_name_out}'...")

        # Init metadata dict
        metadata = {}

        # Readability
        c = self.manager.metadata["c"]

        # Work out the dimensions of the FDTD grid
        # Add 2 to each spatial dimension to include boundary nodes
        dx, dt = self.get_dx_dt(f_max=f_max, ppw=ppw)
        n_x, n_y, n_t = self.get_grid_dims(dx=dx, dt=dt)
        metadata["dim_lengths_samples"] = n_x, n_y, n_t
        metadata["sample_rate"] = 1 / dt
        metadata["sims_per_file"] = sims_per_block

        # Init empty array
        def init_array():
            return np.zeros([sims_per_block, n_x, n_y, n_t])

        data = init_array()

        # Prepare counters and start timer for entire computation
        solution_index = 0
        total_solution_index = 0
        file_num = 0
        t0_all = datetime.now()

        # Loop for number of meshes to be generated
        # TODO: better mesh generation
        for i in range(num_meshes):
            for j in range(num_sims):
                # Load mesh
                mesh = util.load_data(f"{self.manager.get_proj_path()}mesh/{mesh_dir}/{i}.mesh")

                # Get IC point
                ic_pos_samples = self.manager.util.real_to_sample_pos(ic_points[i][j],
                                                                      metadata["dim_lengths_samples"])

                print(f"Mesh {i + 1}/{num_meshes}: running FDTD simulation {j + 1}/{num_sims} "
                      f"(IC = {util.array_to_formatted_str(ic_points[i][j])})", end="")

                # Create IC
                u_in = np.zeros(n_t)
                ricker_len = int(np.ceil(5 * metadata["sample_rate"] / f_max))
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
                k_map[1: n_x - 1, 1: n_y - 1] += in_mask[2:n_x, 1:n_y - 1]
                k_map[1: n_x - 1, 1: n_y - 1] += in_mask[0:n_x - 2, 1:n_y - 1]
                k_map[1: n_x - 1, 1: n_y - 1] += in_mask[1:n_x - 1, 2:n_y]
                k_map[1: n_x - 1, 1: n_y - 1] += in_mask[1:n_x - 1, 0:n_y - 2]
                k_map *= in_mask

                # Create gamma map (absorption)
                g_map = np.zeros_like(gam_mask)
                g_map[1: n_x - 1, 1: n_y - 1] += gam_mask[2:n_x, 1:n_y - 1]
                g_map[1: n_x - 1, 1: n_y - 1] += gam_mask[0:n_x - 2, 1:n_y - 1]
                g_map[1: n_x - 1, 1: n_y - 1] += gam_mask[1:n_x - 1, 2:n_y]
                g_map[1: n_x - 1, 1: n_y - 1] += gam_mask[1:n_x - 1, 0:n_y - 2]
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
                if solution_index >= sims_per_block or i + 1 == num_meshes:
                    # Save FDTD data to file
                    self.save_data(data,
                                   file_name_out=file_name_out,
                                   file_num=file_num)

                    # Handle variables and clear pressure matrix
                    file_num += 1
                    solution_index = 0
                    data = init_array()

                # Update simulation metadata
                metadata[total_solution_index] = {"impulse_xy": ic_points[i][j].tolist(),
                                                  "i": i}
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

    # Creates and saves a set of meshes
    def create_meshes(self,
                      file_name_out,
                      dim_lengths_samples,
                      bc_coeffs,
                      mesh_shape=cfg.FDTD_MESH_SHAPE):
        # Assertions
        if mesh_shape == "rect":
            assert np.shape(bc_coeffs)[1] == 4
        elif mesh_shape == "l":
            assert np.shape(bc_coeffs)[1] == 6
        else:
            print("Mesh shape not recognised, meshes will not be created.")
            return
        print("Creating meshes...")

        num_meshes = np.shape(bc_coeffs)[0]
        for i in range(num_meshes):
            mesh = self.create_mesh(dim_lengths_samples,
                                    mesh_shape=mesh_shape,
                                    bc_coeffs=bc_coeffs[i])
            self.save_mesh(mesh,
                           file_name_out=file_name_out,
                           mesh_id=i)

    # Get sampling density of FDTD grid
    def get_dx_dt(self,
                  f_max=cfg.FDTD_F_MAX,
                  ppw=cfg.FDTD_PPW):
        c = self.manager.metadata["c"]
        lmb = np.sqrt(0.5)
        dx = c / f_max / ppw
        dt = lmb * dx / c
        return dx, dt

    # Get size of FDTD grid from real-space (X, Y, T) measurements in meters & seconds
    def get_grid_dims(self, dx, dt):
        n_x = int(np.ceil(self.manager.metadata["dim_lengths"][0] / dx)) + 2
        n_y = int(np.ceil(self.manager.metadata["dim_lengths"][1] / dx)) + 2
        n_t = int(np.ceil(self.manager.metadata["dim_lengths"][2] / dt))
        return n_x, n_y, n_t

    # Creates a mesh with reflective surfaces specified by bc_coeffs
    # TODO: better mesh generation
    def create_mesh(self,
                    dim_lengths_samples,
                    mesh_shape,
                    bc_coeffs):
        # Create empty domain
        x_l = dim_lengths_samples[0]
        y_l = dim_lengths_samples[1]
        mesh = np.full([x_l, y_l], -1,
                       dtype=self.manager.metadata["dtype"])

        # Create walls
        if mesh_shape == "rect":
            for i in range(4):
                if i == 0:
                    mesh[0, 1:y_l] = np.full([y_l - 1], bc_coeffs[i])
                elif i == 1:
                    mesh[x_l - 1, :y_l - 1] = np.full([y_l - 1], bc_coeffs[i])
                elif i == 2:
                    mesh[:x_l - 1, 0] = np.full([x_l - 1], bc_coeffs[i])
                elif i == 3:
                    mesh[1:x_l, y_l - 1] = np.full([x_l - 1], bc_coeffs[i])
        elif mesh_shape == "l":
            x_l2 = int(np.floor(x_l / 2))
            y_l2 = int(np.floor(y_l / 2))

            for i in range(6):
                if i == 0:
                    mesh[0, 1:y_l] = np.full([y_l - 1], bc_coeffs[i])
                elif i == 1:
                    mesh[:x_l - 1, 0] = np.full([x_l - 1], bc_coeffs[i])
                elif i == 2:
                    mesh[x_l - 1, :y_l2] = np.full([y_l2], bc_coeffs[i])
                elif i == 3:
                    mesh[x_l2 + 1:, y_l2] = np.full([x_l2], bc_coeffs[i])
                elif i == 4:
                    mesh[x_l2, y_l2:y_l - 1] = np.full([y_l2], bc_coeffs[i])
                elif i == 5:
                    mesh[1:x_l2 + 1, y_l - 1] = np.full([x_l2], bc_coeffs[i])
            mesh[x_l2 + 1:, y_l2 + 1:] = np.zeros([x_l2, y_l2])

        return mesh

    # Save metadata as .json after all simulations are complete.
    def save_mesh(self,
                  mesh,
                  file_name_out,
                  mesh_id):
        # Make folder
        file_path = f"{self.manager.get_proj_path()}mesh/{file_name_out}/"
        util.create_folder(file_path)

        # Save mesh data
        full_path = f"{file_path}{mesh_id}.mesh"
        util.save_data(full_path, mesh)
        print(f"Mesh saved to '{full_path}'.")

    # Get number of .mesh files in directory
    def count_meshes_in_dir(self,
                            mesh_dir):
        return len(glob.glob1(f"{self.manager.get_proj_path()}mesh/{mesh_dir}/", "*.mesh"))

    # Sample a set of (x, y) collocation points in real-space for a set of meshes.
    def sample_ic_points(self,
                         mesh_dir,
                         num_sims_per_mesh):
        num_meshes = self.count_meshes_in_dir(mesh_dir)
        ic_points = np.zeros([num_meshes, num_sims_per_mesh, 2])
        for i in range(num_meshes):
            mesh = util.load_data(f"{self.manager.get_proj_path()}mesh/{mesh_dir}/{i}.mesh")
            for j in range(num_sims_per_mesh):
                # Avoid duplicate entries
                while True:
                    ic_point = self.sample_collocation_point(mesh)
                    if ic_point not in ic_points:
                        ic_points[i, j] = ic_point
                        break
        return ic_points

    # Sample an (x, y) collocation point in real-space using LHC sampling.
    # Ensures the sampled point isn't within a mesh boundary.
    def sample_collocation_point(self,
                                 mesh):
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

    # Sample a set of boundary absorption coefficients sized depending on mesh shape
    def sample_bc_coeffs(self,
                         num_meshes,
                         mesh_shape=cfg.FDTD_MESH_SHAPE,
                         absorption=cfg.FDTD_BOUNDARY_ABSORPTION,
                         round_bc_coeffs=cfg.FDTD_ROUND_BC_COEFFS):
        def get_bc_abs():
            bc_abs = np.random.random()
            if round_bc_coeffs:
                bc_abs = round(bc_abs, 2)
            return bc_abs

        # Create array
        if mesh_shape == "rect":
            count = 4
        elif mesh_shape == "l":
            count = 6
        else:
            print("Mesh shape not recognised, no boundary coefficients will be produced.")
            return
        bc_coeffs = np.zeros([num_meshes, count])

        # No boundary absorption - return array of zeros
        if not absorption:
            return bc_coeffs

        # Fill with absorption coefficients
        for i in range(num_meshes):
            # Avoid duplicate entries
            while True:
                bc_abs = np.zeros(4)
                for j in range(count):
                    bc_abs[j] = get_bc_abs()
                if bc_abs not in bc_coeffs[i]:
                    bc_coeffs[i] = bc_abs
                    break

        return bc_coeffs
