import util
import config as cfg
from project import ProjectManager
import numpy as np

RENDER_ERROR = True
RENDER_ANIMS = True
RENDER_FDTD_ANIMS = False
RENDER_IRS = False

CREATE_MESHES = False
CREATE_MESHES_TEST = False
RUN_FDTD = False
RUN_FDTD_TEST = False
CREATE_RAW_INPUT_DATA = False
PROCESS_RAW_INPUT_DATA = False

LOAD_MODEL = False
GET_PREDICTIONS = True

# Init project
PROJ_NAME = "square_0.1s"
FDTD_NAME = "no_absorb"
SIM_NAME = "no_absorb"
manager = ProjectManager(proj_name=PROJ_NAME)

# Create meshes
if CREATE_MESHES:
    bc_coeffs = manager.fdtd.sample_bc_coeffs(num_meshes=cfg.FDTD_NUM_MESHES_TRAIN)
    dx, dt = manager.fdtd.get_dx_dt()
    manager.fdtd.create_meshes(file_name_out=FDTD_NAME,
                               dim_lengths_samples=manager.fdtd.get_grid_dims(dx=dx, dt=dt),
                               bc_coeffs=bc_coeffs)

# Create test meshes
if CREATE_MESHES_TEST:
    bc_coeffs = manager.fdtd.sample_bc_coeffs(num_meshes=cfg.FDTD_NUM_MESHES_TEST)
    dx, dt = manager.fdtd.get_dx_dt()
    manager.fdtd.create_meshes(file_name_out=f"{FDTD_NAME}_test",
                               dim_lengths_samples=manager.fdtd.get_grid_dims(dx=dx, dt=dt),
                               bc_coeffs=bc_coeffs)

# Run FDTD
if RUN_FDTD:
    ic_points = manager.fdtd.sample_ic_points(mesh_dir=FDTD_NAME,
                                              num_sims_per_mesh=1)
    manager.fdtd.run(file_name_out=SIM_NAME,
                     mesh_dir=FDTD_NAME,
                     ic_points=ic_points)

# Create test data
if RUN_FDTD_TEST:
    ic_points = manager.fdtd.sample_ic_points(mesh_dir=f"{FDTD_NAME}_test",
                                              num_sims_per_mesh=1)
    manager.fdtd.run(file_name_out=f"{SIM_NAME}_test",
                     mesh_dir=f"{FDTD_NAME}_test",
                     ic_points=ic_points)

# Create raw input data from FDTD data
if CREATE_RAW_INPUT_DATA:
    manager.util.create_raw_input_data(fdtd_dir=FDTD_NAME)

# Shuffle training data for input to network
if PROCESS_RAW_INPUT_DATA:
    manager.util.prepare_raw_data_for_network(fdtd_dir=FDTD_NAME)

# Load model or fit a new model to data
fdtd_meta = util.load_json(f"{manager.get_proj_path()}fdtd/{FDTD_NAME}/meta.json")
if LOAD_MODEL is not None:
    if LOAD_MODEL:
        manager.nn.load_model(SIM_NAME)
    else:
        # Init model
        network_shape = util.network_shape_from_data_shape(fdtd_meta["dim_lengths_samples"])
        manager.nn.init_model(network_shape=network_shape)

        # Fit and save
        manager.nn.fit_model(fdtd_dir=FDTD_NAME,
                             mesh_dir=FDTD_NAME,
                             num_blocks=20)#fdtd_meta["num_files"])
        manager.nn.save_model(model_name_out=SIM_NAME)

# Loop through test data files
fdtd_test_meta = util.load_json(f"{manager.get_proj_path()}fdtd/{FDTD_NAME}_test/meta.json")
num_files = fdtd_test_meta["num_files"]
errors_mae, titles_mae, file_names_out_mae = [], [], []
errors_rmse, titles_rmse, file_names_out_rmse = [], [], []
for i in range(num_files):
    # Get predictions and save data
    test_data = util.load_data(f"{manager.get_proj_path()}fdtd/{FDTD_NAME}_test/{i}.pkl")

    if GET_PREDICTIONS:
        pred_data = manager.nn.get_predictions(data=test_data,
                                               mesh_dir=FDTD_NAME,
                                               file_num=i,
                                               file_name_out=f"{SIM_NAME}_{i}")
    else:
        pred_data = util.load_data(f"{manager.get_proj_path()}pred/{SIM_NAME}_{i}.pkl")

    # Render animations
    num_simulations = np.shape(test_data)[0]
    for test_index in range(num_simulations):
        total_index = i * num_simulations + test_index
        ic_pos = np.around(fdtd_test_meta[str(total_index)]["impulse_xy"], 3)
        mesh_id = fdtd_test_meta[str(total_index)]["i"]
        mesh = util.load_data(f"{manager.get_proj_path()}mesh/{FDTD_NAME}_test/{mesh_id}.mesh")

        if RENDER_ERROR:
            errors_mae.append(util.calc_error_heatmap(true_data=test_data[test_index],
                                                      pred_data=pred_data[test_index],
                                                      error_mode="mre"))
            titles_mae.append("MRE$(Data_{true}, Data_{prediction})$\n"
                              f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$")
            file_names_out_mae.append(f"{SIM_NAME}_mre_"
                                      f"{total_index}")

            errors_rmse.append(util.calc_error_heatmap(true_data=test_data[test_index],
                                                       pred_data=pred_data[test_index],
                                                       error_mode="rmse"))
            titles_rmse.append("RMSE$(Data_{true}, Data_{prediction})$\n"
                               f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$")
            file_names_out_rmse.append(f"{SIM_NAME}_rmse_"
                                       f"{total_index}")

        if RENDER_FDTD_ANIMS:
            manager.renderer.animate_sound_field(data=test_data[test_index],
                                                 mesh=mesh,
                                                 file_name_out=f"{SIM_NAME}_fdtd_{total_index}",
                                                 title="Data$_{true}$\n"
                                                       f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$")
        if RENDER_ANIMS:
            manager.renderer.animate_sound_field(data=pred_data[test_index],
                                                 mesh=mesh,
                                                 file_name_out=f"{SIM_NAME}_pred_{total_index}",
                                                 title="Data$_{prediction}$\n"
                                                       f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$")
            manager.renderer.animate_sound_field_difference(true_data=test_data[test_index],
                                                            pred_data=pred_data[test_index],
                                                            mesh=mesh,
                                                            file_name_out=f"{SIM_NAME}_test_error_"
                                                                          f"{total_index}",
                                                            title="Data$_{true}$ - Data$_{prediction}$\n"
                                                                  f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$")

    # Render impulse responses
    if RENDER_IRS:
        xy_pos_real = [0, 0]
        for test_index in range(num_simulations):
            total_index = i * num_simulations + test_index
            xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                     dim_lengths_samples=fdtd_meta["dim_lengths_samples"])
            manager.renderer.get_impulse_response(data=test_data[test_index],
                                                  xy_pos=xy_pos,
                                                  sample_rate=fdtd_test_meta["sample_rate"],
                                                  file_name_out=f"{SIM_NAME}_fdtd_"
                                                                f"{total_index}_"
                                                                f"{xy_pos_real}")

            xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                     dim_lengths_samples=np.shape(pred_data))
            manager.renderer.get_impulse_response(data=pred_data[test_index],
                                                  xy_pos=xy_pos,
                                                  sample_rate=fdtd_test_meta["sample_rate"],
                                                  file_name_out=f"{SIM_NAME}_pred_"
                                                                f"{total_index}_"
                                                                f"{xy_pos_real}")

if RENDER_ERROR:
    manager.renderer.plot_error_heatmaps(data=errors_mae,
                                         titles=titles_mae,
                                         file_names_out=file_names_out_mae,
                                         show=False)
    manager.renderer.plot_error_heatmaps(data=errors_rmse,
                                         titles=titles_rmse,
                                         file_names_out=file_names_out_rmse,
                                         show=False)
