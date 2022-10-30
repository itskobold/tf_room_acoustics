import util
from project import ProjectManager
import numpy as np

RENDER_ERROR = True
RENDER_ANIMS = True
RENDER_FDTD_ANIMS = False
RENDER_IRS = True
RUN_FDTD = False
RUN_FDTD_TEST = False
LOAD_MODEL = False
GET_PREDICTIONS = True
FIT_ADAM = True
FIT_L_BFGS_B = False

# Init project
PROJ_NAME = "test_0.1s"
SIM_NAME = "ic_bc_dirac"
manager = ProjectManager(proj_name=PROJ_NAME)

# Run FDTD
if RUN_FDTD:
    manager.fdtd.run(file_name_out=SIM_NAME)

# Create test data
if RUN_FDTD_TEST:
    manager.fdtd.run(file_name_out=f"{SIM_NAME}_test",
                     num_solutions=10,
                     round_bc_coeffs=False)

# Load FDTD metadata
fdtd_meta = util.load_json(f"{manager.get_proj_path()}fdtd/{SIM_NAME}/meta.json")
fdtd_test_meta = util.load_json(f"{manager.get_proj_path()}fdtd/{SIM_NAME}_test/meta.json")

# Load model or fit a new model to data
if LOAD_MODEL:
    manager.nn.load_model(SIM_NAME)
else:
    # Init model
    input_shape = util.input_shape_from_data_shape(fdtd_meta["dim_lengths_samples"])
    output_shape = util.output_shape_from_data_shape(fdtd_meta["dim_lengths_samples"])
    manager.nn.init_model(input_shape=input_shape,
                          output_shape=output_shape)

    # Fit and save
    if FIT_ADAM:
        manager.nn.fit_model(train_data_dir=SIM_NAME,
                             optimizer_mode="adam",
                             num_files=fdtd_meta["num_files"])
    if FIT_L_BFGS_B:
        manager.nn.fit_model(train_data_dir=SIM_NAME,
                             optimizer_mode="l-bfgs-b",
                             num_files=fdtd_meta["num_files"])
    manager.nn.save_model(model_name_out=SIM_NAME)

# Loop through test data files
num_files = fdtd_test_meta["num_files"]
errors_mre, titles_mre, file_names_out_mre = [], [], []
errors_rmse, titles_rmse, file_names_out_rmse = [], [], []
for i in range(num_files):
    # Get predictions and save data
    test_data = util.load_data(f"{manager.get_proj_path()}fdtd/{SIM_NAME}_test/{i}.pkl")

    if GET_PREDICTIONS:
        pred_data = manager.nn.get_predictions(data=test_data,
                                               file_name_out=f"{SIM_NAME}_{i}")
    else:
        pred_data = util.load_data(f"{manager.get_proj_path()}pred/{SIM_NAME}_{i}.pkl")

    # Render animations
    num_simulations = np.shape(test_data)[0]
    for test_index in range(num_simulations):
        total_index = i * num_simulations + test_index
        ic_pos = np.around(fdtd_test_meta[str(total_index)]["impulse_xy"], 3)
        bc_abs = np.around(fdtd_test_meta[str(total_index)]["boundary_abs"], 3)

        if RENDER_ERROR:
            errors_mre.append(util.calc_error_heatmap(true_data=test_data[test_index],
                                                      pred_data=pred_data[test_index],
                                                      error_mode="mre"))
            titles_mre.append("MRE$(Data_{true}, Data_{prediction})$\n"
                              f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$\n"
                              f"BC $(l, r, b, t)$: ${util.array_to_formatted_str(bc_abs)}$")
            file_names_out_mre.append(f"{SIM_NAME}_mre_"
                                      f"{total_index}")

            errors_rmse.append(util.calc_error_heatmap(true_data=test_data[test_index],
                                                       pred_data=pred_data[test_index],
                                                       error_mode="rmse"))
            titles_rmse.append("RMSE$(Data_{true}, Data_{prediction})$\n"
                               f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$\n"
                               f"BC $(l, r, b, t)$: ${util.array_to_formatted_str(bc_abs)}$")
            file_names_out_rmse.append(f"{SIM_NAME}_rmse_"
                                       f"{total_index}")

            if RENDER_FDTD_ANIMS:
                manager.renderer.animate_sound_field(data=test_data[test_index],
                                                     file_name_out=f"{SIM_NAME}_fdtd_{total_index}",
                                                     title="Data$_{true}$\n"
                                                           f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$\n"
                                                           f"BC $(l, r, b, t)$: ${util.array_to_formatted_str(bc_abs)}$")
            if RENDER_ANIMS:
                manager.renderer.animate_sound_field(data=pred_data[test_index],
                                                     file_name_out=f"{SIM_NAME}_pred_{total_index}",
                                                     title="Data$_{prediction}$\n"
                                                           f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$\n"
                                                           f"BC $(l, r, b, t)$: ${util.array_to_formatted_str(bc_abs)}$")
                manager.renderer.animate_sound_field_difference(true_data=test_data[test_index],
                                                                pred_data=pred_data[test_index],
                                                                file_name_out=f"{SIM_NAME}_test_error_"
                                                                              f"{total_index}",
                                                                title="Data$_{true}$ - Data$_{prediction}$\n"
                                                                      f"IC $(x, y)$: ${util.array_to_formatted_str(ic_pos)}$\n"
                                                                      f"BC $(l, r, b, t)$: ${util.array_to_formatted_str(bc_abs)}$")

    # Render impulse responses
    if RENDER_IRS:
        xy_pos_real = [0.5, 0.25]
        for test_index in range(num_simulations):
            total_index = i * num_simulations + test_index
            xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                     dim_lengths_samples=fdtd_meta["dim_lengths_samples"])
            manager.renderer.get_impulse_response(data=test_data[test_index],
                                                  xy_pos=xy_pos,
                                                  file_name_out=f"{SIM_NAME}_fdtd_"
                                                                f"{total_index}_"
                                                                f"{xy_pos_real}")

            xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                     dim_lengths_samples=np.shape(pred_data))
            manager.renderer.get_impulse_response(data=pred_data[test_index],
                                                  xy_pos=xy_pos,
                                                  file_name_out=f"{SIM_NAME}_pred_"
                                                                f"{total_index}_"
                                                                f"{xy_pos_real}")

if RENDER_ERROR:
    manager.renderer.plot_error_heatmaps(data=errors_mre,
                                         titles=titles_mre,
                                         file_names_out=file_names_out_mre,
                                         show=False)
    manager.renderer.plot_error_heatmaps(data=errors_rmse,
                                         titles=titles_rmse,
                                         file_names_out=file_names_out_rmse,
                                         show=False)
