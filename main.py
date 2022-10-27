import numpy as np
import util
from project import ProjectManager

RENDER_ANIMS = True
RENDER_IRS = True
RUN_FDTD = False
RUN_FDTD_TEST = False
LOAD_MODEL = True
FIT_ADAM = True
FIT_L_BFGS_B = False

# Init project
PROJ_NAME = "test_0.1s"
SIM_NAME = "ic_bc_rand"
manager = ProjectManager(proj_name=PROJ_NAME)

# Run FDTD
if RUN_FDTD:
    manager.fdtd.run(file_name_out=SIM_NAME)

# Create test data
if RUN_FDTD_TEST:
    manager.fdtd.run(num_solutions=10,
                     file_name_out=f"{SIM_NAME}_test",
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

# Get predictions and save data
num_files = fdtd_test_meta['num_files']
for i in range(num_files):
    test_data = util.load_data(f"{manager.get_proj_path()}fdtd/{SIM_NAME}_test/{i}.pkl")
    pred_data = manager.nn.get_predictions(data=test_data,
                                           file_name_out=f"{SIM_NAME}_{i}")

    # Render animations
    num_simulations = np.shape(test_data)[0]
    if RENDER_ANIMS:
        for test_index in range(num_simulations):
            ic_pos = np.around(fdtd_test_meta[str(i * num_simulations + test_index)]['impulse_xy'], 3)
            bc_abs = np.around(fdtd_test_meta[str(i * num_simulations + test_index)]['boundary_abs'], 3)

            manager.renderer.animate_sound_field(data=test_data[test_index],
                                                 file_name_out=f"{SIM_NAME}_fdtd_{i * num_simulations + test_index}",
                                                 title=f"FDTD:\n"
                                                       f"IC {ic_pos}\n"
                                                       f"BC {bc_abs}")
            manager.renderer.animate_sound_field(data=pred_data[test_index],
                                                 file_name_out=f"{SIM_NAME}_pred_{i * num_simulations + test_index}",
                                                 title=f"Prediction:\n"
                                                       f"IC {ic_pos}\n"
                                                       f"BC {bc_abs}")
            manager.renderer.animate_sound_field_error(true_data=test_data[test_index],
                                                       pred_data=pred_data[test_index],
                                                       file_name_out=f"{SIM_NAME}_test_error_"
                                                                     f"{i * num_simulations + test_index}",
                                                       title=f"Relative error:\n"
                                                             f"IC {ic_pos}\n"
                                                             f"BC {bc_abs}")

    # Render impulse responses
    if RENDER_IRS:
        xy_pos_real = [0.5, 0.25]
        for test_index in range(num_simulations):
            xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                     dim_lengths_samples=fdtd_meta["dim_lengths_samples"])
            manager.renderer.get_impulse_response(data=test_data[test_index],
                                                  xy_pos=xy_pos,
                                                  save=True,
                                                  file_name_out=f"{SIM_NAME}_fdtd_"
                                                                f"{i * num_simulations + test_index}_"
                                                                f"{xy_pos_real}")

            xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                     dim_lengths_samples=np.shape(pred_data))
            manager.renderer.get_impulse_response(data=pred_data[test_index],
                                                  xy_pos=xy_pos,
                                                  save=True,
                                                  file_name_out=f"{SIM_NAME}_pred_"
                                                                f"{i * num_simulations + test_index}_"
                                                                f"{xy_pos_real}")
