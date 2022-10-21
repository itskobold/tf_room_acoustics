import numpy as np

import util
from project import ProjectManager

RENDER_ANIMS = True
RENDER_IRS = True
RUN_FDTD = False
LOAD_MODEL = False
FIT_ADAM = True
FIT_L_BFGS_B = False

# Init project
proj_name = "test_1s"
manager = ProjectManager(proj_name=proj_name)

# Run FDTD
if RUN_FDTD:
    ic_positions = manager.util.sample_collocation_points()
    bc_coeffs = manager.util.sample_boundary_absorption_coeffs()
    manager.fdtd.run(ic_positions=ic_positions,
                     bc_abs_coeffs=bc_coeffs,
                     file_name_out="fdtd")

# Fit model and save
fdtd_dir = "fdtd"
fdtd_meta = util.load_json(f"{manager.get_proj_path()}{fdtd_dir}/meta.json")
if LOAD_MODEL:
    manager.nn.load_model("model")
else:
    if FIT_ADAM:
        manager.nn.fit_model(train_data_dir=fdtd_dir,
                             optimizer_mode="adam",
                             num_files=fdtd_meta["num_files"] - 1)
    if FIT_L_BFGS_B:
        manager.nn.fit_model(train_data_dir=fdtd_dir,
                             optimizer_mode="l-bfgs-b",
                             num_files=fdtd_meta["num_files"] - 1)
    manager.nn.save_model(model_name_out="model")

# Get predictions and save data
test_data = util.load_data(f"{manager.get_proj_path()}{fdtd_dir}/{fdtd_meta['num_files'] - 1}.pkl")
pred_data = manager.nn.get_predictions(data=test_data,
                                       file_name_out="pred")

# Render animations
num_simulations = np.shape(test_data)[0]
if RENDER_ANIMS:
    for test_index in range(num_simulations):
        manager.renderer.animate_sound_field(data=test_data[test_index],
                                             file_name_out=f"fdtd_anim_{test_index}")
        manager.renderer.animate_sound_field(data=pred_data[test_index],
                                             file_name_out=f"pred_anim_{test_index}")
        manager.renderer.animate_sound_field_error(true_data=test_data[test_index],
                                                   pred_data=pred_data[test_index],
                                                   file_name_out=f"test_error_{test_index}")

# Render impulse responses
if RENDER_IRS:
    xy_pos_real = [0.5, 0.25]
    for test_index in range(num_simulations):
        xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                 dim_lengths_samples=manager.fdtd.metadata["dim_lengths_samples"])
        manager.renderer.get_impulse_response(data=test_data[test_index],
                                              xy_pos=xy_pos,
                                              save=True,
                                              file_name_out=f"ir_fdtd_{xy_pos_real}_{test_index}")

        xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                 dim_lengths_samples=manager.nn.metadata["dim_lengths_samples"])
        manager.renderer.get_impulse_response(data=pred_data[test_index],
                                              xy_pos=xy_pos,
                                              save=True,
                                              file_name_out=f"ir_pred_{xy_pos_real}_{test_index}")
