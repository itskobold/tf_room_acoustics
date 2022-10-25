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
proj_name = "test_0.1s"
manager = ProjectManager(proj_name=proj_name)

# File names
FDTD_DIR = "ic_bc"
FDTD_TEST_DIR = f"{FDTD_DIR}_test"
MODEL_NAME = "ic_bc"
PRED_SUFFIX = "pred_ic_bc_"
ANIM_SUFFIX = "anim_ic_bc_"
IR_SUFFIX = "ir_ic_bc_"

# Run FDTD
if RUN_FDTD:
    ic_positions = manager.util.sample_collocation_points()
    bc_coeffs = manager.util.sample_boundary_absorption_coeffs()
    manager.fdtd.run(ic_positions=ic_positions,
                     bc_abs_coeffs=bc_coeffs,
                     file_name_out=FDTD_DIR)

# Create test data
if RUN_FDTD_TEST:
    ic_positions = manager.util.sample_collocation_points(n=10)
    ic_positions[-1] = np.array([0, 0])
    bc_coeffs = manager.util.sample_boundary_absorption_coeffs(n=10)
    bc_coeffs[-1] = np.array([0, 0, 0, 0])
    bc_coeffs[-2] = np.array([0.5, 0.5, 0.5, 0.5])
    manager.fdtd.run(ic_positions=ic_positions,
                     bc_abs_coeffs=bc_coeffs,
                     file_name_out=FDTD_TEST_DIR)

# Fit model and save
fdtd_meta = util.load_json(f"{manager.get_proj_path()}fdtd/{FDTD_DIR}/meta.json")
fdtd_test_meta = util.load_json(f"{manager.get_proj_path()}fdtd/{FDTD_TEST_DIR}/meta.json")
if LOAD_MODEL:
    manager.nn.load_model(MODEL_NAME)
else:
    if FIT_ADAM:
        manager.nn.fit_model(train_data_dir=FDTD_DIR,
                             optimizer_mode="adam",
                             num_files=fdtd_meta["num_files"])
    if FIT_L_BFGS_B:
        manager.nn.fit_model(train_data_dir=FDTD_DIR,
                             optimizer_mode="l-bfgs-b",
                             num_files=fdtd_meta["num_files"])
    manager.nn.save_model(model_name_out=MODEL_NAME)

# Get predictions and save data
for i in range(fdtd_test_meta['num_files']):
    test_data = util.load_data(f"{manager.get_proj_path()}fdtd/{FDTD_TEST_DIR}/{i}.pkl")
    pred_data = manager.nn.get_predictions(data=test_data,
                                           file_name_out=f"{PRED_SUFFIX}{i}")

    # Render animations
    num_simulations = np.shape(test_data)[0]
    if RENDER_ANIMS:
        for test_index in range(num_simulations):
            manager.renderer.animate_sound_field(data=test_data[test_index],
                                                 file_name_out=f"anim_fdtd_{i}_{test_index}")
            manager.renderer.animate_sound_field(data=pred_data[test_index],
                                                 file_name_out=f"{ANIM_SUFFIX}pred_{i}_{test_index}")
            manager.renderer.animate_sound_field_error(true_data=test_data[test_index],
                                                       pred_data=pred_data[test_index],
                                                       file_name_out=f"{ANIM_SUFFIX}test_error_{i}_{test_index}")

    # Render impulse responses
    if RENDER_IRS:
        xy_pos_real = [0.5, 0.25]
        for test_index in range(num_simulations):
            xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                     dim_lengths_samples=fdtd_meta["dim_lengths_samples"])
            manager.renderer.get_impulse_response(data=test_data[test_index],
                                                  xy_pos=xy_pos,
                                                  save=True,
                                                  file_name_out=f"{IR_SUFFIX}fdtd_{i}_{xy_pos_real}_{test_index}")

            xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                     dim_lengths_samples=manager.nn.metadata["dim_lengths_samples"])
            manager.renderer.get_impulse_response(data=pred_data[test_index],
                                                  xy_pos=xy_pos,
                                                  save=True,
                                                  file_name_out=f"{IR_SUFFIX}pred_{i}_{xy_pos_real}_{test_index}")
