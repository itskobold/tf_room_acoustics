import numpy as np
from project import ProjectManager

RENDER_ANIMS = True
RENDER_IRS = True
LOAD_FDTD = False
LOAD_MODEL = False

# Init project
proj_name = "test"
manager = ProjectManager(proj_name=proj_name)

# Run FDTD
if LOAD_FDTD:
    manager.fdtd.load_data("fdtd")
else:
    ic_positions = manager.util.sample_collocation_points()
    ic_positions[-1] = (0, 0)
    bc_coeffs = manager.util.sample_boundary_absorption_coeffs()
    bc_coeffs[-1] = (0.5, 0.5, 0.5, 0.5)
    manager.fdtd.run(ic_positions=ic_positions,
                     bc_abs_coeffs=bc_coeffs)  # TODO: take Nx4 array
    manager.fdtd.save_data(file_name_out="fdtd")

# Init neural network
manager.nn.init_data_from_fdtd()
manager.nn.init_model()

# Fit model and save
if LOAD_MODEL:
    manager.nn.load_model("model")
else:
    manager.nn.fit_model(optimizer_mode="adam")
    manager.nn.fit_model(optimizer_mode="l-bfgs-b")
    manager.nn.save_model("model")

# Get predictions and save data
manager.nn.get_predictions(test_X=manager.nn.test_X)
manager.nn.save_prediction_data(file_name_out="pred")

num_simulations = np.shape(manager.fdtd.data)[0]
num_predictions = np.shape(manager.nn.data)[0]
diff = num_simulations - num_predictions

# Render animations
if RENDER_ANIMS:
    for test_index in range(num_predictions):
        manager.renderer.animate_sound_field(data=manager.fdtd.data[diff + test_index],
                                             file_name_out=f"fdtd_anim_{test_index}")
        manager.renderer.animate_sound_field(data=manager.nn.data[test_index],
                                             file_name_out=f"pred_anim_{test_index}")
        manager.renderer.animate_sound_field_error(true_data=manager.fdtd.data[diff + test_index],
                                                   pred_data=manager.nn.data[test_index],
                                                   file_name_out=f"test_error_{test_index}")

# Render impulse responses
if RENDER_IRS:
    xy_pos_real = [0.5, 0.25]
    for test_index in range(num_predictions):
        xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                 dim_lengths_samples=manager.fdtd.metadata["dim_lengths_samples"])
        manager.renderer.get_impulse_response(data=manager.fdtd.data[diff + test_index],
                                              xy_pos=xy_pos,
                                              save=True,
                                              file_name_out=f"ir_fdtd_{xy_pos_real}_{test_index}")

        xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                                 dim_lengths_samples=manager.nn.metadata["dim_lengths_samples"])
        manager.renderer.get_impulse_response(data=manager.nn.data[test_index],
                                              xy_pos=xy_pos,
                                              save=True,
                                              file_name_out=f"ir_pred_{xy_pos_real}_{test_index}")
