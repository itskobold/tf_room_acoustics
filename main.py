import numpy as np

from project import ProjectManager
import config as cfg

RENDER_ANIMS = True
RENDER_IRS = True

# Init project
proj_name = "test"
manager = ProjectManager(proj_name=proj_name)

# Run FDTD
ic_positions = manager.util.sample_collocation_points()
ic_positions[-1] = (0, 0)
manager.fdtd.run(ic_positions=ic_positions,
                 bc_abs_coeffs=[cfg.BOUNDARY_ABS])  # TODO: take Nx4 array
manager.fdtd.save_data(file_name_out="fdtd")

# Init neural network
manager.nn.init_data()
manager.nn.init_model()

# Fit model and get prediction
manager.nn.fit_model(optimizer_mode="adam")
manager.nn.fit_model(optimizer_mode="l-bfgs-b")

manager.nn.get_predictions(test_X=manager.nn.test_X)
manager.nn.save_data(file_name_out="pred")

# Render animations
if RENDER_ANIMS:
    num_simulations = np.shape(manager.fdtd.data)[0]
    num_predictions = np.shape(manager.nn.data)[0]
    diff = num_simulations - num_predictions
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
    xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                             x_len_samples=manager.fdtd.metadata["x_len_samples"],
                                             y_len_samples=manager.fdtd.metadata["y_len_samples"])
    manager.renderer.get_impulse_response(data=manager.fdtd.data[-1],
                                          xy_pos=xy_pos,
                                          save=True,
                                          file_name_out=f"IR real {xy_pos_real}")

    xy_pos_relative = [1, 0.75]
    xy_pos = manager.util.relative_to_sample_pos(xy_pos_relative,
                                                 x_len_samples=manager.fdtd.metadata["x_len_samples"],
                                                 y_len_samples=manager.fdtd.metadata["y_len_samples"])
    manager.renderer.get_impulse_response(data=manager.fdtd.data[-1],
                                          xy_pos=xy_pos,
                                          save=True,
                                          file_name_out=f"IR relative {xy_pos_relative}")
