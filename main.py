from project import ProjectManager

RENDER_ANIMS = True
RENDER_IRS = True

# Init project
proj_name = "test"
manager = ProjectManager(proj_name=proj_name)

# Run FDTD
manager.fdtd.run()
manager.fdtd.save_data(file_name_out="fdtd")

# Init neural network
manager.nn.init_data()
manager.nn.init_model()

# Fit model and get prediction
manager.nn.fit_model(optimizer_mode="adam")
manager.nn.fit_model(optimizer_mode="l-bfgs-b")
manager.nn.get_prediction(data=manager.fdtd.data)
manager.nn.save_data(file_name_out="pred")

fdtd_index = 0

# Render animations
if RENDER_ANIMS:
    manager.renderer.animate_sound_field(data=manager.fdtd.data[fdtd_index],
                                         file_name_out="fdtd_anim")
    manager.renderer.animate_sound_field(data=manager.nn.data,
                                         file_name_out="pred_anim")
    manager.renderer.animate_sound_field_error(true_data=manager.fdtd.data[fdtd_index],
                                               pred_data=manager.nn.data,
                                               file_name_out="test_error")

# Render impulse responses
if RENDER_IRS:
    xy_pos_real = [0.5, 0.25]
    xy_pos = manager.util.real_to_sample_pos(xy_pos_real,
                                             x_len_samples=manager.fdtd.metadata["x_len_samples"],
                                             y_len_samples=manager.fdtd.metadata["y_len_samples"])
    manager.renderer.get_impulse_response(data=manager.fdtd.data[fdtd_index],
                                          xy_pos=xy_pos,
                                          save=True,
                                          file_name_out=f"IR real {xy_pos_real}")

    xy_pos_relative = [1, 0.75]
    xy_pos = manager.util.relative_to_sample_pos(xy_pos_relative,
                                                 x_len_samples=manager.fdtd.metadata["x_len_samples"],
                                                 y_len_samples=manager.fdtd.metadata["y_len_samples"])
    manager.renderer.get_impulse_response(data=manager.fdtd.data[fdtd_index],
                                          xy_pos=xy_pos,
                                          save=True,
                                          file_name_out=f"IR relative {xy_pos_relative}")
