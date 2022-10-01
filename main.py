from project import ProjectManager
from fdtd_sim import FDTD
from render import Renderer

RENDER_ANIMS = True
RENDER_IRS = True

# Init project
proj_name = 'test'
manager = ProjectManager(proj_name=proj_name)

# Run FDTD
manager.fdtd.run()
manager.fdtd.save_data(file_name_out='fdtd')

# Init neural network
manager.nn.init_data()
manager.nn.init_model()

# Fit model and get prediction
manager.nn.fit_model(optimizer_mode='adam')
manager.nn.fit_model(optimizer_mode='l-bfgs-b',
                     iterations=100000)
manager.nn.get_prediction()
manager.nn.save_data(file_name_out='pred')

# Render animations
if RENDER_ANIMS:
    manager.renderer.animate_sound_field(data=manager.fdtd.data,
                                         file_name_out='fdtd_anim')
    manager.renderer.animate_sound_field(data=manager.nn.data,
                                         file_name_out='pred_anim')
    manager.renderer.animate_sound_field_error(file_name_out='test_error')

# Render impulse responses
if RENDER_IRS:
    xy_pos_real = [0.5, 0.25]
    manager.renderer.get_impulse_response(xy_pos_real=xy_pos_real,
                                          save=True,
                                          file_name_out=f'IR real {xy_pos_real}')

    xy_pos_relative = [1, 0.75]
    manager.renderer.get_impulse_response(xy_pos_relative=xy_pos_relative,
                                          save=True,
                                          file_name_out=f'IR relative {xy_pos_relative}')
