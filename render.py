import numpy as np
import config as cfg
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
import matplotlib as mpl
mpl.use('Qt5Agg')


# Class for rendering data
class Renderer:
    def __init__(self,
                 manager):
        self.manager = manager

    # Create impulse response from data and saves it if needed
    # Takes (x, y) position in discretized space
    def get_impulse_response(self,
                             data,
                             xy_pos,
                             save=False,
                             file_name_out=None,
                             sample_rate=cfg.SAMPLE_RATE):
        x_pos = xy_pos[0]
        y_pos = xy_pos[1]
        print(f'Obtaining impulse response at {x_pos}, {y_pos}...')

        # Create impulse response for x, y position
        ir = data[x_pos, y_pos, :]

        # Normalise between (-1, 1), preserving zero crossings
        max_pt = 0
        for point in ir:
            abs_pt = abs(point)
            if abs_pt > max_pt:
                max_pt = abs_pt
        if max_pt != 0:
            norm_coeff = 1 / max_pt
        else:
            norm_coeff = 0
        ir_scaled = ir * norm_coeff

        # Save if needed
        if save:
            if file_name_out is not None:
                self.save_impulse_response(file_name_out,
                                           ir_scaled,
                                           sample_rate=sample_rate)
            else:
                print("Couldn't save impulse response: no file name specified.\n")

        # Return scaled impulse response
        return ir_scaled

    # Create animation from sound field data
    def animate_sound_field(self,
                            data,
                            file_name_out,
                            colormap=cfg.COLORMAP,
                            fps=cfg.ANIM_FPS):
        # Do animation
        anim = self.animation(data=data,
                              colormap=colormap,
                              fps=fps)

        # Save animation
        self.save_animation(file_name_out, anim, fps)

    # Animate absolute error between true and predicted solutions
    def animate_sound_field_error(self,
                                  true_data,
                                  pred_data,
                                  file_name_out,
                                  colormap=cfg.COLORMAP,
                                  fps=cfg.ANIM_FPS):
        print("Animating absolute error between predicted and FDTD data.")

        # True and predicted data should be the same shapes
        # TODO: throw exception
        pred_shape = np.shape(pred_data)
        td_diff = np.shape(true_data)[-1] - pred_shape[-1]
        true_data = true_data[:, :, td_diff:]
        true_shape = np.shape(true_data)
        assert true_shape == pred_shape

        # Do animation
        anim = self.animation(data=np.subtract(true_data, pred_data),
                              colormap=colormap,
                              fps=fps,
                              no_lim=True)

        # Save animation
        self.save_animation(file_name_out, anim, fps)

    # Create animation from data in form (x, y, t)
    def animation(self,
                  data,
                  colormap=cfg.COLORMAP,
                  fps=cfg.ANIM_FPS,
                  no_lim=False):

        # Create figure
        fig = plt.figure(figsize=(8, 8))

        # Create image object
        lims = [None, None] if no_lim \
            else [self.manager.metadata["impulse_a"], -self.manager.metadata["impulse_a"]]
        im = plt.imshow(data[0], interpolation='bilinear', cmap=colormap,
                        origin='lower', extent=[-self.manager.metadata["x_len"],
                                                self.manager.metadata["x_len"],
                                                -self.manager.metadata["y_len"],
                                                self.manager.metadata["y_len"]],
                        vmax=lims[0], vmin=lims[1])

        fig.colorbar(im, shrink=0.5, aspect=8)
        print('Creating animation...', end='')

        # Function to loop through pressure matrix and update plot data
        num_frames = np.shape(data)[-1]

        def animate_func(t):
            if t % fps == 0:
                print('.', end='')
            elif t == num_frames - 1:
                print('done.')

            im.set_array(np.swapaxes(data[:, :, t], 0, 1))  # We need to swap axes here for some reason
            return [im]

        # Create animation
        anim = mpl_anim.FuncAnimation(fig,
                                      animate_func,
                                      frames=num_frames,
                                      interval=1000 / fps)  # In ms

        # Return animation object
        return anim

    # Save impulse response as a .wav file
    def save_impulse_response(self,
                              file_name_out,
                              ir,
                              sample_rate=cfg.SAMPLE_RATE):
        # Save as .wav
        wavf.write(f'{self.manager.get_proj_path()}/{file_name_out}.wav', sample_rate, ir)
        print(f'Saved impulse response as "{file_name_out}.wav".\n')

    # Save animation as .mp4 file
    def save_animation(self,
                       file_name_out,
                       anim,
                       fps=cfg.ANIM_FPS):
        file_path = f'{self.manager.get_proj_path()}/{file_name_out}.mp4'
        anim.save(file_path,
                  fps=fps,
                  extra_args=['-vcodec', 'libx264'])
        print(f'Saved animation to "{file_path}".\n')
