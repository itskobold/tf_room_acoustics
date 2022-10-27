import numpy as np
import config as cfg
from pathlib import Path
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
import matplotlib as mpl
mpl.use("Qt5Agg")


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
        print(f"Obtaining impulse response at {x_pos}, {y_pos}...")

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
                            title=None,
                            colormap=cfg.COLORMAP,
                            fps=cfg.ANIM_FPS):
        print(f"Animating sound field '{file_name_out}'.")

        # Do animation
        anim = self.animation(data=data,
                              ax_labels=["X (meters)", "Y (meters)", "Pressure"],
                              title=title,
                              colormap=colormap,
                              fps=fps)

        # Save animation
        self.save_animation(file_name_out, anim, fps)

    # Animate absolute error between true and predicted solutions
    def animate_sound_field_error(self,
                                  true_data,
                                  pred_data,
                                  file_name_out,
                                  title=None,
                                  colormap=cfg.COLORMAP,
                                  fps=cfg.ANIM_FPS):
        print(f"Animating absolute error between true and predicted data '{file_name_out}'.")

        # True and predicted data should be the same shapes
        pred_shape = np.shape(pred_data)
        td_diff = np.shape(true_data)[-1] - pred_shape[-1]
        true_data = true_data[:, :, td_diff:]
        true_shape = np.shape(true_data)
        assert true_shape == pred_shape

        # Do animation
        anim = self.animation(data=np.subtract(true_data, pred_data),
                              ax_labels=["X (meters)", "Y (meters)", "Relative error"],
                              title=title,
                              colormap=colormap,
                              fps=fps,
                              no_lim=True)

        # Save animation
        self.save_animation(file_name_out, anim, fps)

    # Create animation from data in form (x, y, t)
    def animation(self,
                  data,
                  ax_labels,
                  title=None,
                  colormap=cfg.COLORMAP,
                  fps=cfg.ANIM_FPS,
                  max_frames=cfg.MAX_FRAMES,
                  no_lim=False):

        # Create figure
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle(title,
                     fontweight="bold",
                     y=0.9)
        plt.xlabel(ax_labels[0])
        plt.ylabel(ax_labels[1])

        # Create image object
        lims = [None, None] if no_lim \
            else [self.manager.metadata["impulse_a"], -self.manager.metadata["impulse_a"]]
        x_len = self.manager.metadata["dim_lengths"][0] / 2
        y_len = self.manager.metadata["dim_lengths"][1] / 2
        im = plt.imshow(data[0], interpolation="bilinear", cmap=colormap,
                        origin="lower", extent=[-x_len, x_len, -y_len, y_len],
                        vmax=lims[0], vmin=lims[1])

        # Set colorbar
        cb = fig.colorbar(im, shrink=0.5, aspect=8)
        cb.set_label(ax_labels[2])
        plt.tight_layout()

        print("Creating animation.", end="")

        # Function to loop through pressure matrix and update plot data
        num_frames = max_frames if np.shape(data)[-1] > max_frames else np.shape(data)[-1]

        def animate_func(t):
            if t % fps == 0:
                print(".", end="")

            # Need to swap axes here for some reason
            im.set_array(np.swapaxes(data[:, :, t], 0, 1))
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
        # Make impulse response folder
        ir_path = f"{self.manager.get_proj_path()}ir/"
        Path(ir_path).mkdir(parents=True, exist_ok=True)

        # Save as .wav
        wavf.write(f"{ir_path}{file_name_out}.wav", sample_rate, ir)
        print(f"Saved impulse response as '{file_name_out}.wav'.\n")

    # Save animation as .mp4 file
    def save_animation(self,
                       file_name_out,
                       anim,
                       fps=cfg.ANIM_FPS):
        # Make anim folder
        anim_path = f"{self.manager.get_proj_path()}anim/"
        Path(anim_path).mkdir(parents=True, exist_ok=True)

        # Save animation
        file_path = f"{anim_path}{file_name_out}.mp4"
        anim.save(file_path,
                  fps=fps,
                  extra_args=["-vcodec", "libx264"])
        print("done.")  # Finish string from anim loop
        print(f"Saved animation as '{file_path}'.\n")
