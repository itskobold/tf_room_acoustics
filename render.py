import numpy as np
import config as cfg
from pathlib import Path
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
import matplotlib as mpl

import util

mpl.use("Qt5Agg")


# Class for rendering data
class Renderer:
    def __init__(self,
                 manager):
        self.manager = manager

    # Create impulse response from data and saves as .wav if file_name_out is specified.
    # Takes (x, y) position in discretized space.
    def get_impulse_response(self,
                             data,
                             xy_pos,
                             norm=False,
                             file_name_out=None,
                             sample_rate=cfg.IR_SAMPLE_RATE):
        x_pos = xy_pos[0]
        y_pos = xy_pos[1]
        print(f"Obtaining impulse response at {x_pos}, {y_pos}...")

        # Create impulse response for x, y position
        ir = data[x_pos, y_pos, :]

        # Normalise between (-1, 1), preserving zero crossings
        if norm:
            max_pt = 0
            for point in ir:
                abs_pt = abs(point)
                if abs_pt > max_pt:
                    max_pt = abs_pt
            if max_pt != 0:
                norm_coeff = 1 / max_pt
            else:
                norm_coeff = 0
            ir *= norm_coeff

        # Save if needed
        if file_name_out is not None:
            self.save_impulse_response(file_name_out,
                                       ir,
                                       sample_rate=sample_rate)

        # Return scaled impulse response
        return ir

    # Plot pointwise error heatmap between true and predicted data.
    # Get data using util.calc_error_heatmap().
    def plot_error_heatmap(self,
                           data,
                           colormap=cfg.ERROR_COLORMAP,
                           title=None,
                           file_name_out=None,
                           show=True,
                           vmax=None, vmin=None):
        if file_name_out is None and not show:
            print("WARNING: plot is not being saved and shown (no file name given and show = False).")
        else:
            fn_out_buf = "" if file_name_out is None else f" '{file_name_out}'"
            print(f"Plotting error heatmap{fn_out_buf}...")

        # Create figure
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle(title,
                     fontweight="bold",
                     y=0.9)
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")

        # Create image object
        x_len = self.manager.metadata["dim_lengths"][0] / 2
        y_len = self.manager.metadata["dim_lengths"][1] / 2
        im = plt.imshow(data, cmap=colormap, origin="lower",
                        extent=[-x_len, x_len, -y_len, y_len],
                        vmax=vmax, vmin=vmin)

        # Set colorbar
        fig.colorbar(im, shrink=0.5, aspect=8)
        plt.tight_layout()

        # Save/show plot
        if file_name_out is not None:
            self.save_plot(file_name_out=file_name_out,
                           plot=plt)
        if show: plt.show()

    # Plot numerous error heatmaps with the option of using common limits.
    # Data should be an array of heatmaps.
    # titles and file_names_out should be lists of the same length if provided.
    def plot_error_heatmaps(self,
                            data,
                            colormap=cfg.ERROR_COLORMAP,
                            titles=None,
                            file_names_out=None,
                            show=True,
                            common_limits=True):
        data_shape = np.shape(data)
        if titles is not None:
            assert len(titles) == data_shape[0]
        if file_names_out is not None:
            assert len(file_names_out) == data_shape[0]

        if common_limits:
            vmax = np.amax(data)
            vmin = np.amin(data)
        else:
            vmax, vmin = None, None

        for i in range(data_shape[0]):
            title = titles[i] if titles is not None else None
            file_name_out = file_names_out[i] if file_names_out is not None else None
            self.plot_error_heatmap(data[i],
                                    colormap=colormap,
                                    title=title,
                                    file_name_out=file_name_out,
                                    show=show,
                                    vmax=vmax, vmin=vmin)

    # Create animation from sound field data
    def animate_sound_field(self,
                            data,
                            file_name_out,
                            title=None,
                            colormap=cfg.ANIM_COLORMAP,
                            fps=cfg.ANIM_FPS):
        print(f"Animating sound field '{file_name_out}'.")

        # Do animation
        anim = self.animation(data=data,
                              ax_labels=["X (meters)", "Y (meters)", "Pressure"],
                              title=title,
                              colormap=colormap,
                              fps=fps,
                              scale_cb=False)

        # Save animation
        self.save_animation(file_name_out, anim, fps)

    # Animate difference between true and predicted solutions
    def animate_sound_field_difference(self,
                                       true_data,
                                       pred_data,
                                       file_name_out,
                                       title=None,
                                       colormap=cfg.ANIM_COLORMAP,
                                       fps=cfg.ANIM_FPS):
        print(f"Animating difference between true and predicted data '{file_name_out}'.")

        # True and predicted data should be the same shapes
        pred_shape = np.shape(pred_data)
        td_diff = np.shape(true_data)[-1] - pred_shape[-1]
        true_data = true_data[:, :, td_diff:]
        true_shape = np.shape(true_data)
        assert true_shape == pred_shape

        # Do animation
        anim = self.animation(data=true_data - pred_data,
                              ax_labels=["X (meters)", "Y (meters)", "Diff."],
                              title=title,
                              colormap=colormap,
                              fps=fps,
                              scale_cb=True)

        # Save animation
        self.save_animation(file_name_out, anim, fps)

    # Create animation from data in form (x, y, t)
    def animation(self,
                  data,
                  ax_labels,
                  title=None,
                  colormap=cfg.ANIM_COLORMAP,
                  fps=cfg.ANIM_FPS,
                  max_frames=cfg.ANIM_MAX_FRAMES,
                  scale_cb=False):

        # Create figure
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle(title,
                     fontweight="bold",
                     y=0.9)
        plt.xlabel(ax_labels[0])
        plt.ylabel(ax_labels[1])

        # Create image object
        x_len = self.manager.metadata["dim_lengths"][0] / 2
        y_len = self.manager.metadata["dim_lengths"][1] / 2
        im = plt.imshow(data[0], interpolation="spline36", cmap=colormap,
                        origin="lower", extent=[-x_len, x_len, -y_len, y_len])

        # Set colorbar
        cb = fig.colorbar(im, shrink=0.5, aspect=8)
        cb.ax.set_title(ax_labels[2], y=-0.1)
        plt.tight_layout()

        print("Creating animation.", end="")
        num_frames = max_frames if np.shape(data)[-1] > max_frames else np.shape(data)[-1]

        # Function to loop through pressure matrix and update plot data
        def animate_func(t):
            if t % fps == 0:
                print(".", end="")

            # Get data slice
            data_slice = data[:, :, t]

            # Set image array data - need to swap axes here for some reason
            im.set_array(np.swapaxes(data_slice, 0, 1))

            # Rescale colorbar
            if scale_cb:
                vmin = np.amin(data_slice)
                vmax = np.amax(data_slice)
            else:
                vmin, vmax = -1, 1
            lim = max([vmin, vmax])
            im.set_clim(vmax=lim, vmin=-lim)
            return [im]

        # Create animation
        anim = mpl_anim.FuncAnimation(fig,
                                      animate_func,
                                      frames=num_frames,
                                      interval=1000 / fps)  # In ms

        # Return animation object
        return anim

    # Save impulse response as a .wav file.
    def save_impulse_response(self,
                              file_name_out,
                              ir,
                              sample_rate=cfg.IR_SAMPLE_RATE):
        # Make impulse response folder
        ir_path = f"{self.manager.get_proj_path()}ir/"
        Path(ir_path).mkdir(parents=True, exist_ok=True)

        # Save as .wav
        wavf.write(f"{ir_path}{file_name_out}.wav", sample_rate, ir)
        print(f"Saved impulse response as '{file_name_out}.wav'.\n")

    # Save plot as .png file.
    def save_plot(self,
                  file_name_out,
                  plot):
        # Make plot folder
        plot_path = f"{self.manager.get_proj_path()}plot/"
        Path(plot_path).mkdir(parents=True, exist_ok=True)

        # Save plot
        file_path = f"{plot_path}{file_name_out}.png"
        plot.savefig(file_path)
        print(f"Saved plot as '{file_path}'.\n")

    # Save animation as .mp4 file.
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
