import config as cfg
import numpy as np

# Boundary condition constants 
BOUNDARY_CONST = {'left_x0': 1, 'left_y0': 0, 'left_x1': 0, 'left_y1': 1,
                  'right_x0': -1, 'right_y0': 0, 'right_x1': 0, 'right_y1': 1,
                  'bottom_x0': 0, 'bottom_y0': 1, 'bottom_x1': 1, 'bottom_y1': 0,
                  'top_x0': 0, 'top_y0': -1, 'top_x1': 1, 'top_y1': 0}


# Class for initial and boundary conditions
class ICBC:
    def __init__(self,
                 manager):
        self.manager = manager

    # Gaussian pulse of a given radius and amplitude
    # All coordinates are in real space (meters)
    def gaussian_ic(self,
                    x, y,
                    impulse_xy):
        top_half = (x - impulse_xy[0]) ** 2 + (y - impulse_xy[1]) ** 2
        bottom_half = 2 * self.manager.metadata["impulse_r"] ** 2
        return self.manager.metadata["impulse_a"] * np.e ** -(top_half / bottom_half)

    # Absorbing boundary condition: lose energy at edges defined by boundary_abs
    @staticmethod
    def absorbing_bc(side,
                     boundary_abs,
                     current_p,
                     prev_p,
                     x, y,
                     inversion=True):
        rt2 = np.sqrt(2.0)
        a0 = rt2 / (rt2 + boundary_abs[side])
        a1 = 1 / (2.0 + (rt2 * boundary_abs[side]))
        a2 = (boundary_abs[side] - rt2) / (boundary_abs[side] + rt2)

        if inversion:
            inv = -1
        else:
            inv = 1

        return a0 * current_p[x + BOUNDARY_CONST[f'{side}_x0'], y + BOUNDARY_CONST[f'{side}_y0']] + \
               a1 * inv * (current_p[x + BOUNDARY_CONST[f'{side}_x1'], y + BOUNDARY_CONST[f'{side}_y1']] +
                           current_p[x - BOUNDARY_CONST[f'{side}_x1'], y - BOUNDARY_CONST[f'{side}_y1']]) + \
               a2 * prev_p[x, y]
