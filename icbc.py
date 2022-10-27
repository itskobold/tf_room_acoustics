import numpy as np
from util import *

# Boundary condition constants
BOUNDARY_CONST = {f"{SIDE_LEFT}_x0": 1, f"{SIDE_LEFT}_y0": 0, f"{SIDE_LEFT}_x1": 0, f"{SIDE_LEFT}_y1": 1,
                  f"{SIDE_RIGHT}_x0": -1, f"{SIDE_RIGHT}_y0": 0, f"{SIDE_RIGHT}_x1": 0, f"{SIDE_RIGHT}_y1": 1,
                  f"{SIDE_BOTTOM}_x0": 0, f"{SIDE_BOTTOM}_y0": 1, f"{SIDE_BOTTOM}_x1": 1, f"{SIDE_BOTTOM}_y1": 0,
                  f"{SIDE_TOP}_x0": 0, f"{SIDE_TOP}_y0": -1, f"{SIDE_TOP}_x1": 1, f"{SIDE_TOP}_y1": 0}


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
    def absorbing_bc(side_id,
                     boundary_abs,
                     current_p,
                     prev_p,
                     x, y,
                     inversion=True):
        rt2 = np.sqrt(2.0)
        a0 = rt2 / (rt2 + boundary_abs[side_id])
        a1 = 1 / (2.0 + (rt2 * boundary_abs[side_id]))
        a2 = (boundary_abs[side_id] - rt2) / (boundary_abs[side_id] + rt2)
        inv = -1 if inversion else 1

        return a0 * current_p[x + BOUNDARY_CONST[f"{side_id}_x0"], y + BOUNDARY_CONST[f"{side_id}_y0"]] + \
               a1 * inv * (current_p[x + BOUNDARY_CONST[f"{side_id}_x1"], y + BOUNDARY_CONST[f"{side_id}_y1"]] +
                           current_p[x - BOUNDARY_CONST[f"{side_id}_x1"], y - BOUNDARY_CONST[f"{side_id}_y1"]]) + \
               a2 * prev_p[x, y]
