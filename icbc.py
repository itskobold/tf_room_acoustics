import config as cfg
import numpy as np

# Boundary condition constants 
BOUNDARY_CONST = {'left_x0': 1, 'left_y0': 0, 'left_x1': 0, 'left_y1': 1,
                  'right_x0': -1, 'right_y0': 0, 'right_x1': 0, 'right_y1': 1,
                  'bottom_x0': 0, 'bottom_y0': 1, 'bottom_x1': 1, 'bottom_y1': 0,
                  'top_x0': 0, 'top_y0': -1, 'top_x1': 1, 'top_y1': 0}


# Gaussian pulse of a given radius and amplitude
def gaussian_ic(x, y,
                impulse_x=cfg.IMPULSE_X,
                impulse_y=cfg.IMPULSE_Y,
                impulse_r=cfg.IMPULSE_R,
                impulse_a=cfg.IMPULSE_A):
    top_half = (x - impulse_x) ** 2 + (y - impulse_y) ** 2
    bottom_half = 2 * impulse_r ** 2
    return impulse_a * np.e ** -(top_half / bottom_half)


# Absorbing boundary condition: lose energy at edges defined by abs_coeffs
def absorbing_bc(side, boundary_abs, current_p, prev_p, x, y, inversion=True):
    rt2 = np.sqrt(2.0)
    a0 = rt2 / (rt2 + boundary_abs[f'{side}_abs'])
    a1 = 1 / (2.0 + (rt2 * boundary_abs[f'{side}_abs']))
    a2 = (boundary_abs[f'{side}_abs'] - rt2) / (boundary_abs[f'{side}_abs'] + rt2)

    if inversion:
        inv = -1
    else:
        inv = 1

    return a0 * current_p[x + BOUNDARY_CONST[f'{side}_x0'], y + BOUNDARY_CONST[f'{side}_y0']] + \
           a1 * inv * (current_p[x + BOUNDARY_CONST[f'{side}_x1'], y + BOUNDARY_CONST[f'{side}_y1']] +
                       current_p[x - BOUNDARY_CONST[f'{side}_x1'], y - BOUNDARY_CONST[f'{side}_y1']]) + \
           a2 * prev_p[x, y]
