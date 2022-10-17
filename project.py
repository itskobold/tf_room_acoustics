from fdtd_sim import FDTD
from render import Renderer
from util import Util
from nn import AcousticNet
import config as cfg
from pathlib import Path


# Class to manage simulation and rendering for a project
class ProjectManager:
    # Set project details
    def __init__(self,
                 proj_name,
                 seed=cfg.NN_SEED,
                 x_len=cfg.X_LEN,
                 y_len=cfg.Y_LEN,
                 td_len=cfg.TD_LEN,
                 c=cfg.C,
                 impulse_a=cfg.IMPULSE_A,
                 impulse_r=cfg.IMPULSE_R,
                 dtype=cfg.DTYPE):
        self.proj_name = proj_name
        self.renderer = Renderer(self)
        self.fdtd = FDTD(self)
        self.nn = AcousticNet(self, seed)
        self.util = Util(self)

        # Global variables shared between FDTD and NN simulations
        self.x_len = x_len
        self.y_len = y_len
        self.td_len = td_len
        self.c = c
        self.impulse_a = impulse_a
        self.impulse_r = impulse_r
        self.dtype = dtype

        # Make master project directory
        Path("projects/").mkdir(parents=True, exist_ok=True)

        # Make this project directory
        Path(f"projects/{self.proj_name}").mkdir(parents=True, exist_ok=True)

        print(f"Project '{self.proj_name}' initialized.\n")

    # Return path to this project folder
    def get_proj_path(self):
        return f"projects/{self.proj_name}"
