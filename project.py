from fdtd_sim import FDTD
from render import Renderer
from nn import AcousticNet
import config as cfg
from pathlib import Path


# Class to manage simulation and rendering for a project
class ProjectManager:
    # Set project details
    def __init__(self,
                 proj_name,
                 seed=cfg.NN_SEED):
        self.proj_name = proj_name
        self.renderer = Renderer(self)
        self.fdtd = FDTD(self)
        self.nn = AcousticNet(self, seed)

        # Make master project directory
        Path("projects/").mkdir(parents=True, exist_ok=True)

        # Make this project directory
        Path(f"projects/{self.proj_name}").mkdir(parents=True, exist_ok=True)

        print(f"Project '{self.proj_name}' initialized.\n")

    # Return path to this project folder
    def get_proj_path(self):
        return f"projects/{self.proj_name}"
