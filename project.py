from fdtd_sim import FDTD
from render import Renderer
from util import Util
from nn import AcousticNet
from icbc import ICBC
import config as cfg
from pathlib import Path
import json


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
        self.icbc = ICBC(self)
        self.util = Util(self)

        # Make master project directory
        Path("projects/").mkdir(parents=True, exist_ok=True)

        # Load project metadata if project already exists
        proj_path = f"projects/{self.proj_name}"
        if Path(proj_path).exists():
            print(f"Loading project '{self.proj_name}'...")
            with open(f"{proj_path}/proj_meta.json", "r") as inp:
                self.metadata = json.load(inp)

            print(f"Project '{self.proj_name}' loaded.\n")
        # Otherwise make this new project directory and init metadata from args
        else:
            print(f"Creating new project '{self.proj_name}'...")
            Path(proj_path).mkdir(parents=True, exist_ok=True)
            self.metadata = {"x_len": x_len,
                             "y_len": y_len,
                             "td_len": td_len,
                             "c": c,
                             "impulse_a": impulse_a,
                             "impulse_r": impulse_r,
                             "dtype": dtype}

            with open(f"{proj_path}/proj_meta.json", "w") as outp:
                json.dump(self.metadata, outp)
            print(f"Project '{self.proj_name}' created, metadata saved as '{proj_path}/proj_meta.json'.\n")

    # Return path to this project folder
    def get_proj_path(self):
        return f"projects/{self.proj_name}"
