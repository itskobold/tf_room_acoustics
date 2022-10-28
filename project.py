import util
from fdtd_sim import FDTD
from render import Renderer
from nn import AcousticNet
import config as cfg
from pathlib import Path
import tensorflow as tf
import numpy as np


# Class to manage simulation and rendering for a project
class ProjectManager:
    # Set project details
    def __init__(self,
                 proj_name,
                 seed=cfg.SEED,
                 dim_lengths=cfg.DIM_LENGTHS,
                 c=cfg.C,
                 dtype=cfg.DTYPE):
        # Set project name and init seed
        self.proj_name = proj_name
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Make master project directory
        Path("projects/").mkdir(parents=True, exist_ok=True)

        # Load project metadata if project already exists
        proj_path = f"projects/{self.proj_name}/"
        if Path(proj_path).exists():
            print(f"Loading project '{self.proj_name}'...")
            self.metadata = util.load_json(f"{proj_path}meta.json")
            print(f"Project '{self.proj_name}' loaded.\n")
        # Otherwise make this new project directory and init metadata from args
        else:
            print(f"Creating new project '{self.proj_name}'...")
            Path(proj_path).mkdir(parents=True, exist_ok=True)
            self.metadata = {"seed": seed,
                             "dim_lengths": dim_lengths,
                             "c": c,
                             "dtype": dtype}

            # Save metadata
            util.save_json(f"{proj_path}meta.json", self.metadata)
            print(f"Project '{self.proj_name}' created, metadata saved as '{proj_path}meta.json'.\n")

        # Print metadata
        print("Project metadata:")
        print(self.metadata)
        print("")

        # Init modules
        self.renderer = Renderer(self)
        self.fdtd = FDTD(self)
        self.nn = AcousticNet(self)
        self.util = util.Util(self)

    # Return path to this project folder
    def get_proj_path(self):
        return f"projects/{self.proj_name}/"
