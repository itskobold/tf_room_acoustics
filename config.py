# Global constants
SEED = 0
DTYPE = "float64"
C = 343.  # Wave propagation speed (meters per second)
DIM_LENGTHS = 1.5, 1.5, 0.1  # X, Y, T domain lengths in meters (X, Y) and seconds (T)

# Rendering
ANIM_FPS = 30  # Frame rate of rendered animations
ANIM_MAX_FRAMES = 200  # Maximum number of frames to animate
ANIM_COLORMAP = "coolwarm"  # Colormap of rendered animations
MESH_ANIM_COLORMAP = "viridis"  # Colormap of mesh in rendered animations
ERROR_COLORMAP = "inferno"  # Colormap of error heatmaps
ERROR_HEATMAP_MODE = "mae"  # Default method of calculating error

# FDTD stuff
FDTD_MESH_SHAPE = "rect"  # Shape of mesh to generate ("rect" or "l")
FDTD_NUM_MESHES_TRAIN = 1000  # Number of meshes to generate for training data
FDTD_NUM_MESHES_TEST = 10  # Number of meshes to generate for test data
FDTD_SIMS_PER_BLOCK = 5  # Number of FDTD simulations saved in each block
FDTD_F_MAX = 1000  # Maximum resolvable frequency from FDTD solution
FDTD_PPW = 6  # Points per wavelength
FDTD_BOUNDARY_ABSORPTION = False  # Meshes absorb energy at boundaries if True
FDTD_ROUND_BC_COEFFS = True  # Round BC absorption coefficients to 2 decimal places if True

# General neural network options
NN_HIDDEN_LAYERS = 5
NN_HL_WIDTH = 20
NN_OPTIMIZER = "adam"
NN_ITERATIONS = 5
NN_BATCH_SIZE = 64
NN_BIG_BATCH_SIZE = 128
NN_NUM_PASSES = 5
NN_VALIDATION_SPLIT = 0.001

# Fourier layer options
FNO_T_LOOKBACK = 32
FNO_MODES = -1  # Number of modes to keep from FFT.  Set to -1 to drop no modes

# ADAM optimizer options
ADAM_LEARNING_RATE = 0.001
ADAM_LR_DECAY = True
ADAM_LR_DECAY_STEPS = 1000
ADAM_LR_DECAY_RATE = 0.5

# L-BFGS-B optimizer options
BFGS_MAXCOR = 80
BFGS_GTOL = 1e-12
