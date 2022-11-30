# Global constants
SEED = 0
DTYPE = "float64"
C = 343.  # Wave propagation speed (meters per second)
DIM_LENGTHS = 1.5, 1.5, 0.1  # X, Y, T domain lengths in meters (X, Y) and seconds (T)

# FDTD stuff
FDTD_MESH_SHAPE = "rect"  # Shape of mesh to generate ("rect" or "l")
FDTD_NUM_MESHES_TRAIN = 200  # Number of meshes to generate for training data
FDTD_NUM_MESHES_TEST = 25  # Number of meshes to generate for test data
FDTD_SIMS_PER_BLOCK = 25  # Number of FDTD simulations saved in each block
FDTD_F_MAX = 1000  # Maximum resolvable frequency from FDTD solution
FDTD_PPW = 6  # Points per wavelength
FDTD_BOUNDARY_ABSORPTION = False  # Meshes absorb energy at boundaries if True
FDTD_ROUND_BC_COEFFS = True  # Round BC absorption coefficients to 2 decimal places if True

# General neural network options
NN_HL_WIDTH = 128  # Width of final up projection dense layer
NN_BIG_BATCH_SIZE = 256  # Size of chunks passed to network from blocks
NN_BATCH_SIZE = 32  # Size of batches processed by the network
NN_ITERATIONS = 20  # Number of iterations per big batch
NN_NUM_PASSES = 10  # Number of passes through the training dataset
NN_VALIDATION_SPLIT = 0.01  # How much of the dataset to use for validation during training (between 0 and 1)
NN_PAD_DATA = False  # Pad data with zeros if True or drop excess data from datasets if False
NN_SHUFFLE_PASSES = 10  # Number of times datasets should be shuffled together during data preparation

# Fourier layer options
FNO_HIDDEN_LAYERS = 4  # Number of Fourier layers in the network
FNO_T_LOOKBACK = 32  # Size of time steps FNO takes as input and outputs
FNO_WIDTH = 20  # Width of Fourier layers
FNO_PADDING = 0  # Padding around data passed to Fourier layers to address boundary periodicity
FNO_MODES = 8  # Number of modes to keep from FFT.  Set to -1 to drop no modes

# ADAM optimizer options
ADAM_LEARNING_RATE = 0.001
ADAM_LR_DECAY_STEPS = 1000
ADAM_LR_DECAY_RATE = 0.5

# Rendering
ANIM_FPS = 30  # Frame rate of rendered animations
ANIM_MAX_FRAMES = 200  # Maximum number of frames to animate
ANIM_COLORMAP = "coolwarm"  # Colormap of rendered animations
MESH_ANIM_COLORMAP = "viridis"  # Colormap of mesh in rendered animations
ERROR_COLORMAP = "inferno"  # Colormap of error heatmaps
ERROR_HEATMAP_MODE = "mae"  # Default method of calculating error for heatmaps
