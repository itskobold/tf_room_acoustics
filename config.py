# Global constants
SEED = 0
DTYPE = "float64"
C = 343.  # Wave propagation speed (meters per second)
DIM_LENGTHS = 1., 1., 0.1  # X, Y, T domain lengths in meters (X, Y) and seconds (T)

# Rendering
ANIM_FPS = 30  # Frame rate of rendered animations
ANIM_MAX_FRAMES = 200  # Maximum number of frames to animate
ANIM_COLORMAP = "coolwarm"  # Colormap of rendered animations
ERROR_COLORMAP = "inferno"  # Colormap of error heatmaps
ERROR_HEATMAP_MODE = "mre"  # Default method of calculating error
IR_SAMPLE_RATE = 48000  # Sample rate for rendered impulse responses

# FDTD stuff
FDTD_NUM_SOLUTIONS = 1000  # Default number of FDTD simulations to run
FDTD_F_MAX = 100  # Maximum resolvable frequency from FDTD solution
FDTD_PPW = 100  # Points per wavelength
FDTD_SOLUTIONS_PER_FILE = 5  # Number of FDTD simulations saved in each file
FDTD_ROUND_BC_COEFFS = True  # Round BC absorption coefficients to 2 decimal places if True

# General neural network options
NN_HIDDEN_LAYERS = 4
NN_HL_WIDTH = 20
NN_OPTIMIZER = "adam"
NN_ITERATIONS = 5
NN_BATCH_SIZE = 64
NN_BIG_BATCH_SIZE = 128
NN_NUM_PASSES = 1
NN_VALIDATION_SPLIT = 0.001

# Fourier layer options
FNO_T_LOOKBACK = 32
FNO_MODES = 8  # Number of modes to keep from FFT.  Set to -1 to drop no modes

# ADAM optimizer options
ADAM_LEARNING_RATE = 0.001
ADAM_LR_DECAY = True
ADAM_LR_DECAY_STEPS = 100
ADAM_LR_DECAY_RATE = 0.5

# L-BFGS-B optimizer options
BFGS_MAXCOR = 80
BFGS_GTOL = 1e-12
