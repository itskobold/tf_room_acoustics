# Global constants
DTYPE = 'float64'
C = 343.  # Wave propagation speed (meters per second)
DIM_LENGTHS = 1., 1., 0.01  # X, Y, T domain lengths in meters (X, Y) and seconds (T)
IMPULSE_A = 1.  # Impulse amplitude
IMPULSE_R = 0.025  # Impulse variance

# Rendering
ANIM_FPS = 60  # Frame rate of rendered animations
SAMPLE_RATE = 48000  # Sample rate for rendered impulse responses
COLORMAP = 'coolwarm'  # Colormap of rendered animations
MAX_FRAMES = 1000  # Maximum number of frames to animate

# FDTD stuff
FDTD_NUM_IC_POSITIONS = 50  # Number of random IC positions to generate
FDTD_NUM_BC_ABSORPTION_COEFFS = 1  # Number of random sets of BC absorption coeffs to generate
FDTD_F_MAX = 100  # Maximum resolvable frequency from FDTD solution
FDTD_PPW = 100  # Points per wavelength
FDTD_SOLUTIONS_PER_FILE = 5  # Number of FDTD simulations saved in each file

# General neural network options
NN_SEED = 0
NN_HIDDEN_LAYERS = 4
NN_HL_WIDTH = 20
NN_ITERATIONS_ADAM = 100
NN_ITERATIONS_L_BFGS_B = 20
NN_BATCH_SIZE = 64
NN_BIG_BATCH_SIZE = 512
NN_NUM_PASSES = 1
NN_VALIDATION_SPLIT = 0.1

# Fourier layer options
NN_T_LOOKBACK = 10
NN_DROP_MODES = False
NN_MODES = 8

# ADAM optimizer options
NN_LEARNING_RATE = 0.001
NN_LR_DECAY = True
NN_LR_DECAY_STEPS = 100
NN_LR_DECAY_RATE = 0.5

# L-BFGS-B optimizer options
NN_MAXCOR = 80
NN_GTOL = 1e-12
