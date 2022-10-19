# Global constants
DTYPE = 'float64'
C = 343.  # Wave propagation speed (meters per second)
DIM_LENGTHS = 1., 1., 0.1  # X, Y, T domain lengths in meters (X, Y) and seconds (T)
IMPULSE_A = 1.  # Impulse amplitude
IMPULSE_R = 0.025  # Impulse variance

# Rendering
ANIM_FPS = 60  # Frame rate of rendered animations
SAMPLE_RATE = 48000  # Sample rate for rendered impulse responses
COLORMAP = 'coolwarm'  # Colormap of rendered animations
MAX_FRAMES = 1000  # Maximum number of frames to animate

# FDTD stuff
FDTD_NUM_IC_POSITIONS = 20  # Number of random IC positions to generate
FDTD_NUM_BC_ABSORPTION_COEFFS = 800  # Number of random sets of BC absorption coeffs to generate
FDTD_F_MAX = 100  # Maximum resolvable frequency from FDTD solution
FDTD_PPW = 100  # Points per wavelength
FDTD_NUM_SOLUTIONS = 1  # Number of FDTD simulations to run

# General neural network options
NN_SEED = 0
NN_HIDDEN_LAYERS = 4
NN_HL_WIDTH = 20
NN_ITERATIONS_ADAM = 10
NN_ITERATIONS_L_BFGS_B = 10
NN_BATCH_SIZE = 128
NN_BIG_BATCH_SIZE = 2048
NN_VALIDATION_SPLIT = 0.1
NN_TEST_SPLIT = 0.01

# Fourier layer options
NN_T_LOOKBACK = 70
NN_DROP_MODES = False
NN_MODES = 8

# ADAM optimizer options
NN_LEARNING_RATE = 0.001
NN_LR_DECAY = True
NN_LR_DECAY_STEPS = 100
NN_LR_DECAY_RATE = 0.5

# L-BFGS-B optimizer options
NN_MAXCOR = 100
NN_GTOL = 1e-12
