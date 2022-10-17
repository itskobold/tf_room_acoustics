# Global constants
DTYPE = 'float64'
C = 343.  # Wave propagation speed (meters per second)
X_LEN = 1.  # X dimension length in meters
Y_LEN = 1.  # Y dimension length in meters
TD_LEN = 0.005  # Time domain length in seconds
IMPULSE_A = 1.  # Impulse amplitude
IMPULSE_R = 0.025  # Impulse variance

# Rendering
ANIM_FPS = 60  # Frame rate of rendered animations
SAMPLE_RATE = 48000  # Sample rate for rendered impulse responses
COLORMAP = 'coolwarm'  # Colormap of rendered animations

# Initial and boundary conditions
IMPULSE_X = 0  # Impulse X real position within domain (meters)
IMPULSE_Y = 0  # Impulse Y real position within domain (meters)
BOUNDARY_L_ABS = 0.5  # Left boundary absorption
BOUNDARY_R_ABS = 0.5  # Right boundary absorption
BOUNDARY_B_ABS = 0.5  # Bottom boundary absorption
BOUNDARY_T_ABS = 0.5  # Top boundary absorption
BOUNDARY_ABS = {'left': BOUNDARY_L_ABS,
                'right': BOUNDARY_R_ABS,
                'bottom': BOUNDARY_B_ABS,
                'top': BOUNDARY_T_ABS}

# FDTD stuff
FDTD_F_MAX = 100  # Maximum resolvable frequency from FDTD solution
FDTD_PPW = 100  # Points per wavelength
FDTD_NUM_SOLUTIONS = 1  # Number of FDTD simulations to run

# General network stuff
NN_SEED = 0
NN_HIDDEN_LAYERS = 4
NN_HL_WIDTH = 20
NN_ITERATIONS_ADAM = 1000
NN_ITERATIONS_L_BFGS_B = 1000
NN_BATCH_SIZE = 500
NN_T_LOOKBACK = 10
NN_DROP_MODES = False
NN_MODES = 8

# PDE learning stuff
NN_PDE_DOMAIN_SAMPLES = 5000
NN_PDE_BOUNDARY_SAMPLES = 1000
NN_PDE_INITIAL_SAMPLES = 1000
NN_PDE_TEST_SAMPLES = 1000

# ADAM optimizer
NN_LEARNING_RATE = 0.001
NN_LR_DECAY = True
NN_LR_DECAY_STEPS = 100
NN_LR_DECAY_RATE = 0.5

# L-BFGS-B optimizer
NN_MAXCOR = 100
NN_GTOL = 1e-12
