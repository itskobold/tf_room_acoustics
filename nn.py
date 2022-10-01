import config as cfg
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import kormos
import pickle
import json


class AcousticNet:
    def __init__(self,
                 manager,
                 seed=None,
                 dtype=cfg.DTYPE):
        self.manager = manager
        self.seed = seed
        self.dtype = dtype

        # Init variables
        self.model = None
        self.history = None
        self.train_X = None
        self.train_y = None
        self.data = None

        print("Initializing neural network...")

        # Handle backend stuff
        tf.keras.backend.set_floatx(self.dtype)
        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

    # Load prediction data from file
    def load_data(self, file_name):
        with open(f'{self.manager.get_proj_path()}/{file_name}.pkl', 'rb') as inp:
            self.data = pickle.load(inp)
        # with open(f'{self.manager.get_proj_path()}/{file_name}_meta.json', 'r') as inp:
        #    self.metadata = json.load(inp)

    # Saves prediction data and metadata
    def save_data(self,
                  file_name_out):
        # Save prediction data
        file_path = f'{self.manager.get_proj_path()}/{file_name_out}.pkl'
        with open(file_path, 'wb') as outp:
            pickle.dump(self.data, outp, pickle.HIGHEST_PROTOCOL)
        print(f'Saved prediction data to "{file_path}".')

        # Save metadata
        # file_path = f'{self.manager.get_proj_path()}/{file_name_out}_meta.json'
        # with open(file_path, 'w') as outp:
        #    json.dump(self.metadata, outp)
        # print(f'Saved prediction metadata to "{file_path}".\n')

    # Init data from FDTD simulation
    def init_data(self):
        if self.manager.fdtd.data is None:
            print("Could not initialize neural network data: no FDTD data loaded in module.")

        # Create labels for training data
        grid = self.create_x_labels_from_fdtd_data()

        # Create dataset
        df = grid.join(pd.DataFrame({"u": self.manager.fdtd.data.flatten()}))

        # Shuffle data
        df = shuffle(df, random_state=self.seed)

        # Split training and target labels
        self.train_X = df.drop("u", axis=1)
        self.train_y = df["u"]

        # Create placeholder variables
        self.model = None
        self.history = None

    # Create feature labels
    @staticmethod
    def create_x_labels(dims, samples):
        # Create training data from FDTD data
        x_len_half = dims[0] / 2
        y_len_half = dims[1] / 2
        td_len = dims[2]
        x_grid, y_grid, t_grid = np.meshgrid(
            np.linspace(-x_len_half, x_len_half, samples[0]),
            np.linspace(-y_len_half, y_len_half, samples[1]),
            np.linspace(0, td_len, samples[2]))
        return pd.DataFrame({"x": x_grid.flatten(),
                             "y": y_grid.flatten(),
                             "t": t_grid.flatten()})

    # Create feature labels from FDTD data
    def create_x_labels_from_fdtd_data(self):
        return self.create_x_labels(dims=[self.manager.fdtd.metadata["x_len"],
                                          self.manager.fdtd.metadata["y_len"],
                                          self.manager.fdtd.metadata["td_len"]],
                                    samples=[self.manager.fdtd.metadata["x_len_samples"],
                                             self.manager.fdtd.metadata["y_len_samples"],
                                             self.manager.fdtd.metadata["td_len_samples"]])

    # Create tensorflow model
    def init_model(self,
                   num_hidden_layers=cfg.NN_HIDDEN_LAYERS,
                   neurons_per_layer=cfg.NN_HL_WIDTH):
        print("Initializing neural network model...")

        # Create layers
        layers = [tf.keras.Input(len(self.train_X.columns))]
        for i in range(num_hidden_layers):
            layers.append(DenseSine(neurons_per_layer, i)(layers[-1]))
        layers.append(DenseSine(1, num_hidden_layers)(layers[-1]))

        # Create model
        self.model = kormos.models.BatchOptimizedModel(inputs=layers[0],
                                                       outputs=layers[-1])
        self.model.summary()

    # Set optimizer and compile model
    # Mode can be 'adam' or 'l-bfgs-b'
    def fit_model(self,
                  optimizer_mode,
                  options=None,
                  iterations=cfg.NN_ITERATIONS,
                  batch_size=cfg.NN_BATCH_SIZE):
        print(f"Compiling model: optimizer mode is '{optimizer_mode}'...")

        # Using ADAM optimizer
        if optimizer_mode == "adam":
            # Load default options from config if none specified
            if options is None:
                options = {"learning_rate": cfg.NN_LEARNING_RATE,
                           "lr_decay": cfg.NN_LR_DECAY,
                           "decay_steps": cfg.NN_LR_DECAY_STEPS,
                           "decay_rate": cfg.NN_LR_DECAY_RATE}

            # Set learning rate
            if options["lr_decay"]:
                lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=options['learning_rate'],
                                                                    decay_steps=options['decay_steps'],
                                                                    decay_rate=options['decay_rate'])
            else:
                lr = options["learning_rate"]

            # Set optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Using L-BFGS-B optimizer
        elif optimizer_mode == "l-bfgs-b":
            # Load default options from config if none specified
            if options is None:
                options = {"maxcor": cfg.NN_MAXCOR,
                           "gtol": cfg.NN_GTOL}

            # Set optimizer
            optimizer = kormos.optimizers.ScipyBatchOptimizer()

        # Incorrect mode specified
        else:
            print(f"Optimizer mode '{optimizer_mode}' unrecognised, quitting training.")
            return

        # Define loss function and compile
        loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer,
                           loss=loss)

        # Begin training
        print("Starting training.")

        if optimizer_mode == 'adam':
            self.history = self.model.fit(self.train_X,
                                          self.train_y,
                                          batch_size=batch_size,
                                          epochs=iterations)
        else:
            self.history = self.model.fit(self.train_X,
                                          self.train_y,
                                          batch_size=batch_size,
                                          epochs=iterations,
                                          method=optimizer_mode,
                                          options=options)

    # Get predictions for a data set
    def get_prediction(self,
                       data=None,
                       batch_size=cfg.NN_BATCH_SIZE):
        # Predict whole field at source FDTD resolution if no data specified
        if data is None:
            data = self.create_x_labels_from_fdtd_data()

        # Get prediction
        raw = self.model.predict(data,
                                 batch_size=batch_size)

        # Reshape to match source
        self.data = raw.reshape((self.manager.fdtd.metadata["x_len_samples"],
                                 self.manager.fdtd.metadata["y_len_samples"],
                                 self.manager.fdtd.metadata["td_len_samples"]))


# Dense hidden layer with adaptive sinusoidal activation
class DenseSine(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 layer_num,
                 adaptive_activation=cfg.NN_ADAPTIVE_ACTIVATION):
        super(DenseSine, self).__init__()
        self.units = units
        self.layer_num = layer_num
        self.adaptive_activation = adaptive_activation
        self.w = None
        self.b = None
        self.a = None

    def build(self, input_shape):
        # Initialize weights
        w_init = SineInit(self.layer_num)
        self.w = tf.Variable(name="kernel",
                             initial_value=w_init(shape=([input_shape[-1], self.units]),
                                                  dtype=self.dtype),
                             trainable=True,
                             dtype=self.dtype)

        # Initialize biases
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
                             initial_value=b_init(shape=(self.units,),
                                                  dtype=self.dtype),
                             trainable=True,
                             dtype=self.dtype)

        # Initialise activation coefficients
        if self.adaptive_activation:
            self.a = tf.Variable(name="activation_coeff",
                                 initial_value=2,
                                 trainable=True,
                                 dtype=self.dtype)
        else:
            self.a = tf.constant(2, dtype=self.dtype)

    def call(self, inputs):
        w_b = tf.matmul(inputs, self.w) + self.b
        ac = tf.math.multiply(self.a, np.pi)
        return tf.math.sin(tf.math.multiply(w_b, ac))


# Weight initialisation for sinusoidal layers
class SineInit(tf.keras.initializers.Initializer):
    def __init__(self, layer):
        self.layer = layer

    def __call__(self, shape, dtype=None, **kwargs):
        num_input = shape[0]

        if self.layer == 0:
            dist = tf.random.uniform(shape,
                                     minval=-1 / num_input,
                                     maxval=1 / num_input,
                                     dtype=dtype)
        else:
            dist = tf.random.uniform(shape,
                                     minval=-np.sqrt(6 / num_input) / 30,
                                     maxval=np.sqrt(6 / num_input) / 30,
                                     dtype=dtype)
        return dist

    def get_config(self):
        return {"layer": self.layer}
