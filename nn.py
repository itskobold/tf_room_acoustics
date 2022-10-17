import config as cfg
import numpy as np
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
        self.data = None
        self.dataset = None
        self.padding = None
        self.input_shape = None
        self.output_shape = None

        print("Initializing neural network...")

        # Handle backend stuff
        tf.keras.backend.set_floatx(self.dtype)
        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

    # Load prediction data from file
    def load_data(self, file_name):
        with open(f"{self.manager.get_proj_path()}/{file_name}.pkl", "rb") as inp:
            self.data = pickle.load(inp)
        # with open(f'{self.manager.get_proj_path()}/{file_name}_meta.json', 'r') as inp:
        #    self.metadata = json.load(inp)

    # Saves prediction data and metadata
    def save_data(self,
                  file_name_out):
        # Save prediction data
        file_path = f"{self.manager.get_proj_path()}/{file_name_out}.pkl"
        with open(file_path, "wb") as outp:
            pickle.dump(self.data, outp, pickle.HIGHEST_PROTOCOL)
        print(f'Saved prediction data to "{file_path}".')

        # Save metadata
        # file_path = f'{self.manager.get_proj_path()}/{file_name_out}_meta.json'
        # with open(file_path, 'w') as outp:
        #    json.dump(self.metadata, outp)
        # print(f'Saved prediction metadata to "{file_path}".\n')

    # Init data from FDTD simulation
    def init_data(self,
                  t_lookback=cfg.NN_T_LOOKBACK,
                  batch_size=cfg.NN_BATCH_SIZE):
        if self.manager.fdtd.data is None:
            print("Could not initialize neural network data: no FDTD data loaded in module.")

        # Create training dataset
        self.dataset, self.input_shape, self.output_shape = \
            self.create_training_data_from_fdtd_data(t_lookback=t_lookback,
                                                     batch_size=batch_size)

    # Create dataset from data
    @staticmethod
    def create_dataset(data,
                       t_lookback=cfg.NN_T_LOOKBACK,
                       batch_size=cfg.NN_BATCH_SIZE):
        # Readability
        x_len_samples = data.shape[-3]
        y_len_samples = data.shape[-2]
        td_len_samples = data.shape[-1]

        # Create linearly spaced grid
        x_grid, y_grid = np.meshgrid(np.linspace(0, 1, x_len_samples),
                                     np.linspace(0, 1, y_len_samples))
        xy_grid = np.stack([x_grid, y_grid], axis=-1)

        # Loop through all solutions in data set
        steps_to_predict = td_len_samples - t_lookback
        a = np.zeros([steps_to_predict, x_len_samples, y_len_samples, t_lookback + 2])  # +2 for (x, y) coord
        u = np.zeros([steps_to_predict, x_len_samples, y_len_samples, 1])
        for fdtd_sol in data:
            # Loop through each solution and create training data for each time slice
            for i in range(steps_to_predict):
                # Split data tensor and add to buffers (a, u)
                # Also add (x, y) coordinate to feature labels a
                a[i, :, :, :t_lookback] = fdtd_sol[:, :, i:t_lookback + i]
                a[i, :, :, t_lookback:] = xy_grid
                u[i] = np.expand_dims(fdtd_sol[:, :, t_lookback + i], axis=-1)

        # Convert to tensors
        dataset = tf.data.Dataset.from_tensor_slices((a, u)).batch(batch_size=batch_size,
                                                                   drop_remainder=False)

        # Return split data tensors and input/output shapes
        return dataset, a.shape[1:], u.shape[1:]

    # Create training data from FDTD data
    def create_training_data_from_fdtd_data(self,
                                            t_lookback=cfg.NN_T_LOOKBACK,
                                            batch_size=cfg.NN_BATCH_SIZE):
        return self.create_dataset(data=self.manager.fdtd.data,
                                   t_lookback=t_lookback,
                                   batch_size=batch_size)

    # Create tensorflow model
    def init_model(self,
                   num_hidden_layers=cfg.NN_HIDDEN_LAYERS,
                   width=cfg.NN_HL_WIDTH,
                   drop_modes=cfg.NN_DROP_MODES,
                   modes=cfg.NN_MODES):
        print("Initializing neural network model...")

        # Create layers
        layers = [tf.keras.Input(self.input_shape)]
#
        # Append up projection layer (linear, no activation)
        layers.append(tf.keras.layers.Dense(width, activation=None)(layers[-1]))

        # Append FNO layers
        for i in range(num_hidden_layers):
            last_layer = layers[-1]
            fourier_layer = FourierLayer(in_width=width,
                                         out_width=width,
                                         drop_modes=drop_modes,
                                         modes=modes)(last_layer)
            conv = tf.keras.layers.Conv2D(filters=width,
                                          kernel_size=1)(last_layer)
            layers.append(tf.keras.layers.Add()([fourier_layer, conv]))
            layers.append(tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x))(layers[-1]))

        # Append down projection layer (linear, no activation)
        layers.append(tf.keras.layers.Dense(128,
                                            activation=None)(layers[-1]))

        # Append output layer
        layers.append(tf.keras.layers.Dense(self.output_shape[-1],
                                            activation=None)(layers[-1]))

        # Create model and print summary
        self.model = kormos.models.BatchOptimizedModel(inputs=layers[0],
                                                       outputs=layers[-1])
        self.model.summary()

    # Set optimizer and compile model
    # Mode can be 'adam' or 'l-bfgs-b'
    def fit_model(self,
                  optimizer_mode,
                  options=None,
                  iterations=None,
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

            # Same for number of iterations
            if iterations is None:
                iterations = cfg.NN_ITERATIONS_ADAM

            # Set learning rate
            if options["lr_decay"]:
                lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=options["learning_rate"],
                                                                    decay_steps=options["decay_steps"],
                                                                    decay_rate=options["decay_rate"])
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

            # Same for number of iterations
            if iterations is None:
                iterations = cfg.NN_ITERATIONS_L_BFGS_B

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

        if optimizer_mode == "adam":
            self.history = self.model.fit(self.dataset,
                                          batch_size=batch_size,
                                          epochs=iterations)
        else:
            self.history = self.model.fit(self.dataset,
                                          batch_size=batch_size,
                                          epochs=iterations,
                                          method=optimizer_mode,
                                          options=options)

    # Get predictions for a data set
    def get_prediction(self,
                       data,
                       batch_size=cfg.NN_BATCH_SIZE):
        # Prepare dataset from data
        dataset, in_shape, out_shape = self.create_dataset(data)
        assert in_shape == self.input_shape
        assert out_shape == self.output_shape

        # Get prediction
        raw = self.model.predict(dataset,
                                 batch_size=batch_size)

        # Transpose to match source
        self.data = np.transpose(np.squeeze(raw), (1, 2, 0))


# Fourier neural operator layer
class FourierLayer(tf.keras.layers.Layer):
    def __init__(self,
                 in_width=cfg.NN_HL_WIDTH,
                 out_width=cfg.NN_HL_WIDTH,
                 drop_modes=cfg.NN_DROP_MODES,
                 modes=cfg.NN_MODES,
                 batch_size=cfg.NN_BATCH_SIZE):
        super(FourierLayer, self).__init__()
        self.in_width = in_width
        self.out_width = out_width
        self.drop_modes = drop_modes
        self.modes = modes
        self.batch_size = batch_size
        self.scale = 1 / (self.in_width * self.out_width)
        self.w = None
        self.b = None

    def build(self, input_shape):
        # Get shape of weight matrix in real space
        weight_shape = input_shape[1:]

        # Initialize weights
        w_init = self.scale * tf.random.uniform(shape=weight_shape,
                                                dtype=self.dtype)
        self.w = tf.Variable(name="kernel",
                             initial_value=w_init,
                             trainable=True,
                             dtype=self.dtype)

        # Initialize biases
        b_init = tf.zeros_initializer()(shape=weight_shape,
                                        dtype=self.dtype)
        self.b = tf.Variable(name="bias",
                             initial_value=b_init,
                             trainable=True,
                             dtype=self.dtype)

    def call(self, inputs):
        # Fourier transform on inputs, weights and bias weights
        x_ft = tf.signal.rfft2d(inputs)
        w_ft = tf.signal.rfft2d(self.w)
        b_ft = tf.signal.rfft2d(self.b)

        # Multiply Fourier modes with transformed weight matrix and add bias
        xw_ft = tf.add(tf.multiply(x_ft, w_ft), b_ft)

        # Mask out Fourier modes as a regularisation measure
        if self.drop_modes:
            mask = np.zeros(w_ft.shape)
            ones = np.ones([self.modes, self.modes, w_ft.shape[-1]])
            mask[:self.modes, :self.modes, :] = ones
            mask[-self.modes:, :self.modes, :] = ones
            mask[:self.modes, -self.modes:, :] = ones
            mask[-self.modes:, -self.modes:, :] = ones
            mask_tf = tf.convert_to_tensor(mask, dtype="complex128")
            xw_ft = tf.multiply(xw_ft, mask_tf)

        # Inverse FFT and return
        return tf.signal.irfft2d(xw_ft)
