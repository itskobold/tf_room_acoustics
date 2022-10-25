import config as cfg
import util
import numpy as np
import tensorflow as tf
import kormos
from sklearn.utils import shuffle
from datetime import datetime
from pathlib import Path


# Class for managing neural networks for solving 2D acoustic wave equation
class AcousticNet:
    def __init__(self,
                 manager):
        self.manager = manager

        # Init variables
        self.model = None
        self.t_lookback = None
        self.train_X = None
        self.train_y = None
        self.val_X = None
        self.val_y = None
        self.test_X = None
        self.test_y = None
        self.metadata = {}

        # Handle backend stuff
        tf.keras.backend.set_floatx(self.manager.metadata["dtype"])

    # Train from simulation data in a directory split into numbered blocks
    # This is the only function that needs to be called when initializing a new network
    # Optimizer mode can be 'adam' or 'l-bfgs-b'
    def fit_model(self,
                  train_data_dir,
                  num_files,
                  optimizer_mode,
                  t_lookback=cfg.NN_T_LOOKBACK,
                  options=None,
                  iterations=None,
                  batch_size=cfg.NN_BATCH_SIZE,
                  big_batch_size=cfg.NN_BIG_BATCH_SIZE,
                  num_passes=cfg.NN_NUM_PASSES,
                  num_hidden_layers=cfg.NN_HIDDEN_LAYERS,
                  width=cfg.NN_HL_WIDTH,
                  drop_modes=cfg.NN_DROP_MODES,
                  modes=cfg.NN_MODES):
        print(f"Compiling model: optimizer mode is '{optimizer_mode}'...")

        # Initialize data and model if not already initialized
        self.metadata["t_lookback"] = t_lookback
        self.init_data(util.load_data(f"{self.manager.get_proj_path()}{train_data_dir}/0.pkl"))
        if self.model is None:
            self.init_model(num_hidden_layers=num_hidden_layers,
                            width=width,
                            drop_modes=drop_modes,
                            modes=modes)

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

        # Loop over numerous passes so entire dataset is passed through relatively evenly
        # as learning rate decays over time.
        initial_epoch = 0
        t0 = datetime.now()
        for pass_num in range(num_passes):
            print(f"Beginning training pass {pass_num + 1}/{num_passes}.")

            # Shuffle order of blocks
            block_ids = np.arange(num_files)
            block_ids = shuffle(block_ids)

            # Loop through all files to train from
            for file_num in range(num_files):
                print(f"Training on data block {block_ids[file_num]} "
                      f"({file_num + 1}/{num_files})...")

                # Load data from file
                self.init_data(
                    util.load_data(f"{self.manager.get_proj_path()}{train_data_dir}/{block_ids[file_num]}.pkl"))

                # Prepare datasets to be passed to model and shuffle data, mixing all simulations together
                train_X = self.prepare_dataset_for_model(self.train_X)
                train_y = self.prepare_dataset_for_model(self.train_y)
                train_X, train_y, = shuffle(train_X, train_y,
                                            random_state=self.manager.metadata["seed"])
                num_big_batches = int(np.ceil(np.shape(train_X)[0] / big_batch_size))

                # Do the same for validation data (if any exists)
                if self.val_X is not None and self.val_y is not None:
                    use_val = True
                    val_X = self.prepare_dataset_for_model(self.val_X)
                    val_y = self.prepare_dataset_for_model(self.val_y)
                    val_X, val_y = shuffle(val_X, val_y,
                                           random_state=self.manager.metadata["seed"])
                    big_batch_size_val = int(np.floor(np.shape(val_X)[0] / num_big_batches))
                else:
                    use_val, val_X, val_y, big_batch_size_val = False, None, None, None

                # Loop through big batches
                for big_batch in range(num_big_batches):
                    print(f"Training big batch {big_batch + 1}/{num_big_batches}...\n")
                    end = -1 if big_batch == num_big_batches - 1 else (big_batch + 1) * big_batch_size

                    # Prepare validation data
                    if use_val:
                        end_val = -1 if big_batch == num_big_batches - 1 else (big_batch + 1) * big_batch_size_val
                        val_data = (val_X[big_batch * big_batch_size_val:end_val],
                                    val_y[big_batch * big_batch_size_val:end_val])
                    else:
                        val_data = None

                    if optimizer_mode == "adam":
                        self.model.fit(train_X[big_batch * big_batch_size:end],
                                       train_y[big_batch * big_batch_size:end],
                                       batch_size=batch_size,
                                       epochs=iterations + initial_epoch,
                                       initial_epoch=initial_epoch,
                                       validation_data=val_data)
                        initial_epoch += iterations
                    else:
                        self.model.fit(train_X[big_batch * big_batch_size:end],
                                       train_y[big_batch * big_batch_size:end],
                                       batch_size=batch_size,
                                       epochs=iterations,
                                       validation_data=val_data,
                                       method=optimizer_mode,
                                       options=options)

        # Add training time to metadata
        training_time = datetime.now() - t0
        self.metadata["training_time"] = training_time.total_seconds() * 1000  # In MS

        # All done
        print(f"Training complete. "
              f"Took {self.manager.util.timedelta_to_str(training_time)}.\n")

    # Load model and weights from file
    def load_model(self,
                   model_dir):
        # Load model and weights
        model_path = f"{self.manager.get_proj_path()}models/{model_dir}/"
        model_json = util.load_json(f"{model_path}/model.json")
        self.model = tf.keras.models.model_from_json(model_json,
                                                     custom_objects={
                                                         "BatchOptimizedModel": kormos.models.BatchOptimizedModel,
                                                         "FourierLayer": FourierLayer})
        self.model.load_weights(model_path)

        # Load metadata
        self.metadata = util.load_json(f"{model_path}/meta.json")
        print(f"Loaded model, weights and metadata from '{model_path}'.\n")

    # Save model and weights to file
    def save_model(self,
                   model_name_out):
        # Make model & parent directory
        model_path = f"{self.manager.get_proj_path()}models/{model_name_out}/"
        Path(model_path).mkdir(parents=True, exist_ok=True)

        # Save model and weights
        util.save_json(f"{model_path}/model.json", self.model.to_json())
        self.model.save_weights(filepath=model_path)

        # Save metadata
        util.save_json(f"{model_path}/meta.json", self.metadata)

        print(f"Saved model, weights and metadata to '{model_path}'.\n")

    # Create datasets from data and set metadata parameters
    def init_data(self,
                  data,
                  val_split=cfg.NN_VALIDATION_SPLIT):
        self.train_X, self.train_y, \
        self.val_X, self.val_y, \
        self.metadata["dim_lengths_samples"], \
        self.metadata["input_shape"], \
        self.metadata["output_shape"] = \
            self.create_datasets(data=data,
                                 val_split=val_split)

    # Create training/validation/testing datasets from an array of simulations
    def create_datasets(self,
                        data,
                        val_split=cfg.NN_VALIDATION_SPLIT):
        # Readability
        x_len_samples = data.shape[-3]
        y_len_samples = data.shape[-2]
        t_len_samples = data.shape[-1]
        num_solutions = data.shape[0]
        t_lookback = self.metadata["t_lookback"]

        # Create linearly spaced grid
        x_grid, y_grid = np.meshgrid(np.linspace(0, 1, x_len_samples),
                                     np.linspace(0, 1, y_len_samples))
        xy_grid = np.stack([x_grid, y_grid], axis=-1)

        # Loop through all solutions in dataset
        steps_to_predict = t_len_samples - t_lookback
        a = np.zeros(
            [num_solutions, steps_to_predict, x_len_samples, y_len_samples, t_lookback + 2])  # +2 for (x, y) coord
        u = np.zeros([num_solutions, steps_to_predict, x_len_samples, y_len_samples, 1])
        for i, fdtd_sol in enumerate(data):
            # Loop through each solution and create training data for each time slice
            for t in range(steps_to_predict):
                # Split data tensor and add to buffers (a, u)
                # Also add (x, y) coordinate to feature labels a
                a[i, t, :, :, :t_lookback] = fdtd_sol[:, :, t:t_lookback + t]
                a[i, t, :, :, t_lookback:] = xy_grid
                u[i, t] = np.expand_dims(fdtd_sol[:, :, t_lookback + t], axis=-1)

        # Split datasets
        val_amt = int(np.ceil(num_solutions * val_split))
        train_amt = num_solutions - val_amt
        assert val_amt + train_amt == num_solutions
        train_X = a[:train_amt]
        train_y = u[:train_amt]
        if val_amt > 0:
            val_X = a[train_amt:train_amt + val_amt]
            val_y = u[train_amt:train_amt + val_amt]
        else:
            val_X, val_y = None, None

        # Return split datasets, dimension length in samples and input/output shapes
        return train_X, train_y, \
               val_X, val_y, \
               (x_len_samples, y_len_samples, steps_to_predict,), \
               a.shape[2:], u.shape[2:]

    # Saves prediction data
    def save_data(self,
                  data,
                  file_name_out):
        file_path = f"{self.manager.get_proj_path()}pred/{file_name_out}.pkl"
        util.save_data(file_path, data)
        print(f'Saved prediction data as {file_path}".\n')

    # Wrapper to pass data into model, stacks individual simulations into one big block for batching
    @staticmethod
    def prepare_dataset_for_model(data):
        shape = np.shape(data)
        new_shape = (shape[0] * shape[1],) + shape[2:]
        data_reshaped = np.reshape(data, new_shape)

        return data_reshaped

    # Create tensorflow model
    def init_model(self,
                   num_hidden_layers=cfg.NN_HIDDEN_LAYERS,
                   width=cfg.NN_HL_WIDTH,
                   drop_modes=cfg.NN_DROP_MODES,
                   modes=cfg.NN_MODES):
        print("Initializing neural network model...")

        # Create layers
        layers = [tf.keras.Input(self.metadata["input_shape"])]

        # Append up projection layer (linear, no activation)
        layers.append(tf.keras.layers.Dense(width, activation=None)(layers[-1]))

        # Append FNO layers
        for i in range(num_hidden_layers):
            last_layer = layers[-1]
            fourier_layer = FourierLayer(dtype=self.manager.metadata["dtype"],
                                         in_width=width,
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
        layers.append(tf.keras.layers.Dense(self.metadata["output_shape"][-1],
                                            activation=None)(layers[-1]))

        # Create model and print summary
        self.model = kormos.models.BatchOptimizedModel(inputs=layers[0],
                                                       outputs=layers[-1])
        self.model.summary()

    # Get predictions for a block of simulation data
    # Saves if file_name is not None
    # TODO: evaluate against test_Y if not none
    def get_predictions(self,
                        data,
                        file_name_out=None,
                        batch_size=cfg.NN_BATCH_SIZE,
                        big_batch_size=cfg.NN_BIG_BATCH_SIZE):
        print(f"Obtaining predictions...")

        # Create test X and y datasets
        test_X, test_y, _, _, _, _, _ = self.create_datasets(data=data,
                                                             val_split=0)

        # Prepare test_X data
        num_simulations = np.shape(test_X)[0]
        test_X = self.prepare_dataset_for_model(test_X)

        # Predict in big batches
        num_big_batches = int(np.ceil(np.shape(test_X)[0] / big_batch_size))
        shape = np.shape(test_X)[:-1]
        raw = np.zeros(shape)
        t0 = datetime.now()
        for big_batch in range(num_big_batches):
            print(f"Predicting big batch {big_batch + 1}/{num_big_batches}...\n")

            # Get prediction, drop extra dimension, store in buffer
            end = -1 if big_batch == num_big_batches - 1 else (big_batch + 1) * big_batch_size
            pred = self.model.predict(test_X[big_batch * big_batch_size:end], batch_size=batch_size)
            raw[big_batch * big_batch_size:end] = np.squeeze(pred, axis=-1)

        # Transpose and reshape full buffer to an array of individual solutions
        t1 = datetime.now() - t0
        t1_ms = round(t1.total_seconds() * 1000, 2)
        raw_shape = np.shape(raw)
        new_shape = (num_simulations, int(raw_shape[0] / num_simulations),) + raw_shape[1:]
        data = np.transpose(np.reshape(raw, new_shape), (0, 2, 3, 1))
        print(f"Predictions obtained. Took {t1_ms}ms "
              f"(average {round(t1_ms / num_simulations, 2)}ms per solution).\n")  # In MS

        # Save data
        if file_name_out is not None:
            self.save_data(data,
                           file_name_out=file_name_out)

        # Return predicted data
        return data


# Fourier neural operator layer
class FourierLayer(tf.keras.layers.Layer):
    def __init__(self,
                 dtype,
                 in_width=cfg.NN_HL_WIDTH,
                 out_width=cfg.NN_HL_WIDTH,
                 drop_modes=cfg.NN_DROP_MODES,
                 modes=cfg.NN_MODES,
                 **kwargs):
        super(FourierLayer, self).__init__()
        self.dtype_ = dtype
        self.in_width = in_width
        self.out_width = out_width
        self.drop_modes = drop_modes
        self.modes = modes
        self.scale = 1 / (self.in_width * self.out_width)
        self.w = None
        self.b = None

    def build(self, input_shape):
        # Get shape of weight matrix in real space
        weight_shape = input_shape[1:]

        # Initialize weights
        w_init = self.scale * tf.random.uniform(shape=weight_shape,
                                                dtype=self.dtype_)
        self.w = tf.Variable(name="kernel",
                             initial_value=w_init,
                             trainable=True,
                             dtype=self.dtype_)

        # Initialize biases
        b_init = tf.zeros_initializer()(shape=weight_shape,
                                        dtype=self.dtype_)
        self.b = tf.Variable(name="bias",
                             initial_value=b_init,
                             trainable=True,
                             dtype=self.dtype_)

    def call(self, inputs):
        # Fourier transform on inputs, weights and bias weights
        x_ft = tf.signal.rfft2d(inputs)
        w_ft = tf.signal.rfft2d(self.w)
        b_ft = tf.signal.rfft2d(self.b)

        # Multiply Fourier modes with transformed weight matrix and add bias
        xw_ft = tf.add(tf.multiply(x_ft, w_ft), b_ft)

        # Drop Fourier modes as a regularization measure
        # TODO: fix? investigate
        if self.drop_modes:
            xw_ft_m = tf.zeros([self.modes * 2, self.modes * 2, w_ft.shape[-1]])
            m_tl = xw_ft[:self.modes, :self.modes]
            m_tr = xw_ft[-self.modes:, :self.modes]
            m_bl = xw_ft[:self.modes, -self.modes:]
            m_br = xw_ft[-self.modes:, -self.modes:]
            m_t = tf.concat([m_tl, m_tr], axis=1)
            m_b = tf.concat([m_bl, m_br], axis=1)
            xw_ft = tf.concat([m_t, m_b], axis=0)

            # xw_ft_m[:self.modes, :self.modes] = xw_ft[:self.modes, :self.modes]
            # xw_ft_m[-self.modes:, :self.modes] = xw_ft[-self.modes:, :self.modes]
            # xw_ft_m[:self.modes, -self.modes:] = xw_ft[:self.modes, -self.modes:]
            # xw_ft_m[-self.modes:, -self.modes:] = xw_ft[-self.modes:, -self.modes:]
            # xw_ft = xw_ft_m

            # mask = np.zeros(w_ft.shape)
            # ones = np.ones([self.modes, self.modes, w_ft.shape[-1]])
            # mask[:self.modes, :self.modes, :] = ones
            # mask[-self.modes:, :self.modes, :] = ones
            # mask[:self.modes, -self.modes:, :] = ones
            # mask[-self.modes:, -self.modes:, :] = ones
            # mask_tf = tf.convert_to_tensor(mask, dtype="complex128")
            # xw_ft = tf.multiply(xw_ft, mask_tf)

        # Inverse FFT and return
        return tf.signal.irfft2d(xw_ft)

    # BUG: saving .h5 model with SciPy optimizer breaks as it doesn't have a get_config function
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'in_width': self.in_width,
            'out_width': self.out_width,
            'drop_modes': self.drop_modes,
            'modes': self.modes
        })
        return config
