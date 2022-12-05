import config as cfg
import util
import numpy as np
import tensorflow as tf
import kormos
from sklearn.utils import shuffle
from datetime import datetime


# Class for managing neural networks for solving 2D acoustic wave equation.
class AcousticNet:
    def __init__(self,
                 manager):
        self.manager = manager

        # Init variables
        self.model = None
        self.metadata = {}

        # Handle backend stuff
        tf.keras.backend.set_floatx(self.manager.metadata["dtype"])

    # Load model and weights from file.
    def load_model(self,
                   model_dir):
        # Load model and weights
        model_path = f"{self.manager.get_proj_path()}models/{model_dir}/"
        model_json = util.load_json(f"{model_path}/model.json")
        self.model = tf.keras.models.model_from_json(model_json,
                                                     custom_objects={
                                                         "BatchOptimizedModel": kormos.models.BatchOptimizedModel,
                                                         "FourierLayer": FourierLayer,
                                                         "ConcatAbsorption": ConcatAbsorption,
                                                         "ConstrainOutput": ConstrainOutput})
        self.model.load_weights(model_path)

        # Load metadata
        self.metadata = util.load_json(f"{model_path}/meta.json")
        print(f"Loaded model, weights and metadata from '{model_path}'.\n")

    # Save model and weights to file.
    def save_model(self,
                   model_name_out):
        # Make model & folder
        model_path = f"{self.manager.get_proj_path()}models/{model_name_out}/"
        util.create_folder(model_path)

        # Save model and weights
        util.save_json(f"{model_path}/model.json", self.model.to_json())
        self.model.save_weights(filepath=model_path)

        # Save metadata
        util.save_json(f"{model_path}/meta.json", self.metadata)

        print(f"Saved model, weights and metadata to '{model_path}'.\n")

    # Create a new Tensorflow model.
    def init_model(self,
                   input_shape,
                   output_shape,
                   fdtd_dir,
                   num_blocks,
                   num_hidden_layers=cfg.FNO_HIDDEN_LAYERS,
                   fno_width=cfg.FNO_WIDTH,
                   dense_width=cfg.NN_HL_WIDTH,
                   t_lookback=cfg.FNO_T_LOOKBACK,
                   t_out_size=cfg.NN_OUTPUT_T_LEN,
                   modes=cfg.FNO_MODES):
        print("Initializing neural network model...")

        # Set metadata parameters
        self.metadata["input_shape"] = input_shape
        self.metadata["mesh_input_shape"] = input_shape[:2] + [1]
        self.metadata["output_shape"] = output_shape
        self.metadata["t_lookback"] = t_lookback
        self.metadata["t_out_size"] = t_out_size

        # Print warning if t_lookback is too short to capture first order reflections from boundaries for sure
        if self.metadata["input_shape"][0] > t_lookback or \
                self.metadata["input_shape"][1] > t_lookback:
            print("\nWARNING: t_lookback might be too short to capture first order reflections from all boundaries! "
                  f"t_lookback = {self.metadata['t_lookback']}, "
                  f"input_shape (X, Y) = {self.metadata['input_shape'][:2]}.")

        # Get dataset mean & variance
        means, vars = [], []
        for i in range(num_blocks):
            data = util.load_data(f"{self.manager.get_proj_path()}fdtd/{fdtd_dir}/input/X/{i}.pkl")
            means.append(np.mean(data))
            vars.append(np.std(data))
        mean = np.mean(means)
        var = np.mean(vars)

        # Create layer for mesh input
        mesh_input = tf.keras.Input(self.metadata["mesh_input_shape"])

        # Create solution input & list for layers
        layers = [tf.keras.Input(self.metadata["input_shape"])]

        # Append normalization layer
        layers.append(tf.keras.layers.Normalization(axis=None,
                                                    mean=mean,
                                                    variance=var)(layers[-1]))

        # Concatenate absorption map with solution layers
        # layers.append(ConcatAbsorption(dtype=self.manager.metadata["dtype"])([layers[-1], mesh_input]))

        # Append up projection layer (linear, no activation)
        layers.append(tf.keras.layers.Dense(fno_width, activation=None)(layers[-1]))

        # Append FNO layers
        for i in range(num_hidden_layers):
            last_layer = layers[-1]
            fourier_layer = FourierLayer(dtype=self.manager.metadata["dtype"],
                                         fft_type="3d",
                                         in_width=fno_width,
                                         t_lookback=t_lookback,
                                         modes=modes)(last_layer)
            conv = tf.keras.layers.Conv2D(filters=fno_width,
                                          kernel_size=1)(last_layer)
            layers.append(tf.keras.layers.Add()([fourier_layer, conv]))
            layers.append(tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x))(layers[-1]))

        # Append down projection layer (linear, no activation)
        layers.append(tf.keras.layers.Dense(dense_width,
                                            activation=None)(layers[-1]))

        # Append output layer
        layers.append(tf.keras.layers.Dense(self.metadata["output_shape"][-1],
                                            activation=None)(layers[-1]))

        # Append transformation layer to constrain predictions within boundaries
        layers.append(ConstrainOutput(dtype=self.manager.metadata["dtype"])([layers[-1], mesh_input]))

        # Create model and print summary
        self.model = kormos.models.BatchOptimizedModel(inputs=[layers[0], mesh_input],
                                                       outputs=layers[-1])
        self.model.summary()

    # Train from simulation data in a directory split into numbered blocks.
    # Remember to create a model with init_model or load one with load_model before fitting.
    def fit_model(self,
                  fdtd_dir,
                  mesh_dir,
                  num_blocks,
                  iterations=cfg.NN_ITERATIONS,
                  learning_rate=cfg.ADAM_LEARNING_RATE,
                  lr_decay=cfg.ADAM_LR_DECAY,
                  decay_steps=cfg.ADAM_LR_DECAY_STEPS,
                  decay_rate=cfg.ADAM_LR_DECAY_RATE,
                  val_split=cfg.NN_VALIDATION_SPLIT,
                  batch_size=cfg.NN_BATCH_SIZE,
                  big_batch_size=cfg.NN_BIG_BATCH_SIZE,
                  num_passes=cfg.NN_NUM_PASSES):
        # Loop over numerous passes so entire dataset is passed through relatively evenly
        # as learning rate decays over time.
        fdtd_path = f"{self.manager.get_proj_path()}fdtd/{fdtd_dir}/"
        mesh_path = f"{self.manager.get_proj_path()}mesh/{mesh_dir}/"
        meta = util.load_json(f"{fdtd_path}meta.json")
        initial_epoch = 0

        loss_fn = tf.keras.losses.MeanSquaredError()
        train_metric = tf.keras.metrics.MeanSquaredError()
        val_metric = tf.keras.metrics.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer,
                           loss=loss_fn,
                           metrics=[train_metric])

        t0 = datetime.now()
        for pass_num in range(num_passes):
            print(f"Beginning training pass {pass_num + 1}/{num_passes}.")

            # Shuffle order of blocks
            block_ids = np.arange(num_blocks)
            block_ids = shuffle(block_ids)

            # Loop through all files to train from
            for file_num in range(num_blocks):
                block_id = block_ids[file_num]
                print(f"Training on block {file_num + 1}/{num_blocks} (ID {block_id})...")

                # Load X, X_mesh and y data from "{fdtd_dir}/input/"
                X_data = util.load_data(f"{fdtd_path}input/X/{block_id}.pkl")
                mesh_X_ids = util.load_data(f"{fdtd_path}input/mesh_X/{block_id}.pkl")
                y_data = util.load_data(f"{fdtd_path}input/y/{block_id}.pkl")

                # Split datasets
                val_amt = int(np.ceil(X_data.shape[0] * val_split))
                train_amt = X_data.shape[0] - val_amt
                train_X = X_data[:train_amt]
                train_mesh_X_ids = mesh_X_ids[:train_amt]
                train_y = y_data[:train_amt]

                # Shuffle datasets
                train_X, train_mesh_X_ids, train_y, = shuffle(train_X, train_mesh_X_ids, train_y)

                # Load meshes
                train_mesh_X = np.zeros([train_mesh_X_ids.shape[0],
                                         meta["dim_lengths_samples"][0],
                                         meta["dim_lengths_samples"][1],
                                         1])
                for i, mesh_id in enumerate(train_mesh_X_ids):
                    train_mesh_X[i] = np.expand_dims(util.load_data(f"{mesh_path}{int(mesh_id)}.mesh"), axis=-1)

                # Loop through big batches
                num_big_batches = int(np.ceil(train_X.shape[0] / big_batch_size))
                #for big_batch in range(num_big_batches):
                #    print(f"Training big batch {big_batch + 1}/{num_big_batches}...\n")

                # Create train dataset tensor
                train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_mesh_X, train_y)) \
                    .batch(batch_size)

                # Create validation data
                if val_amt > 0:
                    val_X = X_data[train_amt:train_amt + val_amt]
                    val_mesh_X_ids = mesh_X_ids[train_amt:train_amt + val_amt]
                    val_y = y_data[train_amt:train_amt + val_amt]

                    # Load meshes
                    val_mesh_X = np.zeros([val_mesh_X_ids.shape[0],
                                           meta["dim_lengths_samples"][0],
                                           meta["dim_lengths_samples"][1],
                                           1])
                    for i, mesh_id in enumerate(val_mesh_X_ids):
                        val_mesh_X[i] = np.expand_dims(util.load_data(f"{mesh_path}{int(mesh_id)}.mesh"), axis=-1)

                    # Create data tensor
                    val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_mesh_X, val_y)) \
                        .batch(batch_size)
                else:
                    val_dataset = None

                # Train over number of iterations
                for i in range(iterations):
                    # Train each batch
                    for step, (train_batch_X, train_mesh_batch_X, train_batch_y) in enumerate(train_dataset):
                        with tf.GradientTape() as tape:
                            logits = self.model([train_batch_X, train_mesh_batch_X], training=True)
                            losses = loss_fn(train_batch_y, logits)
                        grads = tape.gradient(losses, self.model.trainable_weights)

                        # Decay learning rate
                        if lr_decay:
                            lr = self.inverse_time_decay(initial_epoch + i,
                                                         learning_rate=learning_rate,
                                                         decay_steps=decay_steps,
                                                         decay_rate=decay_rate)
                        else:
                            lr = learning_rate

                        # Optimize network weights
                        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                        train_metric.update_state(train_batch_y, logits)

                    # Evaluate validation dataset
                    if val_dataset is not None:
                        for val_batch_X, val_mesh_batch_X, val_batch_y in val_dataset:
                            val_logits = self.model([val_batch_X, val_mesh_batch_X], training=False)
                            val_metric.update_state(val_batch_y, val_logits)
                        str_app = f", Val = {val_metric.result():.8e}"
                        val_metric.reset_states()
                    else:
                        str_app = ""

                    print(f"It: {initial_epoch + i}, Loss = {train_metric.result():.8e}{str_app}")
                    train_metric.reset_states()

                # Update initial epoch and clear train dataset tensor
                initial_epoch += iterations
                del train_dataset
                del val_dataset

        # Add training time to metadata
        training_time = datetime.now() - t0
        self.metadata["training_time"] = training_time.total_seconds() * 1000  # In MS

        # All done
        print(f"Training complete. "
              f"Took {util.timedelta_to_str(training_time)}.\n")

    # Saves prediction data
    def save_data(self,
                  data,
                  file_name_out):
        # Make folder
        file_path = f"{self.manager.get_proj_path()}pred/"
        util.create_folder(file_path)

        # Save data
        full_path = f"{file_path}{file_name_out}.pkl"
        util.save_data(full_path, data)
        print(f'Saved prediction data as {full_path}".\n')

    # Get predictions for a block of simulation data
    # Saves if file_name is not None
    def get_predictions(self,
                        block,
                        mesh_dir,
                        file_num,
                        file_name_out=None,
                        batch_size=cfg.NN_BATCH_SIZE,
                        pad_data=cfg.NN_PAD_DATA):
        print(f"Obtaining predictions...")
        # Readability
        t_lookback = self.metadata["t_lookback"]
        t_out_size = self.metadata["t_out_size"]
        x_len_samples = block.shape[-3]
        y_len_samples = block.shape[-2]
        t_len_samples = block.shape[-1]
        sims_per_block = block.shape[0]
        steps_to_predict = int((t_len_samples - t_lookback) / t_out_size)

        # Pad or trim data
        if pad_data:
            padding_amt = t_out_size - (np.shape(block)[-1] % t_out_size)
            block = np.pad(block, [[0, 0], [0, 0], [0, 0], [0, padding_amt]])
        elif t_out_size > 1:
            block = block[..., :-(np.shape(block)[-1] % t_out_size)]

        # Only take first t_lookback steps from true data
        pred_data = np.zeros_like(block)
        pred_data[..., :t_lookback] = block[..., :t_lookback]

        # Buffers only take 1 step at a time
        a = np.zeros([sims_per_block, x_len_samples, y_len_samples, t_lookback])
        a_mesh = np.zeros([sims_per_block, x_len_samples, y_len_samples, 1])

        # Load initial data and mesh into buffers
        for i, fdtd_data in enumerate(block):
            mesh = util.load_data(f"{self.manager.get_proj_path()}mesh/{mesh_dir}/"
                                  f"{sims_per_block * file_num + i}.mesh")
            a[i] = fdtd_data[..., :t_lookback]
            a_mesh[i] = np.expand_dims(mesh, axis=-1)

        # Predict & set next data for input as newly predicted data
        start = int(t_lookback / t_out_size)
        for step in range(start, steps_to_predict + 1):
            pred = self.model.predict([a, a_mesh], batch_size=batch_size)
            pred_data[..., step * t_out_size:step * t_out_size + t_out_size] = pred
            if t_lookback - t_out_size > 0:
                a[..., :t_lookback - t_out_size] = a[..., t_out_size:]
            a[..., t_lookback - t_out_size:] = pred#block[..., step * t_out_size:step * t_out_size + t_out_size]  # pred

        # Save data
        if file_name_out is not None:
            self.save_data(pred_data,
                           file_name_out=file_name_out)

        # Return predicted data
        return pred_data

    # Compute inverse time decay of learning rate
    @staticmethod
    def inverse_time_decay(step,
                           learning_rate=cfg.ADAM_LEARNING_RATE,
                           decay_steps=cfg.ADAM_LR_DECAY_STEPS,
                           decay_rate=cfg.ADAM_LR_DECAY_RATE):
        return learning_rate / (1 + decay_rate * step / decay_steps)


# Fourier neural operator layer
class FourierLayer(tf.keras.layers.Layer):
    def __init__(self,
                 fft_type,
                 dtype,
                 fno_width=cfg.FNO_WIDTH,
                 t_lookback=cfg.FNO_T_LOOKBACK,
                 modes=cfg.FNO_MODES,
                 padding=cfg.FNO_PADDING,
                 **kwargs):
        super(FourierLayer, self).__init__()
        self.dtype_ = dtype
        self.fft_type = fft_type
        self.fno_width = fno_width
        self.t_lookback = t_lookback
        self.modes = modes
        self.padding = padding
        self.scale = 1 / (self.fno_width * self.t_lookback)

    def build(self, input_shape):
        # Init weights as randomly sampled complex numbers
        def init_weights(shape):
            w_init = self.scale * tf.random.uniform(shape=shape,
                                                    dtype=self.dtype_)
            return tf.Variable(name="kernel",
                               initial_value=w_init,
                               trainable=True,
                               dtype="float64")

        # Init biases as complex zeros
        def init_bias(shape):
            b_init = tf.zeros(shape=shape,
                              dtype=self.dtype_)
            return tf.Variable(name="bias",
                               initial_value=b_init,
                               trainable=True,
                               dtype="float64")

        # Initialize weights and biases
        if self.modes > 0:
            wb_shape = [self.modes,
                        self.modes]
            if self.fft_type == "3d":
                wb_shape.append(self.modes)

            self.w_tl = init_weights(wb_shape)
            self.w_tr = init_weights(wb_shape)
            self.w_bl = init_weights(wb_shape)
            self.w_br = init_weights(wb_shape)
            self.b_tl = init_bias(wb_shape)
            self.b_tr = init_bias(wb_shape)
            self.b_bl = init_bias(wb_shape)
            self.b_br = init_bias(wb_shape)
        else:
            # Transform a dummy tensor of zeros to get the weight shape
            x_shape = (input_shape[1:-1] + [self.fno_width, ]).as_list()
            x = tf.complex(tf.zeros(x_shape),
                           tf.zeros(x_shape))
            if self.fft_type == "2d":
                x_ft = tf.signal.fft2d(x)
            else:
                x_ft = tf.signal.fft3d(x)
            wb_shape = tf.shape(x_ft)

            self.w = init_weights(wb_shape)
            self.b = init_bias(wb_shape)

    # Dimensions:
    # 0: batch
    # 1: x value
    # 2: y value
    # 3: t value
    def call(self, inputs):
        # Fourier transform on inputs
        # Ideally rfft3d should be used, but tf hasn't got a gradient defined for that yet
        # Use fft3d instead for now and just ignore the imaginary part
        # Zero pad beginning & end of signal to account for FFT periodicity
        if self.padding > 0:
            x_data = tf.pad(inputs, [[0, 0],
                                     [self.padding, self.padding],
                                     [self.padding, self.padding],
                                     [self.padding, self.padding]])
        else:
            x_data = inputs
        x = tf.complex(x_data, x_data)
        x_ft = tf.math.real(tf.signal.fft3d(x))

        # Drop Fourier modes as a regularization measure
        if self.modes > 0:
            if self.fft_type == "2d":
                m_tl = x_ft[:, :self.modes, :self.modes]
                m_tr = x_ft[:, -self.modes:, :self.modes]
                m_bl = x_ft[:, :self.modes, -self.modes:]
                m_br = x_ft[:, -self.modes:, -self.modes:]
            else:
                m_tl = x_ft[:, :self.modes, :self.modes, :self.modes]
                m_tr = x_ft[:, -self.modes:, :self.modes, :self.modes]
                m_bl = x_ft[:, :self.modes, -self.modes:, :self.modes]
                m_br = x_ft[:, -self.modes:, -self.modes:, :self.modes]

            # Multiply Fourier modes with weight matrices and add biases
            xwb_tl = tf.add(tf.multiply(m_tl, self.w_tl), self.b_tl)
            xwb_tr = tf.add(tf.multiply(m_tr, self.w_tr), self.b_tr)
            xwb_bl = tf.add(tf.multiply(m_bl, self.w_bl), self.b_bl)
            xwb_br = tf.add(tf.multiply(m_br, self.w_br), self.b_br)

            # Recombine into one signal
            xwb_t = tf.concat([xwb_tl, xwb_tr], axis=1)
            xwb_b = tf.concat([xwb_bl, xwb_br], axis=1)
            xwb = tf.concat([xwb_t, xwb_b], axis=2)

            # Pad to match output shape
            xwb_shape = tf.shape(xwb)
            in_shape = tf.shape(inputs)
            paddings = [[0, 0],  # Batch
                        [0, in_shape[1] - xwb_shape[1]],  # X
                        [0, in_shape[2] - xwb_shape[2]]]  # Y
            if self.fft_type == "3d":
                paddings.append([0, self.fno_width - xwb_shape[3]])  # T
            xwb = tf.pad(xwb, paddings)

        # Otherwise handle the signal normally
        else:
            if self.padding > 0:
                x_ft = x_ft[:,
                       self.padding:-self.padding,
                       self.padding:-self.padding,
                       self.padding:-self.padding]
            xwb = tf.add(tf.multiply(x_ft, self.w), self.b)

        # Inverse FFT, take real part and return
        # See earlier comment on rfft3d to explain this weirdness
        if self.fft_type == "2d":
            x_r = tf.signal.ifft2d(tf.complex(xwb, xwb))
        else:
            x_r = tf.signal.ifft3d(tf.complex(xwb, xwb))
        output = tf.math.real(x_r)
        return output  # [:, pad_sz:-pad_sz, pad_sz:-pad_sz, pad_sz:-pad_sz]

    # BUG: saving .h5 model with SciPy optimizer breaks as it doesn't have a get_config function
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "fft_type": self.fft_type,
            "fno_width": self.fno_width,
            "t_lookback": self.t_lookback,
            "modes": self.modes
        })
        return config


# Append gamma map to solution
class ConcatAbsorption(tf.keras.layers.Layer):
    def __init__(self,
                 dtype,
                 **kwargs):
        self.dtype_ = dtype
        super(ConcatAbsorption, self).__init__()

    def call(self, inputs):
        # Readability
        solution = inputs[0]
        mesh = inputs[1]
        mesh_shape = tf.shape(mesh)
        n_x, n_y = mesh_shape[1], mesh_shape[2]

        # Create gamma and interior masks
        in_mask = tf.cast(tf.math.greater(mesh, 0), self.dtype_)
        gam_mask = tf.identity(mesh)
        gam_mask *= in_mask
        in_mask = tf.ones_like(in_mask) - in_mask

        # Create K map (interior neighbors)
        k_map = in_mask[:, 2:n_x, 1:n_y - 1] + \
                in_mask[:, 0:n_x - 2, 1:n_y - 1] + \
                in_mask[:, 1:n_x - 1, 2:n_y] + \
                in_mask[:, 1:n_x - 1, 0:n_y - 2]
        k_map = tf.pad(k_map, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        k_map = k_map * in_mask

        # Create gamma map (absorption)
        g_map = gam_mask[:, 2:n_x, 1:n_y - 1] + \
                gam_mask[:, 0:n_x - 2, 1:n_y - 1] + \
                gam_mask[:, 1:n_x - 1, 2:n_y] + \
                gam_mask[:, 1:n_x - 1, 0:n_y - 2]
        g_map = tf.pad(g_map, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        g_map = g_map / (tf.cast(tf.fill(tf.shape(g_map), 4), dtype=self.dtype_) - k_map)
        g_map = tf.where(tf.math.is_nan(g_map), tf.constant(-1, dtype=self.dtype_), g_map)

        # Concat gamma map with solution and return
        return tf.concat([solution, g_map], axis=-1)


# Transform network output
class ConstrainOutput(tf.keras.layers.Layer):
    def __init__(self,
                 dtype,
                 **kwargs):
        self.dtype_ = dtype
        super(ConstrainOutput, self).__init__()

    def call(self, inputs):
        solution = inputs[0]
        mesh = inputs[1]
        in_mask = tf.cast(tf.math.less(mesh, 0),
                          dtype=self.dtype_)
        return solution * in_mask
