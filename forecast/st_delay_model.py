import tensorflow.keras.models as KM
import tensorflow as tf
import inspect
import numpy as np
from tensorflow.keras import layers
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
from utils import layers
import os
import datetime

class M_KAN_Block(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, seq_len, order, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.channel_mixer = ChebyKANLayer(
            in_channels, out_channels, min(order, 3),
            name=f"{self.name}/channel_mixer"
        )
        self.temporal_conv = tf.keras.layers.SeparableConv1D(
            filters=out_channels,
            kernel_size=kernel_size,
            padding='same',
            depth_multiplier=1,
            use_bias=False,
            name=f"{self.name}/temporal_conv"
        )

    def call(self, inputs):
        x1 = self.channel_mixer(inputs)
        x2 = self.temporal_conv(inputs)
        return x1 + x2


class ChebyKANLayer(tf.keras.layers.Layer):

    def __init__(self, in_features, out_features, degree, **kwargs):
        super().__init__(**kwargs)
        self.cheby_coeffs = self.add_weight(
            name='cheby_coeffs' if not kwargs.get('name') else f"{kwargs['name']}/cheby_coeffs",
            shape=(in_features, out_features, degree + 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.epsilon = 1e-7
        self.arange = tf.range(0, degree + 1, dtype=tf.float32)

    @tf.function
    def call(self, x):

        shape = tf.shape(x)
        batch_size = shape[0]
        seq_len = shape[1]

        x_norm = tf.tanh(x)
        x_norm = tf.tanh(x_norm)

        x_clipped = tf.clip_by_value(x_norm, -1 + self.epsilon, 1 - self.epsilon)
        theta = tf.acos(x_clipped)

        theta_exp = tf.expand_dims(theta, axis=-1)
        n_theta = theta_exp * self.arange

        basis = tf.cos(n_theta)
        output = tf.einsum('bsid,iod->bso', basis, self.cheby_coeffs)
        return output
def spatial_cnn_mlp(inputs, cnn_filters, mlp_units, kernel_sizes=None, activation='relu',
                    name_prefix="spatial", kernel_initializer='glorot_uniform',
                    weight_decay=0.0):
    if kernel_sizes is None:
        kernel_sizes = [5] * len(cnn_filters)
    x = KL.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)
    for i, (filters, k_size) in enumerate(zip(cnn_filters, kernel_sizes)):
        x = KL.TimeDistributed(
            KL.Conv1D(
                filters=filters,
                kernel_size=k_size,
                padding='same',
                activation=activation,
                name=f"{name_prefix}/Conv1D_{i}"
            )
        )(x)

    x = KL.TimeDistributed(KL.Flatten(), name=f"{name_prefix}_flatten")(x)

    nodes_count = len(mlp_units)

    for i in range(nodes_count - 1):
        x = KL.TimeDistributed(KL.Dense(mlp_units[i],
                                        activation=activation,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        name=name_prefix + '_Dense{}'.format(i + 1)))(x)
    x = KL.TimeDistributed(KL.Dense(mlp_units[-1],
                                    activation=None,
                                    name=name_prefix + '_Dense{}'.format(nodes_count)))(x)
    return x

def time_distributed_graph(input, nodes, last_activation, last_bn, activation, name_prefix,
                           kernel_initialzer, weight_decay, bn):
    nodes_count = len(nodes)

    x = input
    for i in range(nodes_count - 1):
        x = KL.TimeDistributed(KL.Dense(nodes[i],
                                        activation=activation,
                                        kernel_initializer=kernel_initialzer,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        name=name_prefix + '_Dense{}'.format(i + 1)))(x)
    activation = activation if last_activation else None
    x = KL.TimeDistributed(KL.Dense(nodes[-1],
                                    activation=activation,
                                    kernel_initializer=kernel_initialzer,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                    name=name_prefix + '_Dense{}'.format(nodes_count)))(x)

    return x

def film_fusion(spatial_features, temporal_features, name_prefix='film_fusion'):
    gamma = KL.Dense(spatial_features.shape[-1], name=f'{name_prefix}_gamma')(temporal_features)
    beta = KL.Dense(spatial_features.shape[-1], name=f'{name_prefix}_beta')(temporal_features)

    modulated = gamma * spatial_features + beta

    return KL.Add()([modulated, temporal_features])

def gated_fusion(spatial_features, temporal_features, hidden_dim=None, name_prefix='gated_fusion'):
    if hidden_dim is None:
        hidden_dim = spatial_features.shape[-1]
    gate = KL.Dense(hidden_dim, activation='sigmoid', name=f'{name_prefix}_gate')(
        KL.Concatenate()([spatial_features, temporal_features])
    )
    return gate * spatial_features + (1 - gate) * temporal_features

class DataGeneratorForLengthCmp(tf.keras.utils.Sequence):
    def __init__(self, data_x, y, idxs, config, shuffle=True):
        self.data_x = data_x
        self.y = y
        self.idxs = idxs
        self.config = config
        self.shuffle = shuffle

    def __len__(self):
        return len(self.idxs) // self.config.BATCH_SIZE
    def __getitem__(self, item):
        batch = self.get_batch(item)
        return [batch[0][0], batch[1][0], batch[2]], []

    def get_item(self):
        for i in range(len(self.idxs) // self.config.BATCH_SIZE):
            yield self.get_batch(i)

    def get_sample(self, t_idx):
        y_matrix = []
        for i in range(self.config.TRAIN_LEN):
            try:
                y_matrix.append(self.data_x[
                                t_idx - self.config.TRAIN_LEN + i:
                                t_idx - self.config.TRAIN_LEN + i + self.config.EMBEDDING_LEN, self.config.Y_IDX])
            except Exception:
                print(self.data_x.shape)
                print(t_idx - self.config.TRAIN_LEN + i, t_idx - self.config.TRAIN_LEN + i + self.config.EMBEDDING_LEN,
                      self.config.Y_IDX)
        y_matrix = np.stack(y_matrix)

        x = self.data_x[t_idx - self.config.TRAIN_LEN: t_idx]
        return x, y_matrix, self.y[t_idx: t_idx + self.config.EMBEDDING_LEN - 1]

    def get_batch(self, b_idx):
        xs = []
        y_matrixs = []
        ys = []
        for i in range(self.config.BATCH_SIZE):
            t_idx = self.idxs[b_idx * self.config.BATCH_SIZE + i]
            x, y_matrix, y = self.get_sample(t_idx)
            xs.append(x)
            y_matrixs.append(y_matrix)
            ys.append(y)
        xs = np.stack(xs)
        y_matrixs = np.stack(y_matrixs)
        ys = np.stack(ys)

        return [xs], [y_matrixs], ys

    def get_long_term_item(self, iterations_count):
        for i in range(len(self.idxs)):
            yield self.get_long_term_batch(i, iterations_count)

    def get_long_term_batch(self, b_idx, iterations_count):
        xs = []
        y_matrixs = []
        ys = []
        for i in range(iterations_count):
            t_idx = self.idxs[b_idx] + i * (self.config.EMBEDDING_LEN - 1)
            x, y_matrix, y = self.get_sample(t_idx)
            xs.append(x)
            y_matrixs.append(y_matrix)
            ys.append(y)
        xs = np.stack(xs)
        y_matrixs = np.stack(y_matrixs)
        ys = np.stack(ys)

        return [xs], [y_matrixs], ys

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_x, y, idxs, config, shuffle=True):
        self.data_x = data_x
        self.y = y
        self.idxs = idxs
        self.config = config
        self.shuffle = shuffle

    def __len__(self):
        return len(self.idxs) // self.config.BATCH_SIZE

    def __getitem__(self, item):
        batch = self.get_batch(item)

        return [batch[0][0], batch[1][0], batch[2]], []

    def get_item(self):
        for i in range(len(self.idxs) // self.config.BATCH_SIZE):
            yield self.get_batch(i)

    def get_sample(self, t_idx):
        y_matrix = []
        for i in range(self.config.TRAIN_LEN):
            try:
                y_matrix.append(self.data_x[t_idx + i: t_idx + i + self.config.EMBEDDING_LEN, self.config.Y_IDX])
            except BaseException:
                print(self.data_x.shape)
                print(t_idx + i, t_idx + i + self.config.EMBEDDING_LEN, self.config.Y_IDX)
        y_matrix = np.stack(y_matrix)

        x = self.data_x[t_idx: t_idx + self.config.TRAIN_LEN]
        return x, y_matrix, self.y[t_idx + self.config.TRAIN_LEN:
                                   t_idx + self.config.TRAIN_LEN +
                                   self.config.EMBEDDING_LEN - 1]

    def get_batch(self, b_idx):
        xs = []
        y_matrixs = []
        ys = []
        for i in range(self.config.BATCH_SIZE):
            t_idx = self.idxs[b_idx * self.config.BATCH_SIZE + i]
            x, y_matrix, y = self.get_sample(t_idx)
            xs.append(x)
            y_matrixs.append(y_matrix)
            ys.append(y)
        xs = np.stack(xs)
        y_matrixs = np.stack(y_matrixs)
        ys = np.stack(ys)

        return [xs], [y_matrixs], ys

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
class TemporalConvBlock(KL.Layer):
    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 repeat_times: int,
                 residual: bool = False,
                 last_norm: bool = True,
                 last_acti: bool = True,
                 last_drop: bool = True,
                 activation: str = 'relu',
                 norm_way: str = None,
                 kernel_initializer: str = 'he_normal',
                 kernel_regularizer: dict = 'l2',
                 weight_decay=0,
                 dropout_rate: float = 0,
                 **kwargs):

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.repeat_times = repeat_times
        self.residual = residual
        self.last_norm = last_norm
        self.last_acti = last_acti
        self.last_drop = last_drop
        self.activation = activation
        self.norm_way = norm_way
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        self.build_output_shape = None
        self.layers = []
        self.shape_match_conv = None
        self.final_activation = None

        if self.residual:
            assert self.last_drop, 'residual block must has last drop'
            assert self.last_acti, 'residual block must has last acti'
            assert self.last_norm, 'residual block must has last norm'

        super(TemporalConvBlock, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer

        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.

        """
        self.layers.append(layer)
        self.layers[-1].build(self.build_output_shape)
        self.build_output_shape = self.layers[-1].compute_output_shape(self.build_output_shape)

    def build(self, input_shape):
        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.build_output_shape = input_shape
            for i in range(self.repeat_times):
                layer_name = f'conv1d_{i}' if not self.residual else f'res_conv1d_{i}'
                with K.name_scope(layer_name):
                    conv_layer = tf.keras.layers.Conv1D(
                        self.nb_filters,
                        self.kernel_size,
                        padding='causal',
                        dilation_rate=self.dilation_rate,
                        kernel_initializer=tf.keras.initializers.get(self.kernel_initializer),
                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                        name=layer_name
                    )
                    self._add_and_activate_layer(conv_layer)
                # normalization layer
                if self.norm_way and (i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_norm)):
                    with K.name_scope('norm_{}_{}'.format(self.norm_way,
                                                          i)):  # name scope used to make sure weights get unique names
                        if self.norm_way == 'bn':
                            self._add_and_activate_layer(tf.keras.layers.BatchNormalization())
                        elif self.norm_way == 'ln':
                            self._add_and_activate_layer(tf.keras.layers.LayerNormalization())
                        else:
                            raise NotImplementedError()
                if i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_acti):
                    self._add_and_activate_layer(tf.keras.layers.Activation(self.activation))
                else:
                    self._add_and_activate_layer(tf.keras.layers.Activation('linear'))

                if i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_drop):
                    self._add_and_activate_layer(tf.keras.layers.SpatialDropout1D(rate=self.dropout_rate))

            if self.residual:
                if self.nb_filters != input_shape[-1]:
                    # 1x1 conv to match the shapes (channel dimension).
                    name = 'matching_conv1D'
                    with K.name_scope(name):
                        # make and build this layer separately because it directly uses input_shape
                        self.shape_match_conv = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                                       kernel_size=1,
                                                                       padding='same',
                                                                       name=name,
                                                                       kernel_initializer=self.kernel_initializer)

                else:
                    name = 'matching_identity'
                    self.shape_match_conv = tf.keras.layers.Lambda(lambda x: x, name=name)

                with K.name_scope(name):
                    # self.shape_match_conv.build(self.build_output_shape)
                    self.shape_match_conv.build(input_shape)
                    # self.build_output_shape = self.shape_match_conv.compute_output_shape(self.build_output_shape)
                    self.build_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

                self.final_activation = tf.keras.layers.Activation(self.activation)
                self.final_activation.build(self.build_output_shape)  # probably isn't necessary
                # this is done to force Keras to add the layers in the list to self._layers
                self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
                self.__setattr__(self.final_activation.name, self.final_activation)

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)

        super(TemporalConvBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=True, **kwargs):
        x = inputs
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)

        if self.residual:
            x2 = self.shape_match_conv(inputs)
            res_x = tf.keras.layers.add([x, x2])
            res_act_x = self.final_activation(res_x)
            return res_act_x
        return x

class CausalInferenceLayer(KL.Layer):
    def __init__(self, config, **kwargs):
        super(CausalInferenceLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):

        full_pred, ablated_preds = inputs
        full_rmse = tf.sqrt(tf.reduce_mean(tf.square(full_pred), axis=-1))

        causality_scores = []
        for i in range(self.config.K):
            ablated_rmse = tf.sqrt(tf.reduce_mean(tf.square(ablated_preds[i]), axis=-1))
            causality_score = ablated_rmse - full_rmse
            causality_scores.append(causality_score)

        causality_matrix = tf.stack(causality_scores, axis=1)

        return causality_matrix


class VariableAblationLayer(KL.Layer):

    def __init__(self, variable_idx, config, **kwargs):
        super(VariableAblationLayer, self).__init__(**kwargs)
        self.variable_idx = variable_idx
        self.config = config

    def call(self, inputs):
        mask = tf.ones_like(inputs)
        mask = tf.tensor_scatter_nd_update(
            mask,
            [[i, j, self.variable_idx] for i in range(tf.shape(inputs)[0])
             for j in range(tf.shape(inputs)[1])],
            tf.zeros([tf.shape(inputs)[0] * tf.shape(inputs)[1]])
        )

        ablated_input = inputs * mask
        return ablated_input

class STDelayModel(object):
    def __init__(self, config, mode='training', log_dir_suffix=None):
        self.config = config
        self.mode = mode
        self.model = self.build_model()
        self.model.summary()
        self.epoch = 0
        self.log_dir_suffix = log_dir_suffix
        self.log_dir = self.get_log_dir()

    def _build_spatial_encoder(self, input_tensor):

        input_T = tf.transpose(input_tensor, perm=[0, 2, 1])  # [batch, K, TRAIN_LEN]

        x = input_T
        for _ in range(self.config.ENCODING_LAYER_NUMS):
            x = layers.encoder_graph(
                self.config.SPARTICAL_DIM,
                self.config.NUM_HEADS,
                self.config.DIFF,
                self.config.TRAINING,
                rate=self.config.DROP_RATE,
                x=x
            )

        x = tf.transpose(x, perm=[0, 2, 1])  # [batch, TRAIN_LEN, SPATIAL_DIM]

        return x

    def _build_mkan_encoder(self):
        encoder_layers = []
        in_channels = self.config.K

        for i, out_channels in enumerate(self.config.ENCODER_NODES):
            order = self.config.BASE_ORDER + i

            encoder_layers.append(
                M_KAN_Block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    seq_len=self.config.TRAIN_LEN,
                    order=order,
                    kernel_size=3,
                    name=f'temporal_conv_block_{i}'
                )
            )
            in_channels = out_channels

        return tf.keras.Sequential(encoder_layers, name='tcn_encoder')


    def build_model(self):
        t_layer = KL.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]), name='t_layer')
        input = KL.Input(shape=(self.config.TRAIN_LEN, self.config.K), name='input', dtype=tf.float32)
        print(input.shape)
        if self.mode == 'training':
            gt_y_matrix = KL.Input(shape=(self.config.TRAIN_LEN, self.config.EMBEDDING_LEN), name='gt_y_matrix',
                                   dtype=tf.float32)
            gt_y = KL.Input(shape=(self.config.EMBEDDING_LEN - 1,), name='gt_y', dtype=tf.float32)


        input_drop = KL.SpatialDropout1D(rate=self.config.DROP_RATE, name='input_drop')(input)
        # input_T = t_layer(input_dro p)
        print("input_drop shape：", input_drop.shape)

        if not self.config.MERGE_ONLY:

            if hasattr(self.config, 'SPATIAL_KERNEL_SIZES'):
                kernel_sizes = self.config.SPATIAL_KERNEL_SIZES
            else:
                kernel_sizes = None
            cnn_filters = self.config.CNN_FILTERS if hasattr(self.config, 'CNN_FILTERS') else [64, 32, 16]
            mlp_units = self.config.SPATIAL_NODES
            spatial_features = spatial_cnn_mlp(
                inputs=input_drop,
                cnn_filters=cnn_filters,
                kernel_sizes=kernel_sizes,
                mlp_units=mlp_units,
                activation=self.config.ACITVATION,
                name_prefix="spatial",
                kernel_initializer=self.config.KERNEL_INITIALIZER,
                weight_decay=self.config.WEIGHT_DECAY
            )
            print("spatial_features shape：", spatial_features.shape)

            mkan_encoder = self._build_mkan_encoder()
            temporal_features = mkan_encoder(input_drop)
            print("M-KAN Output Shape:", temporal_features.shape)

            merge_features = None
            if self.config.MERGE_FUNC == 'add':

                merge_features = KL.Add(name='merge_features')([spatial_features, temporal_features])
            elif self.config.MERGE_FUNC == 'concat':
                # merge_features = KL.Concatenate(name='merge_features')([spatial_features, temporal_features_t])
                merge_features = KL.Concatenate(name='merge_features')([spatial_features, temporal_features])
            elif self.config.MERGE_FUNC == 'gated':
                merge_features = gated_fusion(spatial_features, temporal_features,
                                              name_prefix='merge_gated')
            elif self.config.MERGE_FUNC == 'film':
                merge_features = film_fusion(spatial_features, temporal_features,
                                             name_prefix='merge_film')
            else:
                print('unknown func')
                exit(0)

        else:
            merge_features = input_drop

        print(merge_features.shape)

        delay_output = time_distributed_graph(merge_features, self.config.MERGE_MAP_NODES, last_activation=False,
                                              last_bn=False, activation=self.config.ACITVATION,
                                              name_prefix='merge_delay',
                                              kernel_initialzer=self.config.KERNEL_INITIALIZER,
                                              weight_decay=self.config.WEIGHT_DECAY,
                                              bn=self.config.BN)

        predict_y = layers.Matrix2Y(self.config, name='predict_y')(delay_output)

        if self.mode == 'training':
            known_y_loss = layers.KnownYLoss(self.config, name='known_y_loss')([gt_y_matrix, delay_output])
            consistent_loss = layers.TimeConsistentLoss(self.config, name='consistent_loss')(delay_output)
            predict_y_loss = layers.PredictYLoss(name='predict_y_loss')([gt_y, predict_y])

            return KM.Model(
                inputs=[input, gt_y_matrix, gt_y],
                outputs=[delay_output, predict_y, known_y_loss, consistent_loss, predict_y_loss]
            )
        else:
            return KM.Model(inputs=input, outputs=[delay_output, predict_y])

    def compile(self):
        print(self.model.outputs)
        optimizer = tf.keras.optimizers.Adam(lr=self.config.LR)

        # losses = ['known_y_loss', 'consistent_loss', 'contrastive_loss']
        losses = ['known_y_loss', 'consistent_loss']

        print(self.model.losses)
        # add loss
        for loss_name in losses:
            layer = self.model.get_layer(loss_name)
            # if layer.output not in self.model.losses
            loss = layer.output * self.config.LOSS_WEIGHTS.get(loss_name, 1.)
            self.model.add_loss(loss)
        print(self.model.losses)

        self.model.compile(optimizer=optimizer, loss=[None for _ in self.model.outputs])

        losses.extend(['predict_y_loss'])
        # add loss metric
        for loss_name in losses:
            layer = self.model.get_layer(loss_name)
            loss = layer.output * self.config.LOSS_WEIGHTS.get(loss_name, 1.)
            self.model.add_metric(loss, name=loss_name, aggregation='mean')

        print(self.model.metrics_names)

    def load_weights(self, checkpoint_prefix):

        if self.mode == 'training':

            checkpoint = tf.train.Checkpoint(
                optimizer=self.model.optimizer,
                model=self.model,
                epoch=tf.Variable(0, dtype=tf.int64)
            )

            checkpoint_status = checkpoint.restore(checkpoint_prefix)
            checkpoint_status.expect_partial()

            if hasattr(checkpoint, 'epoch'):
                self.epoch = checkpoint.epoch.numpy()

            self.log_dir = os.path.dirname(os.path.dirname(checkpoint_prefix))

        else:
            if checkpoint_prefix.endswith('.index'):
                checkpoint_prefix = checkpoint_prefix.replace('.index', '')
            self.model.load_weights(checkpoint_prefix)
    def get_log_dir(self):
        time_stamp = datetime.datetime.now()
        suffix = '({})'.format(self.log_dir_suffix) if self.log_dir_suffix is not None else ''
        log_dir = os.path.join('logs', time_stamp.strftime('%Y_%m_%d-%H_%M_%S') + suffix)
        return log_dir
    def train(self, train_generator, val_generator):
        os.makedirs(self.log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        if val_generator is not None:
            filepath_format = os.path.join(
                self.log_dir,
                'weights_epoch_{epoch:04d}_loss_{loss:.3f}_val_loss_{val_loss:.3f}_predict_loss_{predict_y_loss:.3f}.h5'
            )
        else:
            filepath_format = os.path.join(
                self.log_dir,
                'weights_epoch_{epoch:04d}_loss_{loss:.3f}_predict_loss_{predict_y_loss:.3f}.h5'
            )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath_format,
                save_weights_only=True,
                save_format='h5',
                save_best_only=False,
                mode='auto'
            )
        ]

        fit_args = {
            'x': train_generator,
            'initial_epoch': self.epoch,
            'epochs': self.config.EPOCHS,
            'callbacks': callbacks,
            'max_queue_size': 10,
            'workers': 1,
            'use_multiprocessing': False,
            'shuffle': True
        }

        if val_generator is not None:
            fit_args['validation_data'] = val_generator
            fit_args[
                'validation_steps'] = None if self.config.VALIDATION_STEPS is None else self.config.VALIDATION_STEPS

        self.model.fit(**fit_args)

