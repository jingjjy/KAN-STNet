
import tensorflow.keras.models as KM
import tensorflow as tf
import inspect
import re
import numpy as np
from tensorflow.keras import layers
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
from utils import layers
import os
import datetime
from tensorflow.python.keras.utils import conv_utils
from math import sqrt

import uuid
from collections import Counter



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
    """Chebyshev KAN层实现"""

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
        theta = tf.acos(x_clipped)  # 形状: [batch_size, seq_len, in_features]


        theta_exp = tf.expand_dims(theta, axis=-1)  # [batch_size, seq_len, in_features, 1]
        n_theta = theta_exp * self.arange  # [batch_size, seq_len, in_features, degree+1]


        basis = tf.cos(n_theta)  # [batch_size, seq_len, in_features, degree+1]


        output = tf.einsum('bsid,iod->bso', basis, self.cheby_coeffs)
        return output




def spatial_cnn_mlp(inputs, cnn_filters, mlp_units, kernel_sizes=None, activation='relu',
                    name_prefix="spatial", kernel_initializer='he_normal',  # 改为 he_normal 更适配 ReLU
                    weight_decay=0.0):
    """
    尺度不敏感的空间特征提取 (CNN + MLP)
    核心改进：每一层后强制加 LayerNorm，并使用残差连接
    """
    if kernel_sizes is None:
        kernel_sizes = [5] * len(cnn_filters)


    x = KL.Lambda(lambda x: tf.expand_dims(x, axis=-1), name=f'{name_prefix}_expand')(inputs)


    cnn_outputs = []
    for i, (filters, k_size) in enumerate(zip(cnn_filters, kernel_sizes)):

        conv = KL.TimeDistributed(
            KL.Conv1D(filters=filters, kernel_size=k_size, padding='same', use_bias=False,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
            name=f'{name_prefix}/conv_{i}'
        )(x)

        ln = KL.TimeDistributed(
            KL.LayerNormalization(epsilon=1e-6),
            name=f'{name_prefix}/ln_conv_{i}'
        )(conv)

        act = KL.Activation(activation, name=f'{name_prefix}/act_conv_{i}')(ln)


        if i == 0:

            shortcut = KL.TimeDistributed(
                KL.Conv1D(filters=filters, kernel_size=1, padding='same', use_bias=False,
                          kernel_initializer=kernel_initializer),
                name=f'{name_prefix}/shortcut_{i}'
            )(x)
            x = KL.Add(name=f'{name_prefix}/add_{i}')([act, shortcut])
        else:

            if x.shape[-1] != filters:
                shortcut = KL.TimeDistributed(
                    KL.Conv1D(filters=filters, kernel_size=1, padding='same', use_bias=False,
                              kernel_initializer=kernel_initializer),
                    name=f'{name_prefix}/shortcut_{i}'
                )(x)
            else:
                shortcut = x
            x = KL.Add(name=f'{name_prefix}/add_{i}')([act, shortcut])

        cnn_outputs.append(x)


    x = KL.TimeDistributed(KL.Flatten(), name=f'{name_prefix}_flatten')(x)


    mlp_units = list(mlp_units)
    for i, units in enumerate(mlp_units):

        shortcut = x


        dense = KL.TimeDistributed(
            KL.Dense(units, use_bias=False,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
            name=f'{name_prefix}/dense_{i}'
        )(x)
        # LayerNorm
        ln = KL.TimeDistributed(
            KL.LayerNormalization(epsilon=1e-6),
            name=f'{name_prefix}/ln_dense_{i}'
        )(dense)


        if i == len(mlp_units) - 1:

            act = ln
        else:
            act = KL.Activation(activation, name=f'{name_prefix}/act_dense_{i}')(ln)


        if shortcut.shape[-1] == units and i < len(mlp_units) - 1:
            x = KL.Add(name=f'{name_prefix}/add_dense_{i}')([shortcut, act])
        else:
            x = act

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
    """
    使用FiLM（特征线性调制）方法融合空间和时间特征 ；spatial_features: 空间特征；temporal_features: 时间特征
    """

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




class DataGeneratorForLengthCmp(tf.keras.utils.Sequence):  # 用于长度比较的数据生成器，继承自Keras Sequence类
    def __init__(self, data_x, y, idxs, config, shuffle=True):
        self.data_x = data_x
        self.y = y
        self.idxs = idxs
        self.config = config
        self.shuffle = shuffle


    def __len__(self):
        return len(self.idxs) // self.config.BATCH_SIZE  # 返回一个epoch中的批次数

    def __getitem__(self, item):
        batch = self.get_batch(item)
        return [batch[0][0], batch[1][0], batch[2]], []  # 获取指定索引的批次数据

    def get_item(self):
        for i in range(len(self.idxs) // self.config.BATCH_SIZE):
            yield self.get_batch(i)

    def get_sample(self, t_idx):
        y_matrix = []


        t_idx = max(t_idx, 0)
        t_idx = min(t_idx, len(self.data_x) - self.config.TRAIN_LEN - self.config.EMBEDDING_LEN)

        for i in range(self.config.TRAIN_LEN):
            try:
                start_idx = t_idx + i
                end_idx = t_idx + i + self.config.EMBEDDING_LEN


                if end_idx > len(self.data_x):

                    slice_data = self.data_x[start_idx:, self.config.Y_IDX]

                    if len(slice_data) < self.config.EMBEDDING_LEN:
                        padding = np.full((self.config.EMBEDDING_LEN - len(slice_data),),
                                          self.data_x[-1, self.config.Y_IDX] if len(self.data_x) > 0 else 0.0)
                        slice_data = np.concatenate([slice_data, padding])
                else:
                    slice_data = self.data_x[start_idx:end_idx, self.config.Y_IDX]


                if len(slice_data) != self.config.EMBEDDING_LEN:
                    if len(slice_data) < self.config.EMBEDDING_LEN:
                        padding = np.full((self.config.EMBEDDING_LEN - len(slice_data),),
                                          slice_data[-1] if len(slice_data) > 0 else 0.0)
                        slice_data = np.concatenate([slice_data, padding])
                    else:
                        slice_data = slice_data[:self.config.EMBEDDING_LEN]

                y_matrix.append(slice_data)

            except Exception as e:
                print(f"Error in get_sample: {e}")

                default_slice = np.full((self.config.EMBEDDING_LEN,), 0.0)
                y_matrix.append(default_slice)

        y_matrix = np.stack(y_matrix)


        x_start = t_idx
        x_end = t_idx + self.config.TRAIN_LEN
        if x_end > len(self.data_x):
            x_data = self.data_x[x_start:]
            if len(x_data) < self.config.TRAIN_LEN:
                padding_len = self.config.TRAIN_LEN - len(x_data)
                padding = np.tile(self.data_x[-1:], (padding_len, 1))
                x_data = np.concatenate([x_data, padding], axis=0)
        else:
            x_data = self.data_x[x_start:x_end]

        if x_data.shape[0] != self.config.TRAIN_LEN:
            if x_data.shape[0] < self.config.TRAIN_LEN:
                padding_len = self.config.TRAIN_LEN - x_data.shape[0]
                padding = np.tile(x_data[-1:], (padding_len, 1))
                x_data = np.concatenate([x_data, padding], axis=0)
            else:
                x_data = x_data[:self.config.TRAIN_LEN]

        y_start = t_idx + self.config.TRAIN_LEN
        y_end = t_idx + self.config.TRAIN_LEN + self.config.EMBEDDING_LEN - 1

        if y_end > len(self.y):
            y_data = self.y[y_start:]
            if len(y_data) < (self.config.EMBEDDING_LEN - 1):
                padding = np.full((self.config.EMBEDDING_LEN - 1 - len(y_data),),
                                  self.y[-1] if len(y_data) > 0 else 0.0)
                y_data = np.concatenate([y_data, padding])
        else:
            y_data = self.y[y_start:y_end]

        if len(y_data) != (self.config.EMBEDDING_LEN - 1):
            if len(y_data) < (self.config.EMBEDDING_LEN - 1):
                padding = np.full((self.config.EMBEDDING_LEN - 1 - len(y_data),),
                                  y_data[-1] if len(y_data) > 0 else 0.0)
                y_data = np.concatenate([y_data, padding])
            else:
                y_data = y_data[:(self.config.EMBEDDING_LEN - 1)]

        return x_data, y_matrix, y_data

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


class DetailedMetricsCallback(tf.keras.callbacks.Callback):
    """自定义回调函数，计算详细指标"""

    def __init__(self, train_generator, val_generator, config):
        super(DetailedMetricsCallback, self).__init__()
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.config = config

    def on_epoch_end(self, epoch, logs=None):

        train_metrics = self._compute_detailed_metrics(self.train_generator, "train")


        if self.val_generator is not None:
            val_metrics = self._compute_detailed_metrics(self.val_generator, "val")
            metrics = {**train_metrics, **val_metrics}
        else:
            metrics = train_metrics


        if logs is not None:
            logs.update(metrics)


        self._print_detailed_metrics(epoch, metrics)

    def _compute_detailed_metrics(self, generator, prefix):

        all_true = []
        all_pred = []


        for i in range(len(generator)):
            batch = generator[i]

            inputs = batch[0]
            true_y = inputs[2]


            try:
                predictions = self.model.predict(inputs)

                if isinstance(predictions, list) and len(predictions) > 1:
                    pred_y = predictions[1]
                else:
                    pred_y = predictions

                pred_y = np.array(pred_y)
                true_y = np.array(true_y)

                if pred_y.ndim == 0:
                    print(f"警告: 批次 {i} 的预测结果是零维数组，跳过")
                    continue

                # 展平数组
                all_true.extend(true_y.flatten())
                all_pred.extend(pred_y.flatten())

            except Exception as e:
                print(f"预测批次 {i} 时出错: {e}")
                continue

        if len(all_true) == 0:
            print(f"警告: {prefix} 没有有效的预测数据")
            return {
                f'{prefix}_detailed_rmse': 0.0,
                f'{prefix}_detailed_pcc': 0.0,
                f'{prefix}_detailed_mae': 0.0
            }

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)


        rmse = np.sqrt(np.mean((all_true - all_pred) ** 2))
        mae = np.mean(np.abs(all_true - all_pred))

        if len(all_true) > 1:
            try:
                pcc = np.corrcoef(all_true, all_pred)[0, 1]
            except:
                pcc = 0.0
        else:
            pcc = 0.0

        return {
            f'{prefix}_detailed_rmse': rmse,
            f'{prefix}_detailed_pcc': pcc,
            f'{prefix}_detailed_mae': mae
        }

    def _print_detailed_metrics(self, epoch, metrics):
        print(f"\n=== Epoch {epoch + 1} 详细指标 ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

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

    def get_item(self):
        for i in range(len(self.idxs) // self.config.BATCH_SIZE):
            yield self.get_batch(i)

    def get_sample(self, t_idx):
        y_matrix = []
        # print('t_idx:{}'.format(t_idx))
        for i in range(self.config.TRAIN_LEN):
            try:
                start_idx = t_idx + i
                end_idx = t_idx + i + self.config.EMBEDDING_LEN

                if end_idx > len(self.data_x):

                    slice_data = self.data_x[start_idx:, self.config.Y_IDX]
                    if len(slice_data) < self.config.EMBEDDING_LEN:
                        padding = np.full((self.config.EMBEDDING_LEN - len(slice_data),),
                                          self.data_x[-1, self.config.Y_IDX])
                        slice_data = np.concatenate([slice_data, padding])
                else:
                    slice_data = self.data_x[start_idx:end_idx, self.config.Y_IDX]

                if len(slice_data) != self.config.EMBEDDING_LEN:
                    if len(slice_data) < self.config.EMBEDDING_LEN:
                        padding = np.full((self.config.EMBEDDING_LEN - len(slice_data),),
                                          slice_data[-1] if len(slice_data) > 0 else 0.0)
                        slice_data = np.concatenate([slice_data, padding])
                    else:
                        slice_data = slice_data[:self.config.EMBEDDING_LEN]

                y_matrix.append(slice_data)

            except Exception as e:
                print(f"Error in get_sample: {e}")
                print(f"Data shape: {self.data_x.shape}")
                print(f"Indices: start={t_idx + i}, end={t_idx + i + self.config.EMBEDDING_LEN}")
                default_slice = np.full((self.config.EMBEDDING_LEN,), 0.0)
                y_matrix.append(default_slice)

        y_matrix = np.stack(y_matrix)

        x = self.data_x[t_idx: t_idx + self.config.TRAIN_LEN]

        y_start = t_idx + self.config.TRAIN_LEN
        y_end = t_idx + self.config.TRAIN_LEN + self.config.EMBEDDING_LEN - 1
        if y_end > len(self.y):
            y_data = self.y[y_start:]
            if len(y_data) < (self.config.EMBEDDING_LEN - 1):
                padding = np.full((self.config.EMBEDDING_LEN - 1 - len(y_data),),
                                  self.y[-1] if len(y_data) > 0 else 0.0)
                y_data = np.concatenate([y_data, padding])
        else:
            y_data = self.y[y_start:y_end]

        return x, y_matrix, y_data

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
        with K.name_scope(self.name):
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
                        name=layer_name  # 显式指定名称
                    )
                    self._add_and_activate_layer(conv_layer)

                if self.norm_way and (i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_norm)):
                    with K.name_scope('norm_{}_{}'.format(self.norm_way,
                                                          i)):
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
        # 前向传播逻辑；如果使用残差连接，执行残差相加和激活


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

        # 创建掩码，排除指定变量
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
            spatial_features = KL.Lambda(lambda x: x, name='spatial_features_output')(spatial_features)
            print("spatial_features shape：", spatial_features.shape)
            print("spatial_features：", spatial_features)

            mkan_encoder = self._build_mkan_encoder()
            temporal_features = mkan_encoder(input_drop)
            temporal_features = KL.Lambda(lambda x: x, name='temporal_features_output')(temporal_features)
            print("M-KAN Output Shape:", temporal_features.shape)
            print("temporal_features：", temporal_features)

            merge_features = None

            if self.config.MERGE_FUNC == 'add':

                merge_features = KL.Add(name='merge_features')([spatial_features, temporal_features])
            elif self.config.MERGE_FUNC == 'concat':
                merge_features = KL.Concatenate(name='merge_features')([spatial_features, temporal_features])
            elif self.config.MERGE_FUNC == 'gated':
                merge_features = gated_fusion(spatial_features, temporal_features,
                                              name_prefix='merge_gated')
            elif self.config.MERGE_FUNC == 'film':
                merge_features = film_fusion(spatial_features, temporal_features,
                                             name_prefix='merge_film')
            else:
                merge_features = KL.Concatenate(name='merge_features')(
                    [spatial_features, temporal_features]
                )

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


            train_rmse = layers.RMSEMetric(self.config, name='train_rmse')([gt_y, predict_y])
            train_pcc = layers.PCCMetric(self.config, name='train_pcc')([gt_y, predict_y])
            train_mae = layers.MAEMetric(self.config, name='train_mae')([gt_y, predict_y])

            return KM.Model(
                inputs=[input, gt_y_matrix, gt_y],
                outputs=[delay_output, predict_y, known_y_loss, consistent_loss, predict_y_loss,
                         train_rmse, train_pcc, train_mae]
            )
        else:
            return KM.Model(
                inputs=input,
                outputs=[delay_output, predict_y, spatial_features, temporal_features]
            )

    def compile(self):
        print(self.model.outputs)
        loss_scale = 100.0
        optimizer = tf.keras.optimizers.Adam(lr=self.config.LR)


        losses = ['known_y_loss', 'consistent_loss']

        print(self.model.losses)

        for loss_name in losses:
            layer = self.model.get_layer(loss_name)
            #loss = layer.output * self.config.LOSS_WEIGHTS.get(loss_name, 1.)
            loss = layer.output * self.config.LOSS_WEIGHTS.get(loss_name, 1.) * loss_scale
            self.model.add_loss(loss)
        print(self.model.losses)

        self.model.compile(
            optimizer=optimizer,
            loss=[None for _ in self.model.outputs]
        )

        metrics_to_add = ['predict_y_loss', 'train_rmse', 'train_pcc', 'train_mae']

        for metric_name in metrics_to_add:
            layer = self.model.get_layer(metric_name)
            metric_value = layer.output
            self.model.add_metric(metric_value, name=metric_name, aggregation='mean')

            if metric_name.startswith('train_'):
                val_metric_name = 'val_' + metric_name
                self.model.add_metric(metric_value, name=val_metric_name, aggregation='mean')


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

            print(f'从检查点恢复: {checkpoint_prefix}')
            print(f'起始epoch: {self.epoch}')
        else:

            if checkpoint_prefix.endswith('.index'):

                checkpoint_prefix = checkpoint_prefix.replace('.index', '')
            self.model.load_weights(checkpoint_prefix, by_name=True)

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

        if val_generator is not None:
            filepath_format = os.path.join(
                self.log_dir,
                'weights_epoch_{epoch:04d}_loss_{loss:.3f}_val_loss_{val_loss:.3f}_'
                'train_rmse_{train_rmse:.3f}_val_rmse_{val_train_rmse:.3f}_'
                'train_pcc_{train_pcc:.3f}_val_pcc_{val_train_pcc:.3f}.h5'
            )
        else:
            filepath_format = os.path.join(
                self.log_dir,
                'weights_epoch_{epoch:04d}_loss_{loss:.3f}_'
                'train_rmse_{train_rmse:.3f}_train_pcc_{train_pcc:.3f}.h5'
            )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath_format,
                save_weights_only=True,
                save_format='h5',
                save_best_only=False,
                mode='auto'
            ),

            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: print(
                    f"Epoch {epoch + 1} 完成 - "
                    f"loss: {logs.get('loss', 0):.4f}, "
                    f"train_rmse: {logs.get('train_rmse', 0):.4f}, "
                    f"val_loss: {logs.get('val_loss', 0):.4f}"
                ) if (epoch + 1) % 10 == 0 else None
            )
        ]

        if hasattr(self.config, 'LR_SCHEDULER'):
            callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(self.config.LR_SCHEDULER)
            )

        fit_args = {
            'x': train_generator,
            'initial_epoch': self.epoch,
            'epochs': self.config.EPOCHS,
            'callbacks': callbacks,
            'max_queue_size': 10,
            'workers': 1,
            'use_multiprocessing': False,
            'shuffle': True,
            'verbose': 1  # 显示详细进度
        }


        if val_generator is not None:
            fit_args['validation_data'] = val_generator
            fit_args[
                'validation_steps'] = None if self.config.VALIDATION_STEPS is None else self.config.VALIDATION_STEPS

        print("开始训练...")
        history = self.model.fit(**fit_args)
        return history
