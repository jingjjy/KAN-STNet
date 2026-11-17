from forecast.lorenz_time_invariant_bzh import st_delay_model
from data import data_processing
from forecast.lorenz_time_invariant_bzh import lorenz_time_invariant_config
import tensorflow as tf
import os
import pickle

import numpy as np


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    config = lorenz_time_invariant_config.LorenzTimeInvariantConfig()
    config.K = 9  # jijia
    # load data
    time_invariant = True
    # data = data_processing.load_lorenz_data(time_invariant=time_invariant, n=3, time=3000)

    data = data_processing.load_lorenz_data(
        file_path='./logs/data/lorenz_d9_t150000.pkl'
    )

    total_length = len(data)  # 获取数据总长度
    print(f"total_length: {total_length}")
    #train_idxs, val_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(rate=1)
    train_idxs, val_idxs, test_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(
        total_length=total_length,
        train_len=config.TRAIN_LEN,
        embedding_len=config.EMBEDDING_LEN,
        rate=1,
        interval=60
    )

    noised_data, y = data_processing.add_noise(data, config)


    train_start = min(train_idxs)
    train_end = max(train_idxs) + config.TRAIN_LEN
    train_slice = noised_data[train_start:train_end]


    mean = np.mean(train_slice, axis=0)
    std = np.std(train_slice, axis=0)
    std[std == 0] = 1e-8  # 防止除零


    scaler_params = {'mean': mean, 'std': std}
    save_dir = os.path.join('logs', config.name)
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_params, f)

    noised_data = (noised_data - mean) / std
    y_mean = mean[config.Y_IDX]
    y_std = std[config.Y_IDX]
    y = (y - y_mean) / y_std
    scaler_params = {'mean': mean, 'std': std, 'y_mean': y_mean, 'y_std': y_std}
    # =================================


    target_mean = np.mean(noised_data[:, config.Y_IDX])
    target_std = np.std(noised_data[:, config.Y_IDX])

    train_generator = st_delay_model.DataGeneratorForLengthCmp(noised_data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(noised_data, y, val_idxs,
                                                                                           config)


    model = st_delay_model.STDelayModel(config, mode='training', log_dir_suffix=config.name)
    model.compile()
    model.train(train_generator, val_generator)
