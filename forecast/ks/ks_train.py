from forecast import st_delay_model
from data import data_processing
from forecast.ks_bzh import ks_config
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

    config = ks_config.KSConfig()
    config.K = 512

    data = data_processing.load_ks_data()

    total_length = len(data)
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
    std[std == 0] = 1e-8

    scaler_params = {'mean': mean, 'std': std}
    save_dir = os.path.join('logs', config.name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_params, f)

    noised_data = (noised_data - mean) / std
    y = (y - mean[config.Y_IDX]) / std[config.Y_IDX]


    train_generator = st_delay_model.DataGeneratorForLengthCmp(noised_data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(noised_data, y, val_idxs, config)

    model = st_delay_model.STDelayModel(config, mode='training', log_dir_suffix=config.name)
    model.compile()
    model.train(train_generator, val_generator)