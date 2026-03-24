from forecast import st_delay_model
from data import data_processing
from forecast.lorenz_time_invariant import lorenz_time_invariant_config
import tensorflow as tf

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    config = lorenz_time_invariant_config.LorenzTimeInvariantConfig()
    config.K = 9
    # load data
    time_invariant = True
    data = data_processing.load_lorenz_data(time_invariant=time_invariant, n=3, time=3000)

    total_length = len(data)
    print(f"total_length: {total_length}")
    train_idxs, val_idxs, test_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(
        total_length=total_length,
        train_len=config.TRAIN_LEN,
        embedding_len=config.EMBEDDING_LEN,
        rate=1,
        interval=60
    )
    print(f"训练集大小: {len(train_idxs)}")
    print(f"验证集大小: {len(val_idxs)}")
    print(f"测试集大小: {len(test_idxs)}")

    data, y = data_processing.add_noise(data, config)
    train_generator = st_delay_model.DataGeneratorForLengthCmp(data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(data, y, val_idxs, config)

    model = st_delay_model.STDelayModel(config, mode='training',
                                        log_dir_suffix=config.name)
    model.compile()

    model.train(train_generator, val_generator)
