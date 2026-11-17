from forecast import st_delay_model
from data import data_processing
from forecast.lorenz96_bzh import lorenz96_config
import tensorflow as tf
import os
import pickle
import numpy as np

if __name__ == '__main__':
    # 设置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    #gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    #for gpu in gpus:
    #    tf.config.experimental.set_memory_growth(gpu, True)

    # 加载配置
    config = lorenz96_config.Lorenz96Config()
    config.K = 60 #jijia

    # 加载数据
    data = data_processing.load_lorenz96_data(
        N=config.K,
        F=config.F,
        time_range=(0, 3000),
        dt=0.02
    )
    total_length = len(data)  # 获取数据总长度
    print(f"total_length: {total_length}")
    # 获取数据索引
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

    # 添加噪声
    noised_data, y = data_processing.add_noise(data, config)  # y 为一维标签序列


    # 使用训练集窗口计算均值和标准差（避免数据泄露）
    if train_idxs is not None and len(train_idxs) > 0:
        train_start = min(train_idxs)
        train_end = max(train_idxs) + config.TRAIN_LEN
        train_slice = noised_data[train_start:train_end]
    else:
        train_slice = noised_data

    mean = np.mean(train_slice, axis=0)
    std = np.std(train_slice, axis=0)
    std[std == 0] = 1e-8  # 防止除零

    # 保存标准化参数（供评估使用）
    scaler_params = {'mean': mean, 'std': std}
    save_dir = os.path.join('logs', config.name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_params, f)

    # 对全部数据应用标准化
    noised_data = (noised_data - mean) / std
    y_mean = mean[config.Y_IDX]
    y_std = std[config.Y_IDX]
    y = (y - y_mean) / y_std

    # 将 y 的统计信息也存入 scaler_params（方便逆变换）
    scaler_params['y_mean'] = y_mean
    scaler_params['y_std'] = y_std
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_params, f)

    print("\n=== 标准化验证 ===")
    print(f"数据形状: {noised_data.shape}")
    target_mean = np.mean(noised_data[:, config.Y_IDX])
    target_std = np.std(noised_data[:, config.Y_IDX])
    print(f"目标列 (Y_IDX={config.Y_IDX}) 均值: {target_mean:.6f}")
    print(f"目标列 (Y_IDX={config.Y_IDX}) 标准差: {target_std:.6f}")
    print("按 Enter 键继续训练...")
    # input()
    # =======================================================

    # 创建数据生成器（使用标准化后的数据）
    train_generator = st_delay_model.DataGeneratorForLengthCmp(
        noised_data, y, train_idxs, config
    )
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(
        noised_data, y, val_idxs, config
    )

    # 构建并训练模型
    model = st_delay_model.STDelayModel(
        config,
        mode='training',
        log_dir_suffix=config.name
    )
    model.compile()
    model.train(train_generator, val_generator)
"""
1.配置GPU内存使用
2.加载Lorenz96模型数据
3.划分训练、验证和测试集
4.为数据添加噪声
5.创建数据生成器
6.初始化、编译和训练时空延迟模型
"""