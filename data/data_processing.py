import pickle  # 序列化/反序列化数据
import numpy as np
import h5py  # 处理HDF5格式文件
import scipy.io as sio  # 读取MATLAB的.mat文件
import pandas as pd
import os  # 操作系统路径管理
import data.lorenz96_try as lorenz96
import data.lorenz as lorenz

# 1.数据目录配置
# # TODO: you should change the base path of data dir
DATA_BASE_DIR = r'./logs/data'



# 添加噪声函数
def add_noise(raw_data, config):
    """
    add white noise to the data matrix
    :param raw_data: data matrix with size [m, K]
    :param config:
    :return: the data matrix with whtie noise noised_data:[m ,K], the original y raw_y:[m,]
    带噪声的数据 + 原始目标变量 y（若启用噪声，则 y 也会被污染）。
    """
    raw_y = np.copy(raw_data[:, config.Y_IDX])
    if config.ADD_NOISE:
        np.random.seed(234)  # with same white noise
        noise = np.random.normal(size=raw_data.shape)
        noise_data = raw_data + noise * config.DATA_NOISE_STRENGTH  # change the stength of noise
        raw_y = np.copy(noise_data[:, config.Y_IDX])
        return noise_data, raw_y
    else:
        return raw_data, raw_y


def add_more_augmentation(data, config):
    
    jitter_strength = 0.02
    jitter = np.random.normal(0, jitter_strength, data.shape)
    data_jittered = data + jitter


    scaling_factor = np.random.uniform(0.95, 1.05)
    data_scaled = data * scaling_factor

    return data_jittered, data_scaled


# 3.数据索引生成函数
# (1) 通用非重叠索引
def get_data_idxs_as_predict_idx_no_overlap(total_length, train_len, embedding_len, rate=1, interval=100):
    """
    返回按顺序排列的数据索引，确保所有数据没有重叠
    且索引在有效范围内

    :param total_length: 数据总长度
    :param train_len: 训练序列长度
    :param embedding_len: 预测序列长度
    :param rate: 训练数据比例
    :param interval: 相邻样本间的时间点间隔
    :return: 训练集索引、验证集索引、测试集索引
    """

    min_start = train_len
    max_start = total_length - embedding_len


    idxs = []
    start = min_start
    while start <= max_start:
        idxs.append(start)
        start += interval


    idxs = np.array(idxs)


    total_samples = len(idxs)
    train_count = int(total_samples * 0.8)
    val_count = int(total_samples * 0.1)
    test_count = total_samples - train_count - val_count  #
    train_idxs = idxs[:train_count]
    val_idxs = idxs[train_count:train_count + val_count]
    test_idxs = idxs[train_count + val_count:]


    count = int(len(train_idxs) * rate)
    train_idxs = train_idxs[:count]


    return train_idxs, val_idxs, test_idxs


# Dataset-Specific Indexing
def get_data_idxs_for_lorenz(threshold=5.0, time_invariant=True, rate=1):
    np.random.seed(123)
    if not time_invariant:
        idxs = np.arange(2900)
        np.random.shuffle(idxs)
        train_idxs = idxs[:1000]
        val_idxs = idxs[1000:1050]
    else:
        idxs = np.arange(5000)
        np.random.shuffle(idxs)
        train_idxs = idxs[:1000]
        val_idxs = idxs[1000:1050]

    total_len = train_idxs.shape[0]
    print(total_len)

    count = int(total_len * rate)
    train_idxs = train_idxs[:count]
    return train_idxs, val_idxs




def get_data_idxs_for_normal(data, train_rate, config, select_num=None, shuffle=True):
    """
    get the training idxs and validation idxs for normal data without specific functions
    :param data:
    :param train_rate:
    :param config:
    :param select_num: the size of data will be selected after shuffle, if None will select all data
    :param shuffle: whether to shuffle
    :return:
    """
    total_num = data.shape[0] - (config.TRAIN_LEN + config.EMBEDDING_LEN - 1) + 1
    idxs = np.arange(total_num)

    if shuffle:
        np.random.seed(123)
        np.random.shuffle(idxs)

    if select_num is not None:
        assert select_num > total_num, 'select num must smaller than total'
        idxs = idxs[:select_num]
        total_num = select_num

    if train_rate != 1:
        train_idxs = idxs[:int(train_rate * total_num)]
        val_idxs = idxs[int(train_rate * total_num): total_num]

    else:
        train_idxs = idxs
        val_idxs = None

    # print(train_idxs)
    return train_idxs, val_idxs



def load_lorenz96_data(N, F, time_range=(0, 20), dt=0.02, zscore=False):
    """
    load lorenz96_bzh data
    :return: data [time_len, dim]
    """
    data = None
    time = time_range[1] - time_range[0]
    file_name = 'lorenz/lorenz96_F{}_d{}_t{}.pkl'.format(F, N, int(time / 0.02))
    if not os.path.exists(os.path.join(DATA_BASE_DIR, file_name)):
        print(file_name)
        print('generating lorenz96_bzh data...')
        data = lorenz96.gen_L96_data(N, F, time_range, dt)
        with open(os.path.join(DATA_BASE_DIR, file_name), 'wb') as file:
            pickle.dump(data, file)
    else:
        print('loading lorenz  data...')
        with open(os.path.join(DATA_BASE_DIR, file_name), 'rb') as file:
            data = pickle.load(file)
        # print(data.shape)
    data = data.T[10000:]
    if zscore:
        data = z_score(data)
    return data


def load_ks_data():
    """
    load KS data
    :return: data [time_len, dim]
    """
    data = np.genfromtxt(os.path.join(DATA_BASE_DIR, 'ks/KSequ_80000.csv'), delimiter=',')
    data = data.T[:, :512]
    return data




def load_lorenz_data(time_invariant=True, n=3, time=100, file_path=None):
    """
    Load Lorenz data from a given pickle file or generate it.
    :param time_invariant: unused when file_path provided
    :param n: unused when file_path provided
    :param time: unused when file_path provided
    :param file_path: path to the .pkl file (if provided, load from it)
    :return: data array [time_len, dim]
    """
    if file_path is not None and os.path.exists(file_path):
        print(f"Loading Lorenz data from {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # ensure shape [time_len, dim]
        if data.ndim == 2 and data.shape[0] < data.shape[1]:
            data = data.T
        return data

    # 若未指定文件或文件不存在，回退到原有的生成逻辑
    data = None
    file_name = 'lorenz{}_d{}_t{}.pkl'.format('' if time_invariant else '_time_variant', n * 3, int(time / 0.02))
    if not os.path.exists(os.path.join(DATA_BASE_DIR, file_name)):
        print(file_name)
        print('generating lorenz time {} data...'.format('invariant' if time_invariant else 'variant'))
        data = lorenz.my_lorenz(n, time=time, time_invariant=time_invariant)
        with open(os.path.join(DATA_BASE_DIR, file_name), 'wb') as file:
            pickle.dump(data, file)
    else:
        print('loading lorenz time {} data...'.format('invariant' if time_invariant else 'variant'))
        with open(os.path.join(DATA_BASE_DIR, file_name), 'rb') as file:
            data = pickle.load(file)
    data = data.T[2000:]
    return data




# 5.工具函数
def window(data, size, stride):  # 滑动窗口；对时间序列数据做滑动平均
    """
    apply a window to the raw data
    :param data: [t, dim]
    :param size:
    :param stride:
    :return:
    """

    t_len = data.shape[0]  # length of raw data
    win_len = (t_len - size) // stride + 1  # length of data after applying a window

    print(t_len)
    print(win_len)

    win_data = []
    for i in range(win_len):
        win_data.append(np.mean(data[i * stride:i * stride + size], axis=0))
    win_data = np.stack(win_data)
    print(win_data.shape)
    return win_data


def load_solar_irradiance(win_size=6, stride=1):
    """
    load data for solar irradiance dataset
    """

    data = sio.loadmat(os.path.join(DATA_BASE_DIR, 'solar_irradiance/201606241058solar_irradiance.mat'))
    data = data['solar_irradiance']

    data = window(data, size=win_size, stride=stride)

    print(data.shape)
    data[data < 0.0] = 0.0  # set all values smaller than 0 as 0.
    return data


def z_score(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)  # 标准化
