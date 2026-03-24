import pickle
import numpy as np
import scipy.io as sio
import pandas as pd
import os
import data.lorenz96 as lorenz96
import data.lorenz as lorenz

#  TODO: you should change the base path of data dir

DATA_BASE_DIR = r'/logs/data'
def add_noise(raw_data, config):
    """
    add white noise to the data matrix
    :param raw_data: data matrix with size [m, K]
    :param config:
    :return: the data matrix with whtie noise noised_data:[m ,K], the original y raw_y:[m,]

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

def get_data_idxs_as_predict_idx_no_overlap(total_length, train_len, embedding_len, rate=1, interval=100):

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
    test_count = total_samples - train_count - val_count

    train_idxs = idxs[:train_count]
    val_idxs = idxs[train_count:train_count + val_count]
    test_idxs = idxs[train_count + val_count:]

    count = int(len(train_idxs) * rate)
    train_idxs = train_idxs[:count]

    return train_idxs, val_idxs, test_idxs

#Dataset-Specific Indexing
def get_data_idxs_for_lorenz(threshold=5.0, time_invariant=True, rate=1):
    np.random.seed(123)
    if not time_invariant:
        idxs = np.arange(2900)
        train_idxs = idxs[:1000]
        val_idxs = idxs[1000:1050]
    else:
        idxs = np.arange(5000)
        train_idxs = idxs[:1000]
        val_idxs = idxs[1000:1050]

    total_len = train_idxs.shape[0]
    print(total_len)

    count = int(total_len * rate)
    train_idxs = train_idxs[:count]
    return train_idxs, val_idxs

def get_data_idxs_for_gene(data, config):
    """
    return the data idxs of gene data
    :param data: raw data
    :param config:
    :return:
    """
    total_num = data.shape[0] - (config.TRAIN_LEN + config.EMBEDDING_LEN - 1) + 1
    return np.arange(total_num), None

def get_data_idxs_for_normal(data, train_rate, config, select_num=None, shuffle=True):

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

def get_data_idxs_for_traffic(data, train_rate, config, select_num=None, y_idxs=None):
    """
    first will drop the data with traffic speed of target sensor is 0.
    get the training idxs and validation idxs for traffic data
    :param data:
    :param train_rate:
    :param config:
    :param select_num:
    :param y_idxs: if is a list, drop the union set of data which traffic speed of target sensors (y_idxs) is 0.
    :return:
    """
    sample_len = config.TRAIN_LEN + config.EMBEDDING_LEN - 1  
    total_num = data.shape[0]  

    # the valid idxs of data that target variable that without 0.
    valid_idxs = np.ones(shape=(total_num,), dtype=np.float32)  

    if y_idxs is not None:  
        for y_idx in y_idxs:  # calculate the union set
            target_vars = data[:, y_idx]  

            for i, var in enumerate(target_vars):  
                if var == 0.0:
                    start = np.maximum(0, i - (sample_len - 1))
                    end = i
                    valid_idxs[start:end] = 0.0
    else:
        target_vars = data[:, config.Y_IDX]  

        for i, var in enumerate(target_vars):  
            if var == 0.0:
                start = np.maximum(0, i - (sample_len - 1))
                end = i
                valid_idxs[start:end] = 0.0

    total_num = total_num - sample_len + 1
    valid_idxs = valid_idxs[:total_num]

    valid_idxs = np.where(valid_idxs)[0]  
    total_num = valid_idxs.shape[0]
    print(total_num)

    # used the first 10000 samples
    # valid_idxs = valid_idxs[:15000]
    # total_num = 15000
    valid_idxs = valid_idxs[:10000]
    total_num = 10000

    np.random.seed(123)
    np.random.shuffle(valid_idxs)

    if select_num is not None:
        assert select_num < total_num, 'select num must smaller than total'
        valid_idxs = valid_idxs[:select_num]
        total_num = select_num

    train_idxs = valid_idxs[:int(train_rate * total_num)]
    val_idxs = valid_idxs[int(train_rate * total_num):int(train_rate * total_num)+300]
    # test_idxs = valid_idxs[1350:1500]
    print("train_idxs (sorted):", np.sort(train_idxs))

    return train_idxs, val_idxs


def load_gene_data():

    gene = pd.read_csv(os.path.join(DATA_BASE_DIR, 'gene/circadian_geneexp_data.txt'), delimiter=',', header=None)

    name = gene[0].map(lambda x: x.split('\t')[0]).values
    gene = gene.iloc[:, 1:].values.T
    print(gene.shape)

    return name, gene

def load_lorenz96_data(N, F, time_range=(0, 20), dt=0.02, zscore=False):
    """
    load lorenz96 data
    :return: data [time_len, dim]
    """
    data = None
    time = time_range[1] - time_range[0]
    file_name = 'lorenz/lorenz96_F{}_d{}_t{}.pkl'.format(F, N, int(time / 0.02))
    if not os.path.exists(os.path.join(DATA_BASE_DIR, file_name)):
        print(file_name)
        print('generating lorenz96 data...')
        data = lorenz96.gen_L96_data(N, F, time_range, dt)
        with open(os.path.join(DATA_BASE_DIR, file_name), 'wb') as file:
            pickle.dump(data, file)
    else:
        print('loading lorenz  data...')
        with open(os.path.join(DATA_BASE_DIR, file_name), 'rb') as file:
            data = pickle.load(file)
    data = data.T[10000:]
    if zscore:
        data = z_score(data)
    return data


def load_ks_data():
    """
    load KS data
    :return: data [time_len, dim]
    """
    data = np.genfromtxt(os.path.join(DATA_BASE_DIR, 'KSequ_80000.csv'), delimiter=',')
    data = data.T[:, :512]
    return data

def load_lorenz_data(time_invariant=True, n=30, time=100):
    """
    load lorenz data
    :return: data [time_len, dim]
    """
    data = None
    file_name = 'lorenz{}_d{}_t{}.pkl'.format('' if time_invariant else '_time_variant', n*3, int(time / 0.02))
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

def load_long_data():
    file_name = 'new.pkl'
    with open(os.path.join(DATA_BASE_DIR, file_name), 'rb') as file:
        data = pickle.load(file)

    return data

def load_traffic_data(loc='metr-la', window_size=None, fill_zero_with_mean=False):
    """
    load traffic dataset
    :param loc:
    :param window_size:
    :param fill_zero_with_mean: whether to use mean value to fill the 0s.
    :return:
    """
    assert loc in ['metr-la', 'pems-bay'], 'error file name'
    data = pd.read_hdf(os.path.join(DATA_BASE_DIR, 'traffic/{}.h5'.format(loc)))

    excel_path = os.path.join(DATA_BASE_DIR, 'traffic/{}.csv'.format(loc))
    if not os.path.exists(excel_path):
        print(excel_path)
        data.to_csv(excel_path)

    if fill_zero_with_mean:
        data[data == 0.0] = np.nan  # first convert 0 to np.nan
        data = data.fillna(data.mean())

    data = data.values
    if window_size is not None:
        windowed_data = []
        for i in range(data.shape[0] - window_size + 1):
            windowed_data.append(np.mean(data[i:i + window_size], axis=0))
        data = np.stack(windowed_data)

    print(data.shape)
    return data


def window(data, size, stride):

    t_len = data.shape[0]  # length of raw data
    win_len = (t_len - size) // stride + 1  # length of data after applying a window

    print(t_len)
    print(win_len)

    win_data = []
    for i in range(win_len):
        win_data.append(np.mean(data[i*stride:i*stride + size], axis=0))
    win_data = np.stack(win_data)
    print(win_data.shape)
    return win_data

def z_score(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0) #标准化
