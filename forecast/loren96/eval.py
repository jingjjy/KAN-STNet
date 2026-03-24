from forecast import st_delay_model
from data import data_processing
from forecast.lorenz96 import lorenz96_config
from scipy.stats import pearsonr
import numpy as np
import pickle
import tensorflow as tf
import re
import os
from utils import utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import pandas as pd


def draw_pic(known_y, label_y, predict_y, loss, pcc, x_label=None, y_label=None, y_lim=None,
             title=None, path=None, figsize=None):

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['savefig.dpi'] = 200

    fontsize = 17
    plt.title(title + ",RMSE:{:.4F},PCC:{:.4f}".format(loss,pcc), fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.xlabel(x_label, fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.ylabel(y_label, fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.yticks(fontproperties='Times New Roman', size=fontsize)
    plt.xticks(fontproperties='Times New Roman', size=fontsize)

    if y_lim is not None:
        plt.ylim(*y_lim)

    train_len = len(known_y)
    all_y = np.concatenate([known_y, label_y])
    x = np.arange(len(all_y))
    plt.plot(x, all_y, color='blue', marker='.')

    x = np.arange(train_len, len(all_y))


    if title == 'lorenz':
        plt.scatter(x, predict_y, color='none', edgecolors='red', marker='o',
                    label='predict'.format(loss, pcc),
                    zorder=10, linewidths=1.2, s=40)
    else:
        plt.plot(x, predict_y, color='red', marker='.', label='predict'.format(loss, pcc))
        connected_y = np.stack([known_y[-1], predict_y[0]])
        x = np.arange(train_len - 1, train_len + 1)
        plt.plot(x, connected_y, color='red')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=2, mode="expand", borderaxespad=0.)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.clf()

def get_model_path(base_log_dir, config):
    name_pattern = re.compile('.*\({}\)'.format(config.name))
    file_pattern = re.compile('weights_epoch:{:0>4d}.*'.format(config.EPOCHS))
    model_path = None
    for d in os.listdir(base_log_dir):
        print(d)
        if name_pattern.match(d):
            for f in os.listdir(os.path.join(base_log_dir, d)):
                if file_pattern.match(f):
                    model_path = os.path.join(base_log_dir, d, f)
                    print('load weights from: {}'.format(model_path))
                    return model_path

    return model_path

def save_delay_matrix(delay_matrix, path):
    np.save(path, delay_matrix)

def plot_delay_matrix(matrix, path):
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Time Steps')
    plt.ylabel('Embedding Dimension')
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    is_solar = False
    config = lorenz96_config.Lorenz96Config()
    config.BATCH_SIZE = 1
    config.ADD_NOISE = True
    config.DATA_NOISE_STRENGTH = 1.0
    tf.keras.backend.clear_session()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    data = data_processing.load_lorenz96_data(N=config.K, F=config.F, time_range=(0, 3000), dt=0.02)
    print(f"数据形状: {data.shape}")
    total_length = len(data)
    train_idxs, val_idxs, test_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(
        total_length=total_length,
        train_len=config.TRAIN_LEN,
        embedding_len=config.EMBEDDING_LEN,
        rate=1,
        interval=60
    )
    print(train_idxs)
    print(val_idxs)
    mean_train_losses = []
    mean_train_pccs = []
    results = {}

    model = st_delay_model.STDelayModel(config, mode='evaluation', log_dir_suffix=config.name)

    noised_data, y = data_processing.add_noise(data, config)

    noise_data_df = pd.DataFrame({
        'time_index': np.arange(len(noised_data)),
        'noised_value': noised_data[:, config.Y_IDX],
        'original_value': y
    })
    lorenz96_noise_data_excel_path = '../../logs/results/{}/lorenz96_noise1.0_data.xlsx'.format(config.name)
    os.makedirs(os.path.dirname(lorenz96_noise_data_excel_path), exist_ok=True)
    noise_data_df.to_excel(lorenz96_noise_data_excel_path, index=False)

    prediction_results_df = pd.DataFrame(columns=[
        'set', 'sample_index', 'time_index', 'known_y', 'true_y', 'pred_y',
        'rmse', 'pcc', 'p_value'
    ])

    train_generator = st_delay_model.DataGeneratorForLengthCmp(data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(data, y, val_idxs, config)
    test_generator = st_delay_model.DataGeneratorForLengthCmp(data, y, test_idxs, config)


    model_path = r'/weights_epoch_0146_loss_1.005_val_loss_1.963_predict_loss_1.101.h5'

    print(model_path)
    model.load_weights(model_path)

    sets = {'train': train_generator, 'val': val_generator, 'test': test_generator}

    test_losses = []
    test_pccs = []
    test_predict_ys = []
    test_label_ys = []
    test_result_dir = None

    for set_name in sets:
        if sets[set_name] is None:
            continue
        print(set_name)
        generator = sets[set_name]
        idx = 0
        result_dir = os.path.join('logs', 'results', config.name, set_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        losses = []
        pccs = []

        predict_ys = []
        label_ys = []
        known_ys = []

        for item in generator.get_item():
            input_x, original_label_y, label_matrix = item[0][0], item[-1], item[1][0]

            delay_output, predict_y = model.model.predict(input_x)
            predict_y_matrix = delay_output[0]
            predict_y = utils.get_y_from_matrix(config, predict_y_matrix.T, weighted=False)

            save_delay_matrix(delay_output[0].T, os.path.join(result_dir, f'delay_matrix_{idx}.npy'))
            plot_delay_matrix(delay_output[0].T, os.path.join(result_dir, f'delay_matrix_heatmap_{idx}.png'))

            known_y = input_x[0, :, config.Y_IDX]
            label_y = original_label_y[0]

            loss = np.sqrt(np.mean(np.square(label_y - predict_y)))
            pcc, p_value = pearsonr(label_y, predict_y)

            if set_name == 'test':

                result_csv_dir = os.path.join(result_dir, 'lorenz96_prediction_noise1.0_details')
                os.makedirs(result_csv_dir, exist_ok=True)

                sample_result = {
                    'index': idx,
                    'known_y': known_y.tolist(),
                    'true_y': label_y.tolist(),
                    'pred_y': predict_y.tolist(),
                    'rmse': float(loss),
                    'pcc': float(pcc)
                }

                with open(f'{result_csv_dir}/sample_{idx}.json', 'w') as f:
                    json.dump(sample_result, f, indent=2)

                df = pd.DataFrame({
                    'time_step': np.arange(len(label_y)),
                    'true_value': label_y,
                    'pred_value': predict_y
                })
                df.to_csv(f'{result_csv_dir}/sample_{idx}.csv', index=False)

            loss = np.sqrt(np.mean(np.square(label_y - predict_y)))
            known_ys.append(known_y)
            predict_ys.append(predict_y)
            label_ys.append(label_y)
            losses.append(loss)

            min_length = min(len(label_y), len(predict_y))
            if len(label_y) != len(predict_y):
                label_y = label_y[:min_length]
                predict_y = predict_y[:min_length]

            pcc, p_value = pearsonr(label_y, predict_y)
            pccs.append(pcc)

            time_index = test_idxs[idx] if set_name == 'test' else (
                val_idxs[idx] if set_name == 'val' else train_idxs[idx])

            new_row = pd.DataFrame({
                'set': [set_name],
                'sample_index': [idx],
                'time_index': [time_index],
                'known_y': [known_y.tolist()],
                'true_y': [label_y.tolist()],
                'pred_y': [predict_y.tolist()],
                'rmse': [loss],
                'pcc': [pcc],
                'p_value': [p_value]
            })
            prediction_results_df = pd.concat([prediction_results_df, new_row], ignore_index=True)

            pathway_1 = result_dir
            os.makedirs(pathway_1, exist_ok=True)
            if set_name == 'test':
                draw_pic(known_y, label_y, predict_y, loss, pcc=pcc, path=os.path.join(pathway_1, '{}.png'.format(idx)),
                         figsize=(8, 6), y_lim=None, x_label='Time', y_label='$d_{k}$',
                         title='Noise-free case($\sigma = 0$)')  # lorenz

            idx += 1

            if idx % 100 == 0:
                print(idx)

            print(set_name)
            print(np.sum(np.array(losses) < 1.0))
            print('PCC:', pccs)
            print('loss:', losses)
            print('mean loss：', np.mean(losses))
            print('mean pcc:', np.mean(pccs))
            print(np.argsort(losses)[:100])
            print(np.argsort(pccs)[::-1][:100])

            if set_name == 'train':
                prediction_results = {}
                prediction_results['train_ys'] = known_ys
                prediction_results['label_ys'] = label_ys
                prediction_results['predict_ys'] = predict_ys
                prediction_results['rmses'] = losses
                prediction_results['mean_rmses'] = np.mean(losses)
                prediction_results['pccs'] = pccs
                prediction_results['mean_pccs'] = np.mean(pccs)
                file_path = os.path.join('logs', 'results', '{}_prediction_noise1.0_results.pkl'.format(config.name))

                with open(file_path, 'wb') as file:
                    pickle.dump(prediction_results, file)

            print("实际预测长度:", len(predict_y))
            print("预测值示例:", predict_y[:5])
            print("标签值示例:", label_y[:5])

        if set_name == 'test':
            test_losses = losses
            test_pccs = pccs
            test_predict_ys = predict_ys
            test_label_ys = label_ys
            test_result_dir = result_dir

    if test_losses:
        summary = {
            'avg_rmse': float(np.mean(test_losses)),
            'avg_pcc': float(np.mean(test_pccs)),
            'best_samples': {
                'lowest_rmse': int(np.argmin(test_losses)),
                'highest_pcc': int(np.argmax(test_pccs))
            },
            'prediction_length': int(len(test_predict_ys[0])) if test_predict_ys else 0
        }

        with open(os.path.join(test_result_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        plt.figure(figsize=(15, 10))
        for i in range(min(9, len(test_label_ys))):
            plt.subplot(3, 3, i + 1)
            plt.plot(test_label_ys[i], 'b-', label='True')
            plt.plot(test_predict_ys[i], 'r--', label='Pred')
            plt.title(f'Sample {i} (RMSE: {test_losses[i]:.3f})')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(test_result_dir, 'lorenz96_prediction_noise1.0_samples.png'))
        plt.close()

    # 保存预测结果到Excel文件
    prediction_excel_path = '../../logs/results/{}/lorenz96_prediction_noise1.0_results.xlsx'.format(config.name)
    os.makedirs(os.path.dirname(prediction_excel_path), exist_ok=True)
    prediction_results_df.to_excel(prediction_excel_path, index=False)

