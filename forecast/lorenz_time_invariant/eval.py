from forecast.lorenz_time_invariant_bzh import st_delay_model
from data import data_processing
from forecast.lorenz_time_invariant_bzh import lorenz_time_invariant_config
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


    plt.axvspan(0, 44, alpha=0.5, color='#d8e1f3')  # 添加紫色背景
    # different
    fontsize = 17

    plt.title("{}\nRMSE:{:.4f},PCC:{:.5f}".format(title, loss, pcc),
              fontdict={'family': 'Times New Roman', 'size': fontsize})
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
                    zorder=10, linewidths=1.2, s=40)
    else:
        plt.plot(x, predict_y, color='red', marker='.')

        connected_y = np.stack([known_y[-1], predict_y[0]])
        x = np.arange(train_len - 1, train_len + 1)
        plt.plot(x, connected_y, color='red')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.clf()


def get_model_path(base_log_dir, config):
    """
    get the model saved path by config
    :param config:
    :return:
    """
    name_pattern = re.compile('.*\({}\)'.format(config.name))
    # name_pattern = re.compile('.*noise_{:.1f}\)'.format(config.DATA_NOISE_STRENGTH))
    # name_pattern = re.compile('.*\(.*rate_{:.2f}\)'.format(config.DATASET_RATE))
    # name_pattern = re.compile('.*\(lorenz_{}.*\)'.format(config.TRAIN_LEN))

    file_pattern = re.compile('weights_epoch_{:0>4d}.*'.format(config.EPOCHS))
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
    plt.imshow(matrix.T, aspect='auto', cmap='viridis')  # 转置以匹配论文中的形式
    plt.colorbar()
    plt.xlabel('Time Steps')
    plt.ylabel('Embedding Dimension')
    plt.savefig(path)
    plt.close()

def calculate_metrics(y_true, y_pred):

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # RMAE (Relative Mean Absolute Error)
    mean_true = np.mean(np.abs(y_true))
    rmae = mae / mean_true if mean_true > 1e-10 else None

    # R-squared (Coefficient of Determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else None

    # Adjusted R-squared
    n = len(y_true)
    k = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if r2 is not None and n > k + 1 else None


    pcc, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (None, None)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'RMAE': rmae,
        'R2': r2,
        'Adjusted_R2': adj_r2,
        'PCC': pcc
    }
def print_metrics(metrics, set_name):
    print(f"\n===== {set_name.upper()} Metrics =====")
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"RMAE: {metrics['RMAE']:.6f}" if metrics['RMAE'] is not None else "RMAE: N/A")
    print(f"R²: {metrics['R2']:.6f}" if metrics['R2'] is not None else "R²: N/A")
    print(f"Adjusted R²: {metrics['Adjusted_R2']:.6f}" if metrics['Adjusted_R2'] is not None else "Adjusted R²: N/A")
    print(f"PCC: {metrics['PCC']:.6f}" if metrics['PCC'] is not None else "PCC: N/A")
    print(f"Mean PCC: {metrics['Mean_PCC']:.6f}" if metrics['Mean_PCC'] is not None else "Mean PCC: N/A")


def analyze_features(model, generator, config):

    all_spatial_features = []
    all_temporal_features = []

    for i, item in enumerate(generator.get_item()):
        if i >= 10:
            break

        input_x = item[0][0]

        predictions = model.model.predict(input_x)

        if len(predictions) >= 4:
            delay_output, predict_y, spatial_features, temporal_features = predictions
        else:
            print(f"模型输出数量不足: {len(predictions)}，跳过特征分析")
            continue

        spatial_feature = spatial_features[0]
        temporal_feature = temporal_features[0]

        all_spatial_features.append(spatial_feature)
        all_temporal_features.append(temporal_feature)

        if i == 0:
            visualize_features(spatial_feature, temporal_feature, input_x[0], i)


    if all_spatial_features:
        spatial_array = np.array(all_spatial_features)
        temporal_array = np.array(all_temporal_features)


        np.save('spatial_features.npy', spatial_array)
        np.save('temporal_features.npy', temporal_array)
        print("特征已保存为 spatial_features.npy 和 temporal_features.npy")


def visualize_features(spatial_feature, temporal_feature, original_input, sample_idx):

    import matplotlib.pyplot as plt


    feature_dir = os.path.join('logs', 'results', 'feature_analysis')
    os.makedirs(feature_dir, exist_ok=True)


    plt.figure(figsize=(20, 12))

    plt.subplot(2, 4, 1)
    for i in range(min(3, original_input.shape[1])):
        plt.plot(original_input[:, i], label=f'Feature{i}', alpha=0.7)
    plt.title('Original input data')
    plt.xlabel('Time step')
    plt.ylabel('value')
    plt.legend()


    plt.subplot(2, 4, 2)
    plt.imshow(spatial_feature.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Spatial feature heat map')
    plt.xlabel('Time step')
    plt.ylabel('Feature dimension')

    plt.subplot(2, 4, 3)
    plt.imshow(temporal_feature.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Temporal characteristic heat map')
    plt.xlabel('Time step')
    plt.ylabel('Feature dimension')


    plt.subplot(2, 4, 4)
    spatial_means = np.mean(spatial_feature, axis=0)
    spatial_stds = np.std(spatial_feature, axis=0)
    plt.bar(range(len(spatial_means)), spatial_means, yerr=spatial_stds, alpha=0.7)
    plt.title('Statistics of spatial feature dimensions')
    plt.xlabel('Feature dimension')
    plt.ylabel('Mean')

    plt.subplot(2, 4, 5)
    temporal_means = np.mean(temporal_feature, axis=0)
    temporal_stds = np.std(temporal_feature, axis=0)
    plt.bar(range(len(temporal_means)), temporal_means, yerr=temporal_stds, alpha=0.7)
    plt.title('Statistics of the time characteristic dimension')
    plt.xlabel('Feature dimension')
    plt.ylabel('Mean')

    plt.subplot(2, 4, 6)
    for i in range(min(5, spatial_feature.shape[1])):
        plt.plot(spatial_feature[:, i], label=f'Spatial characteristics{i}', alpha=0.7)
    plt.title('Spatial characteristics change over time')
    plt.xlabel('Time step')
    plt.ylabel('Characteristic value')
    plt.legend()


    plt.subplot(2, 4, 7)
    for i in range(min(5, temporal_feature.shape[1])):
        plt.plot(temporal_feature[:, i], label=f'Time characteristics{i}', alpha=0.7)
    plt.title('The temporal characteristics change over time')
    plt.xlabel('Time step')
    plt.ylabel('Characteristic value')
    plt.legend()


    plt.subplot(2, 4, 8)
    plt.hist(spatial_feature.flatten(), bins=50, alpha=0.7, label='Spatial characteristics', density=True)
    plt.hist(temporal_feature.flatten(), bins=50, alpha=0.7, label='Time characteristics', density=True)
    plt.title('Feature distribution')
    plt.xlabel('Characteristic value')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(feature_dir, f'features_sample_{sample_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()




if __name__ == '__main__':
    is_solar = False
    config = lorenz_time_invariant_config.LorenzTimeInvariantConfig()
    config.BATCH_SIZE = 1  # set the batch size as 1

    tf.keras.backend.clear_session()  # clear the previous session

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # ========== 1. 加载原始数据 ==========
    time_invariant = True
    # data = data_processing.load_lorenz_data(time_invariant=time_invariant, n=3, time=3000)
    data = data_processing.load_lorenz_data(
        file_path='./logs/data/lorenz_d9_t150000.pkl'
    )

    noised_data, y = data_processing.add_noise(data, config)

    total_length = len(data)
    print(f"total_length: {total_length}")
    train_idxs, val_idxs, test_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(
        total_length=total_length,
        train_len=config.TRAIN_LEN,
        embedding_len=config.EMBEDDING_LEN,
        rate=1,
        interval=60
    )
    print(train_idxs)
    print(val_idxs)

    scaler_path = os.path.join('logs', config.name, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化参数文件不存在: {scaler_path}，请先运行训练脚本。")

    with open(scaler_path, 'rb') as f:
        scaler_params = pickle.load(f)
    mean = scaler_params['mean']
    std = scaler_params['std']
    y_mean = scaler_params.get('y_mean', mean[config.Y_IDX])  # 兼容旧版本
    y_std = scaler_params.get('y_std', std[config.Y_IDX])

    noised_data = (noised_data - mean) / std

    y = (y - y_mean) / y_std


    test_generator = st_delay_model.DataGeneratorForLengthCmp(noised_data, y, test_idxs, config)
    train_generator = st_delay_model.DataGeneratorForLengthCmp(noised_data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(noised_data, y, val_idxs,
                                                                                           config)

    model_path = r'./lorenz_time_invariant/logs/lorenz_96_18_Yidx_0/weights_epoch_0100_loss'


    model = st_delay_model.STDelayModel(config, mode='evaluation', log_dir_suffix=config.name)
    model.load_weights(model_path)

    sets = {'train': train_generator, 'val': val_generator, 'test': test_generator}
    prediction_results_df = pd.DataFrame(columns=[
        'set', 'sample_index', 'time_index', 'known_y', 'true_y', 'pred_y',
        'rmse', 'pcc', 'p_value'
    ])

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
            input_x, original_label_y, label_matrix = item[0][0], item[-1], item[1][0]  # 重命名避免冲突

            predictions = model.model.predict(input_x)
            delay_output = predictions[0]
            predict_y = predictions[1]

            spatial_features = predictions[2]
            temporal_features = predictions[3]
            predict_y_matrix = delay_output[0]
            predict_y = utils.get_y_from_matrix(config, predict_y_matrix.T, weighted=False)

            save_delay_matrix(delay_output[0].T, os.path.join(result_dir, f'delay_matrix_{idx}.npy'))
            plot_delay_matrix(delay_output[0].T, os.path.join(result_dir, f'delay_matrix_heatmap_{idx}.png'))

            known_y = input_x[0, :, config.Y_IDX]
            label_y = original_label_y[0]

            min_length = min(len(label_y), len(predict_y))
            if len(label_y) != len(predict_y):
                print(
                    f"警告: 长度不匹配 - 索引{idx}: label_y长度{len(label_y)}, predict_y长度{len(predict_y)}. 将截断至{min_length}长度.")
                label_y = label_y[:min_length]
                predict_y = predict_y[:min_length]

            loss = np.sqrt(np.mean(np.square(label_y - predict_y)))
            pcc, p_value = pearsonr(label_y, predict_y)

            label_y_orig = label_y * y_std + y_mean
            predict_y_orig = predict_y * y_std + y_mean
            loss_orig = np.sqrt(np.mean(np.square(label_y_orig - predict_y_orig)))

            if set_name == 'test':

                result_csv_dir = os.path.join(result_dir, 'prediction_details')
                os.makedirs(result_csv_dir, exist_ok=True)

                sample_result = {
                    'index': idx,
                    'known_y': [known_y.tolist()],
                    'true_y': [label_y.tolist()],
                    'pred_y': [predict_y.tolist()],
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

            known_ys.append(known_y)
            predict_ys.append(predict_y)
            label_ys.append(label_y)
            losses.append(loss)

            pccs.append(pcc)


            time_index = test_idxs[idx] if set_name == 'test' else (val_idxs[idx] if set_name == 'val' else train_idxs[idx])  # 修复变量名

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
                file_path = os.path.join('logs', 'results', '{}_lorenz_idx0_prediction_noise_results.pkl'.format(config.name))

                with open(file_path, 'wb') as file:
                    pickle.dump(prediction_results, file)

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
            'prediction_length': int(len(test_predict_ys[0])) if test_predict_ys else 0  # 转换为Python int
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
        plt.savefig(os.path.join(test_result_dir, 'lorenz_idx0_prediction_noise_samples.png'))
        plt.close()

    prediction_excel_path = './logs/results/{}/lorenz.xlsx'.format(config.name)
    os.makedirs(os.path.dirname(prediction_excel_path), exist_ok=True)
    prediction_results_df.to_excel(prediction_excel_path, index=False)