from forecast import st_delay_model
from data import data_processing
from forecast.lorenz96_bzh import lorenz96_config
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
    # 设置图形大小和DPI
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['savefig.dpi'] = 200

    # 设置标题、坐标轴标签和字体
    fontsize = 17
    plt.title(title + ",RMSE:{:.4F},PCC:{:.4f}".format(loss,pcc), fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.xlabel(x_label, fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.ylabel(y_label, fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.yticks(fontproperties='Times New Roman', size=fontsize)
    plt.xticks(fontproperties='Times New Roman', size=fontsize)

    if y_lim is not None: # 设置y轴范围
        plt.ylim(*y_lim)

    train_len = len(known_y) #绘制已知数据和标签数据
    all_y = np.concatenate([known_y, label_y])
    x = np.arange(len(all_y)) # 绘制预测数据
    plt.plot(x, all_y, color='blue', marker='.')

    x = np.arange(train_len, len(all_y))
    #x = np.arange(train_len, len(all_y))

    if title == 'lorenz': # Lorenz系统使用空心圆点表示预测
        plt.scatter(x, predict_y, color='none', edgecolors='red', marker='o',
                    label='predict'.format(loss, pcc),  # label='predict'.format(loss, pcc),
                    zorder=10, linewidths=1.2, s=40)
    else: # 其他数据使用实线连接
        plt.plot(x, predict_y, color='red', marker='.', label='predict'.format(loss, pcc)) # *args:
        connected_y = np.stack([known_y[-1], predict_y[0]])
        x = np.arange(train_len - 1, train_len + 1)
        plt.plot(x, connected_y, color='red')
    # 添加图例
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=2, mode="expand", borderaxespad=0.)
    # 显示或保存图形
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.clf()

#模型路径获取函数
def get_model_path(base_log_dir, config): # 使用正则表达式匹配模型路径
    name_pattern = re.compile('.*\({}\)'.format(config.name))
    # name_pattern = re.compile('.*noise_{:.1f}\)'.format(config.DATA_NOISE_STRENGTH))
    # name_pattern = re.compile('.*\(.*rate_{:.2f}\)'.format(config.DATASET_RATE))
    # name_pattern = re.compile('.*\(lorenz_{}.*\)'.format(config.TRAIN_LEN))

    file_pattern = re.compile('weights_epoch:{:0>4d}.*'.format(config.EPOCHS))
    model_path = None

    # 遍历目录查找匹配的模型文件
    for d in os.listdir(base_log_dir):
        print(d)
        if name_pattern.match(d):
            for f in os.listdir(os.path.join(base_log_dir, d)):
                if file_pattern.match(f):
                    model_path = os.path.join(base_log_dir, d, f)
                    print('load weights from: {}'.format(model_path))
                    return model_path

    return model_path
#jijia
def save_delay_matrix(delay_matrix, path):
    """保存延迟矩阵到文件"""
    np.save(path, delay_matrix)

def plot_delay_matrix(matrix, path):
    """绘制延迟矩阵的热力图"""
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix.T, aspect='auto', cmap='viridis')  # 转置以匹配论文中的形式
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
    config.DATA_NOISE_STRENGTH = 0.5

    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 1. 加载原始数据
    data = data_processing.load_lorenz96_data(N=config.K, F=config.F, time_range=(0, 3000), dt=0.02)
    print(f"数据形状: {data.shape}")
    total_length = len(data)

    # 2. 生成训练/验证/测试索引
    train_idxs, val_idxs, test_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(
        total_length=total_length,
        train_len=config.TRAIN_LEN,
        embedding_len=config.EMBEDDING_LEN,
        rate=1,
        interval=60
    )
    print(train_idxs)
    print(val_idxs)

    # 3. 添加噪声
    noised_data, y = data_processing.add_noise(data, config)

    # ===================== 新增：加载标准化参数并标准化数据 =====================
    scaler_path = os.path.join('logs', config.name, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化参数文件不存在: {scaler_path}，请先运行训练脚本。")
    with open(scaler_path, 'rb') as f:
        scaler_params = pickle.load(f)
    mean = scaler_params['mean']
    std = scaler_params['std']
    y_mean = scaler_params.get('y_mean', mean[config.Y_IDX])
    y_std = scaler_params.get('y_std', std[config.Y_IDX])

    noised_data = (noised_data - mean) / std
    y = (y - y_mean) / y_std

    print("\n=== 标准化验证 ===")
    print(f"输入数据目标列均值: {np.mean(noised_data[:, config.Y_IDX]):.6f}, 标准差: {np.std(noised_data[:, config.Y_IDX]):.6f}")
    print(f"标签 y 均值: {np.mean(y):.6f}, 标准差: {np.std(y):.6f}")
    # print("按 Enter 键继续...")
    # input()
    # ========================================================================

    # 4. 保存标准化后的噪声数据（可选，便于检查）
    noise_data_df = pd.DataFrame({
        'time_index': np.arange(len(noised_data)),
        'noised_value': noised_data[:, config.Y_IDX],
        'original_value': y
    })
    lorenz96_noise_data_excel_path = '../../logs/results/{}/lorenz96_noise0.5_data_standardized.xlsx'.format(config.name)
    os.makedirs(os.path.dirname(lorenz96_noise_data_excel_path), exist_ok=True)
    noise_data_df.to_excel(lorenz96_noise_data_excel_path, index=False)
    print(f"标准化噪声数据已保存到: {lorenz96_noise_data_excel_path}")

    # 5. 创建数据生成器（使用标准化数据）
    train_generator = st_delay_model.DataGeneratorForLengthCmp(noised_data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(noised_data, y, val_idxs, config)
    test_generator = st_delay_model.DataGeneratorForLengthCmp(noised_data, y, test_idxs, config)

    # 6. 加载模型权重（请确保路径正确）
    model_path = r'/media/lenovo/918B6148CC0C5D9A/jijingyi/fit_python36_weight2/forecast/lorenz96_bzh/logs/2026_08_17-20_55_07(lorenz96_bzh_96_38_Yidx_1_noise_0.5)/weights_epoch_0046_loss_2.068_val_loss_7.103_train_rmse_0.182_val_rmse_0.182_train_pcc_0.978_val_pcc_0.978.h5'
    print(f"Config name: {config.name}")
    print(f"Data shape: {data.shape}")
    print(f"Test indices: {test_idxs[:10]}...")
    print(model_path)

    model = st_delay_model.STDelayModel(config, mode='evaluation', log_dir_suffix=config.name)
    model.load_weights(model_path)

    # 7. 评估循环
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
            input_x, original_label_y, label_matrix = item[0][0], item[-1], item[1][0]
            predictions = model.model.predict(input_x)  # 获取所有输出
            delay_output = predictions[0]  # 第一个输出
            predict_y = predictions[1]  # 第二个输出
            # 如果你还需要空间/时间特征，可解包为：
            # spatial_features = predictions[2]
            # temporal_features = predictions[3]
            predict_y_matrix = delay_output[0]
            predict_y = utils.get_y_from_matrix(config, predict_y_matrix.T, weighted=False)

            # 保存延迟矩阵（可选）
            save_delay_matrix(delay_output[0].T, os.path.join(result_dir, f'delay_matrix_{idx}.npy'))
            plot_delay_matrix(delay_output[0].T, os.path.join(result_dir, f'delay_matrix_heatmap_{idx}.png'))

            known_y = input_x[0, :, config.Y_IDX]
            label_y = original_label_y[0]

            # 确保长度一致
            min_length = min(len(label_y), len(predict_y))
            if len(label_y) != len(predict_y):
                print(f"警告: 长度不匹配 - 索引{idx}: label_y长度{len(label_y)}, predict_y长度{len(predict_y)}. 截断至{min_length}长度.")
                label_y = label_y[:min_length]
                predict_y = predict_y[:min_length]

            loss = np.sqrt(np.mean(np.square(label_y - predict_y)))
            pcc, p_value = pearsonr(label_y, predict_y)

            # 保存测试集详细信息（标准化值）
            if set_name == 'test':
                result_csv_dir = os.path.join(result_dir, 'lorenz96_prediction_noise0.5_details')
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

            known_ys.append(known_y)
            predict_ys.append(predict_y)
            label_ys.append(label_y)
            losses.append(loss)
            pccs.append(pcc)

            # 记录到总DataFrame
            time_index = test_idxs[idx] if set_name == 'test' else (val_idxs[idx] if set_name == 'val' else train_idxs[idx])
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

            # 绘图
            if set_name == 'test':
                draw_pic(known_y, label_y, predict_y, loss, pcc=pcc,
                         path=os.path.join(result_dir, f'{idx}.png'),
                         figsize=(8, 6), y_lim=None, x_label='Time', y_label='$d_{k}$',
                         title='Noise-free case($\sigma = 0$)')

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
                file_path = os.path.join('logs', 'results', '{}_prediction_noise0.5_results.pkl'.format(config.name))
                with open(file_path, 'wb') as file:
                    pickle.dump(prediction_results, file)

            print("实际预测长度:", len(predict_y))
            print("预测值示例:", predict_y[:5])
            print("标签值示例:", label_y[:5])

        # 保存测试集汇总
        if set_name == 'test':
            test_losses = losses
            test_pccs = pccs
            test_predict_ys = predict_ys
            test_label_ys = label_ys
            test_result_dir = result_dir

    # 8. 汇总测试结果
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
        plt.savefig(os.path.join(test_result_dir, 'lorenz96_prediction_noise0.5_samples.png'))
        plt.close()

    # 9. 保存所有预测结果到Excel（标准化值）
    prediction_excel_path = '../../logs/results/{}/lorenz96_prediction_noise0.5_results_standardized.xlsx'.format(config.name)
    os.makedirs(os.path.dirname(prediction_excel_path), exist_ok=True)
    prediction_results_df.to_excel(prediction_excel_path, index=False)
    print(f"预测结果已保存到: {prediction_excel_path}")
