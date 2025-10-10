import numpy as np
import pandas as pd
import torch
from lstm_model import LSTMModel
from my_dataset import MyDataset
from collate_functions import pad_collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
from manhattan_distance import is_prediction_in_threshold_range
import os
import sys
import argparse

import random

# upload at 2025/07/11

def set_global_seed(seed=None):
    if seed is None:
        # 使用当前时间的毫秒部分作为种子（确保每次不同）
        seed = int(time.time() * 1000) % (2**32)  # 限制在32位整数范围内
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed = 3407
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to: {seed}")  # 可选：打印种子值用于调试
    

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def load_test_data(csv_file, npy_file):

    # 加载 CSV 文件并检查是否为空
    data = pd.read_csv(csv_file)
    if data.empty:
        raise ValueError(f"CSV 文件 {csv_file} 为空或无法加载")

    test_trj_id_set = np.load(npy_file)
    #print(test_trj_id_set)
    source_data = pd.read_csv(csv_file)
    test_data = source_data[source_data['trj_id_set'].isin(test_trj_id_set)].copy()
    #test_data = test_data[:100]
    return test_data

def evaluate_lstm_model(test_data, batch_size, input_size, output_size, device, model_load_path, thresholds):
    # 加载测试数据
    dataset = MyDataset(test_data)
    set_global_seed(3407)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=False)

    # 加载模型
    model = LSTMModel(input_size=input_size, output_size=output_size).to(device)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    # 预测和验证
    all_true_values, all_predictions = [], []

    with torch.no_grad():
        #for x_batch, y_batch, lengths in dataloader:
        for x_batch, y_batch, lengths, first_pos_x, last_pos_x in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_batch = y_batch.view(-1, output_size)
            last_pos_x = last_pos_x.to(device)
            # 模型预测
            y_pred_batch = model(x_batch, lengths,last_pos_x)

            # 保存预测和真实值
            all_true_values.append(y_batch.cpu().numpy())
            all_predictions.append(y_pred_batch.cpu().numpy())

    # 合并预测值和真实值
    all_true_values = np.concatenate(all_true_values, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # 计算精度
    accuracy_results = calculate_accuracy(all_predictions, all_true_values, thresholds)
    return accuracy_results

def calculate_accuracy(aggregated_predictions, true_values, thresholds=[1]):
    results = {threshold: {"correct": 0, "total": len(true_values)} for threshold in thresholds}
    for i in range(len(true_values)):
        true_value, pred_value = true_values[i], aggregated_predictions[i]
        for threshold in thresholds:
            if is_prediction_in_threshold_range(pred_value, true_value, threshold):
                results[threshold]["correct"] += 1
    return results

def main():
    # 参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config-scaling.yaml', help="配置文件路径")
    args = parser.parse_args()
    config = load_config(args.config)
    if config is None:
        print("Config loading failed.")
        return

    csv_file = config['source_file']
    save_path = config['output_data_path']
    
    batch_size = config['Lstm_training_batch_size']
    input_size = config['input_size']
    output_size = config['output_size']
    num_shards = config['num_shards']
    num_slices = config['num_slices']
    
    folder_name = f"Model_shard_{num_shards}_slice_{num_slices}"
    full_path = os.path.join(save_path, folder_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_load_path = full_path + '/2_1_Direct_lstm/final_Lstm_model.pth'
    accuracy_info_txt = full_path + '/2_1_Direct_lstm/Lstm_model_test_accuracy.txt'
    npy_file = full_path +'/npy_index_file/test_trj_id_sets.npy'
    thresholds = [1,2,3,4,5]

    # 加载测试数据
    test_data = load_test_data(csv_file, npy_file)
    print(f"测试数据加载完成，共 {len(test_data)} 条记录")

    # 开始评估
    accuracy_results = evaluate_lstm_model(
        test_data=test_data,
        batch_size=batch_size,
        input_size=input_size,
        output_size=output_size,
        device=device,
        model_load_path=model_load_path,
        thresholds=thresholds
    )

    with open(accuracy_info_txt, "w") as f:
        for threshold, data in accuracy_results.items():
            correct_predictions, total_samples = data["correct"], data["total"]
            accuracy = correct_predictions / total_samples * 100 if total_samples > 0 else 0
            f.write(f"Threshold {threshold}:\nCorrect predictions: {correct_predictions}/{total_samples}\nAccuracy: {accuracy:.2f}%\n\n")
    print(f"Results saved to {accuracy_info_txt}")

if __name__ == "__main__":
    main()
