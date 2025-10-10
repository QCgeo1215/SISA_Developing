import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR,CyclicLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau
from lstm_model import LSTMModel
from shard_slice_utils import shard_slice_data
from my_dataset import MyDataset
from collate_functions import pad_collate_fn
from collate_functions import pad_collate_fn_V2
#from torch.utils.data import Dataset
import ast
import yaml
import time
import sys
import argparse
import os

import random


def set_global_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to: {seed}")

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def Loss_model_training(train_npy_file, source_data, shard_index, num_shards, num_slices, batch_size, output_size, device, epoch_shard_model_abs_error_training, model_A_path, model_B_path):
    # 加载分片数据
    shard_data = np.load(train_npy_file, allow_pickle=True).item()
    #print(shard_data)
    # 检查 shard_index 是否有效
    if shard_index not in shard_data["shards"]:
        raise ValueError(f"无效的 shard_index: {shard_index}。有效范围为 1 到 {num_shards}。")
    trj_id_sets = shard_data["shards"][shard_index]
    if not trj_id_sets:
        raise ValueError(f"Shard {shard_index} 中没有分配到任何 trj_id_set。")
    # 切片数据
    slice_size = max(len(trj_id_sets) // num_slices, 1)
    print(slice_size)
    slices = [
        trj_id_sets[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)
    ]
    print(f"提取 Shard {shard_index} 中的全部数据")
    shard_data = source_data[source_data['trj_id_set'].isin(np.concatenate(slices))]
    # Dataset and DataLoader
    dataset = MyDataset(shard_data)
    set_global_seed(3407)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=False)
    # 加载 model_A 并计算绝对误差
    model_A = LSTMModel(input_size=2, output_size=output_size).to(device)
    model_A.load_state_dict(torch.load(f"{model_A_path}/final_model_shard-idx_{shard_index}_shard-num_{num_shards}.pth"))
    model_A.eval()
    all_true_values, all_predictions = [], []
    with torch.no_grad():
        for x_batch, y_batch, lengths, first_pos_x, last_pos_x in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_batch = y_batch.view(-1, output_size)
            last_pos_x = last_pos_x.to(device)
            # 模型预测
            y_pred_batch = model_A(x_batch, lengths,last_pos_x)
            #y_pred_batch = model_A(x_batch, lengths)
            y_pred_batch = torch.round(y_pred_batch)
            all_true_values.append(y_batch.cpu().numpy())
            all_predictions.append(y_pred_batch.cpu().numpy())            
    all_true_values = np.concatenate(all_true_values, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    model_A_train_sample_pre_absolute_error = np.abs(all_predictions - all_true_values)
    print("dataset_len")
    # Step 1: 创建并保存 DataFrame
    error_data = pd.DataFrame({
        'lat_grid': shard_data['lat_grid'],  # lat_grid 列
        'lng_grid': shard_data['lng_grid'],  # lng_grid 列
        'final_grid_label': list(model_A_train_sample_pre_absolute_error)  # 转换为列表存储
    })
    error_data['final_grid_label'] = error_data['final_grid_label'].apply(
    lambda x: str([int(round(i)) for i in x]))  # 转换为字符串形式并四舍五入为整数)
    print("加载的 Error DataFrame 示例：")
    print(error_data.head(2))
    # 创建 DataLoader
    set_global_seed(3407)
    error_dataloader = DataLoader(MyDataset(error_data), batch_size=batch_size, collate_fn=pad_collate_fn_V2,shuffle=True)
    
    # 定义并训练 model_B
    model_B = LSTMModel(input_size=2, output_size=output_size).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model_B.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
    # Early Stopping 参数
    patience = 20  # 允许的验证损失不下降的 epoch 数
    min_delta = 0.001  # 视为改进的最小变化阈值
    best_loss = float('inf')  # 初始化最佳损失
    early_stop_counter = 0  # 早停计数器
    
    for epoch in range(epoch_shard_model_abs_error_training):
        model_B.train()
        running_loss = 0.0
        for x_batch, y_batch, lengths, first_pos_x, last_pos_x in error_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            last_pos_x = last_pos_x.to(device)
            y_batch = y_batch.view(-1, output_size)
            # 模型前向传播
            outputs = model_B(x_batch, lengths, last_pos_x)
            # 计算损失
            loss = criterion(outputs, y_batch)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 计算平均损失
        avg_loss = running_loss / len(dataloader)
        #scheduler.step()
        scheduler.step(avg_loss)
        if (epoch + 1) % 1 == 0 or epoch == epoch_shard_model_abs_error_training - 1:
            print(f"Shard {shard_index}, Loss model, Epoch {epoch + 1}/{epoch_shard_model_abs_error_training}, Loss: {avg_loss:.4f}")

            # Early Stopping 逻辑
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            early_stop_counter = 0  # 重置计数器
        else:
            early_stop_counter += 1  # 损失未改善，计数器加 1

        # 检查是否触发早停
        if early_stop_counter >= patience:
            print(f"== Early stopping triggered at epoch {epoch + 1} ==")
            break  # 终止训练循环

    model_B_path = f"{model_B_path}/final_loss_model_shard-idx_{shard_index}_shard-num_{num_shards}.pth"
    torch.save(model_B.state_dict(), model_B_path)
    print(f"Loss Model for Shard {shard_index} saved to {model_B_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_index', type=int, help="需要处理shard的索引")
    parser.add_argument('--config', type=str, default='./config-scaling.yaml', help="配置文件路径")
    args = parser.parse_args()
    print("args.shard_index:",int(args.shard_index))
    config = load_config(args.config)
    if config is None:
        print("Config loading failed.")
        return
    num_shards = config['num_shards']
    num_slices = config['num_slices']
    # 参数配置
    start_time = time.time()
    shard_index = int(args.shard_index)
    csv_file = config['source_file']
    # 加载原始数据
    source_data = pd.read_csv(csv_file)
    print(f"原始数据加载完成，共 {len(source_data)} 条记录")
    save_path = config['output_data_path']
    folder_name = f"Model_shard_{num_shards}_slice_{num_slices}"
    full_path = os.path.join(save_path, folder_name)

    # 开始训练
    try:
        print("loss model start training index:", shard_index)
        Loss_model_training(
            train_npy_file=full_path + '/npy_index_file/training_shard_cluster_assignment.npy',
            source_data=source_data,
            shard_index=int(args.shard_index),
            num_shards=config['num_shards'],
            num_slices=config['num_slices'],
            batch_size=config['SISA_Lstm_training_batch_size'],
            output_size=config['output_size'],
            device=config['device'],
            epoch_shard_model_abs_error_training=config['Loss_model_training_epoch'],
            model_A_path=full_path + '/4_1_SISA_Sort_lstm/SISA_final_model',
            model_B_path=full_path + '/4_1_SISA_Sort_lstm/shard_loss_model'
        )
        print("Loss Model training completed.")
    except ValueError as e:
        print(f"错误: {e}")

    time_info_txt = full_path + '/4_1_SISA_Sort_lstm/SISA_Sub_model_Loss_training_time.txt'
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time

    with open(time_info_txt, "w") as file:
        file.write(f"Total Loss training time : {round(elapsed_time / 60, 3)} minutes")


if __name__ == "__main__":
    main()
