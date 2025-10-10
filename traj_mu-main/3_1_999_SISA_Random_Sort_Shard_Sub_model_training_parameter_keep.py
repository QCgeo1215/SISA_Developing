import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR,CyclicLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau
import torch.optim as optim
from lstm_model import LSTMModel
from my_dataset import MyDataset
from data_sta import DataStandardizer
from collate_functions import pad_collate_fn
import torch.nn as nn
import yaml
import time
import sys
import argparse
import os

import random
from torch.optim.lr_scheduler import CosineAnnealingLR

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

def load_config(config_path): # ='./config-scaling.yaml'
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def train_shard_with_slices(train_npy_file, source_data, shard_index, num_shards, num_slices, input_size, output_size, epoch_shard_slice_training, batch_size, device, intermediate_model_save_path, final_model_save_path='./model_final'):
    # 加载分片数据
    print(train_npy_file[:10])
    shard_data = np.load(train_npy_file, allow_pickle=True).item()
    print(type(shard_data))
    # 检查 shard_index 是否有效
    if shard_index not in shard_data["shards"]:
        raise ValueError(f"无效的 shard_index: {shard_index}。有效范围为 1 到 {num_shards}。")
    trj_id_sets = shard_data["shards"].get(shard_index, [])
    if not trj_id_sets:
        raise ValueError(f"Shard {shard_index} 中没有分配到任何 trj_id_set。")
    # 切片操作
    slice_size = max(len(trj_id_sets) // num_slices, 1)
    slices = [
        trj_id_sets[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)
    ]
    # 初始化模型、损失函数和优化器
    model = LSTMModel(input_size, output_size).to(device)
    criterion = nn.SmoothL1Loss()
    #optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) 
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5) # loss 2.3 2.18  #lstm模型改 6.1
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True) # loss 4
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True) # loss 2.8
    # optimizer = optim.Adam(model.parameters(), lr=0.00008, weight_decay=1e-5) 
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # 用于累积训练数据
    cumulative_data = pd.DataFrame()
    models = []
    target_id = []
    for i, trj_id_set_slice in enumerate(slices, start=1):
        #print(f"slice {i} contain {len(trj_id_set_slice)} data")
        # 使用 trj_id_set_slice 作为索引提取数据
        print(trj_id_set_slice[:10])
        target_id.extend(trj_id_set_slice)
        # print("target_id",target_id)
        save_name = f"{intermediate_model_save_path}/model_shard-idx_{shard_index}_shard-num_{num_shards}_slice_{i}_{num_slices}.pth"
        loss_file_name = f"{intermediate_model_save_path}/loss_shard-idx_{shard_index}_shard-num_{num_shards}_slice_{i}_{num_slices}.txt" 
        slice_data = source_data[source_data['trj_id_set'].isin(target_id)].copy()
        print(f"slice {i} contain {len(slice_data)} data")
        # 创建 DataLoader
        # 调用时不传参数，自动基于时间生成种子
        set_global_seed()

        dataloader = DataLoader(
            MyDataset(slice_data),
            batch_size=batch_size,
            collate_fn=pad_collate_fn,
            shuffle=True)  # 此时每次运行的shuffle顺序不同)
        #dataloader = DataLoader(MyDataset(slice_data), batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=True)
        print("start training . . . . ")
        # 打开文件以写入 loss 信息
        with open(loss_file_name, 'w') as loss_file:
            # 开始训练
            for epoch in range(epoch_shard_slice_training * i):
                model.train()
                running_loss = 0.0
                # 替代原来的ReduceLROnPlateau
                scheduler = CosineAnnealingLR(optimizer, 
                                            T_max=(epoch_shard_slice_training * i/2), 
                                            eta_min=5e-7)  # 最小学习率  # 半周期长度  # 调成 3 loss 2.6 # 调成 2 大约 2.3-2.1 #调成1为2.5
                for x_batch, y_batch, lengths, first_pos_x, last_pos_x in dataloader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    last_pos_x = last_pos_x.to(device)
                    y_batch = y_batch.view(-1, output_size)
                    # 模型前向传播
                    outputs = model(x_batch, lengths, last_pos_x)
                    # 计算损失
                    loss = criterion(outputs, y_batch)
                    # 使用关键区域聚焦损失
                    #loss = focused_smooth_l1_loss(outputs, y_batch, lengths)
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    # 增加梯度裁剪 2025/5/7
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 限制梯度范数
                    optimizer.step()
                    running_loss += loss.item()
                # 计算平均损失
                avg_loss = running_loss / len(dataloader)
                scheduler.step(avg_loss)
                
                if (epoch + 1) % 1 == 0 or epoch == epoch_shard_slice_training * i - 1:
                    print(f"Shard {shard_index}-{num_shards}, Slice {i}/{num_slices}, Epoch {epoch + 1} completed, Avg Loss: {avg_loss:.4f}")
                    loss_file.write(f"Shard {shard_index}-{num_shards}, Slice {i}/{num_slices}, Epoch {epoch + 1} completed, Avg Loss: {avg_loss:.4f}\n")
        # 保存模型
        torch.save(model.state_dict(), save_name)
        print(f"Intermediate model for Shard {shard_index}, Slice {i} saved.")
        models.append(model)
    # 保存最终模型
    torch.save(model.state_dict(), f"{final_model_save_path}/final_model_shard-idx_{shard_index}_shard-num_{num_shards}.pth")
    print(f"Final model for Shard {shard_index} saved.")
    return models

# def focused_smooth_l1_loss(outputs, y_batch, lengths, focus_ratio=0.2, base_loss=nn.SmoothL1Loss(reduction='none')):
#     """
#     关键区域聚焦的损失函数（独立定义）
#     参数:
#         outputs: 模型输出 [batch_size * seq_len, output_size]
#         y_batch: 目标值 [batch_size * seq_len, output_size]
#         lengths: 每个样本的实际长度 [batch_size]
#         focus_ratio: 关键区域比例（默认最后20%时间步）
#         base_loss: 基础损失函数
#     """
#     batch_size = len(lengths)
#     seq_len = outputs.shape[0] // batch_size
#     weights = torch.ones_like(outputs)
    
#     # 为每个样本的最后focus_ratio比例时间步赋予更高权重
#     for i in range(batch_size):
#         actual_len = lengths[i]
#         focus_start = int(actual_len * (1 - focus_ratio))
        
#         # 计算该样本在展平后的索引范围
#         start_idx = i * seq_len + focus_start
#         end_idx = i * seq_len + actual_len
        
#         weights[start_idx:end_idx] = 2.0  # 关键区域权重加倍
    
#     # 应用加权损失
#     loss = base_loss(outputs, y_batch)
#     return (weights * loss).mean()

# def train_shard_with_slices(train_npy_file, source_data, shard_index, num_shards, num_slices, 
#                           input_size, output_size, epoch_shard_slice_training, batch_size, 
#                           device, intermediate_model_save_path, final_model_save_path='./model_final'):
#     # 加载分片数据
#     print(train_npy_file[:10])
#     shard_data = np.load(train_npy_file, allow_pickle=True).item()
#     print(type(shard_data))
    
#     # 检查shard_index有效性
#     if shard_index not in shard_data["shards"]:
#         raise ValueError(f"无效的shard_index: {shard_index}。有效范围为1到{num_shards}。")
#     trj_id_sets = shard_data["shards"].get(shard_index, [])
#     if not trj_id_sets:
#         raise ValueError(f"Shard {shard_index}中没有分配到任何trj_id_set。")
    
#     # 切片操作
#     slice_size = max(len(trj_id_sets) // num_slices, 1)
#     slices = [trj_id_sets[i*slice_size:(i+1)*slice_size] for i in range(num_slices)]
    
#     # 初始化模型和优化器
#     model = LSTMModel(input_size, output_size).to(device)
#     optimizer = optim.NAdam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 使用NAdam
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
#     # 训练数据记录
#     models = []
#     target_id = []
    
#     for i, trj_id_set_slice in enumerate(slices, start=1):
#         print(f"Processing slice {i}: {trj_id_set_slice[:10]}...")
#         target_id.extend(trj_id_set_slice)
        
#         # 准备数据和保存路径
#         save_name = f"{intermediate_model_save_path}/model_shard-idx_{shard_index}_slice_{i}.pth"
#         loss_file_name = f"{intermediate_model_save_path}/loss_shard-idx_{shard_index}_slice_{i}.txt"
#         slice_data = source_data[source_data['trj_id_set'].isin(target_id)].copy()
        
#         # 创建DataLoader
#         set_global_seed()
#         dataloader = DataLoader(
#             MyDataset(slice_data),
#             batch_size=batch_size,
#             collate_fn=pad_collate_fn,
#             shuffle=True
#         )
        
#         # 训练循环
#         with open(loss_file_name, 'w') as loss_file:
#             for epoch in range(epoch_shard_slice_training * i):
#                 model.train()
#                 running_loss = 0.0
                
#                 for x_batch, y_batch, lengths, _, last_pos_x in dataloader:
#                     x_batch = x_batch.to(device)
#                     y_batch = y_batch.to(device).view(-1, output_size)
#                     last_pos_x = last_pos_x.to(device)
                    
#                     # 前向传播
#                     outputs = model(x_batch, lengths, last_pos_x)
                    
#                     # 使用关键区域聚焦损失
#                     loss = focused_smooth_l1_loss(outputs, y_batch, lengths)
                    
#                     # 反向传播
#                     optimizer.zero_grad()
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                     optimizer.step()
                    
#                     running_loss += loss.item()
                
#                 # 更新学习率和记录
#                 avg_loss = running_loss / len(dataloader)
#                 scheduler.step(avg_loss)
                
#                 log_msg = f"Shard {shard_index}, Slice {i}/{num_slices}, Epoch {epoch+1}: Loss {avg_loss:.4f}"
#                 print(log_msg)
#                 loss_file.write(log_msg + "\n")
        
#         # 保存中间模型
#         torch.save(model.state_dict(), save_name)
#         models.append(model)
    
#     # 保存最终模型
#     final_path = f"{final_model_save_path}/final_model_shard_{shard_index}.pth"
#     torch.save(model.state_dict(), final_path)
#     print(f"训练完成，最终模型已保存到: {final_path}")
    
#     return models


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
    start_time = time.time()
    shard_index=int(args.shard_index)

    csv_file = config['source_file']
    save_path = config['output_data_path']
    num_shards = config['num_shards']
    num_slices = config['num_slices']
    folder_name = f"Model_shard_{num_shards}_slice_{num_slices}"
    full_path = os.path.join(save_path, folder_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    source_data = pd.read_csv(csv_file)
    print(f"原始数据加载完成，共 {len(source_data)} 条记录")

    # 开始训练
    try:
        #for shard_index in shard_indexs:
        print("model A start training index:",shard_index)
        train_shard_with_slices(
            train_npy_file=full_path +'/npy_index_file/training_shard_random_assignment.npy',
            source_data= source_data,
            shard_index=int(args.shard_index),
            num_shards=config['num_shards'],
            num_slices=config['num_slices'],
            input_size=config['input_size'],
            output_size=config['output_size'],
            epoch_shard_slice_training=config['SISA_Lstm_training_epoch'],
            batch_size=config['SISA_Lstm_training_batch_size'],
            device=config['device'],
            intermediate_model_save_path= full_path +'/3_1_SISA_Random_lstm/SISA_sub_model',
            final_model_save_path= full_path +'/3_1_SISA_Random_lstm/SISA_final_model'
        )
            # shard_index=int(args.shard_index),
        
    except ValueError as e:
        print(f"错误: {e}")
    time_info_txt = full_path+ '/3_1_SISA_Random_lstm/SISA_model_training_time.txt'
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time
    
    with open(time_info_txt, "w") as file:
        file.write(f"Total training time : {round(elapsed_time/60,3)} minutes")

if __name__ == "__main__":
    main()