import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from lstm_model import LSTMModel
from my_dataset import MyDataset
from collate_functions import pad_collate_fn
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import yaml
import os
import time
import sys
import argparse

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
    

def load_training_data(csv_file, npy_file):

    train_trj_id_set = np.load(npy_file)
    #print(test_trj_id_set)
    source_data = pd.read_csv(csv_file)
    train_data = source_data[source_data['trj_id_set'].isin(train_trj_id_set)].copy()

    return train_data

def train_lstm_model(train_data, batch_size, input_size, output_size, num_epochs, device, model_save_path):
    # 数据加载
    dataset = MyDataset(train_data)
    set_global_seed(3407)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=True)

    # 初始化模型、损失函数、优化器
    model = LSTMModel(input_size=input_size, output_size=output_size).to(device)
    criterion = nn.SmoothL1Loss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97)
        # 开始训练
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        scheduler = CosineAnnealingLR(optimizer, T_max=(num_epochs/2), eta_min=5e-7)
        for x_batch, y_batch, lengths, first_pos_x, last_pos_x in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            last_pos_x = last_pos_x.to(device)
            y_batch = y_batch.view(-1, output_size)
            # 模型前向传播
            outputs = model(x_batch, lengths, last_pos_x)
            # 计算损失
            loss = criterion(outputs, y_batch)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 学习率调度
        scheduler.step()
        if (epoch + 1) % 1 == 0:
            # 打印每个 epoch 的损失
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")

    return model


def main():
    # 参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config-scaling.yaml', help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None:
        print("Config loading failed.")
        return
    
    # 记录开始时间
    start_time = time.time()
    csv_file = config['source_file']
    num_shards = config['num_shards']
    num_slices = config['num_slices']
    save_path = config['output_data_path']
    # 生成文件夹名称
    folder_name = f"Model_shard_{num_shards}_slice_{num_slices}"
    # 拼接完整路径
    full_path = os.path.join(save_path, folder_name)
    npy_file = full_path +'/npy_index_file/train_trj_id_sets.npy'
    batch_size = config['Lstm_training_batch_size']  # 可以根据 GPU 资源调整批量大小
    input_size=config['input_size']
    output_size=config['output_size']  # 输出标签维度
    num_epochs = config['Lstm_training_epoch']   # 训练轮次
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = full_path + '/2_1_Direct_lstm/final_Lstm_model.pth'
    
    # 加载训练数据
    train_data = load_training_data(csv_file, npy_file)
    print(f"训练数据加载完成，共 {len(train_data)} 条记录")

    # 开始训练
    train_lstm_model(
        train_data=train_data,
        batch_size=batch_size,
        input_size=input_size,
        output_size=output_size,
        num_epochs=num_epochs,
        device=device,
        model_save_path=model_save_path
    )
    time_info_txt = full_path + '/2_1_Direct_lstm/Direct_Lstm_training_time.txt'
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time
    
    with open(time_info_txt, "w") as file:
        file.write(f"Total training time : {round(elapsed_time/60,3)} minutes")
    

if __name__ == "__main__":
    main()