kjimport os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import ast
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from lstm_model import LSTMModel
from my_dataset import MyDataset
from torch.utils.data import Dataset, DataLoader, Subset
from collate_functions import pad_collate_fn
import argparse

# 加载配置文件
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

# 加载禁用的 trj_id 列表
def load_disabled_trj_ids(file_path='./disabled_trj_ids.yaml'):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            disabled_trj_ids = data.get('disabled_trj_id', [])
            if not isinstance(disabled_trj_ids, list):
                raise ValueError("Disabled trj_id data should be a list.")
            return disabled_trj_ids
    except Exception as e:
        print(f"Error loading disabled trj_id values: {e}")
        return None


# 查找禁用的 trj_id 对应的 slice
def find_slices_for_disabled_ids(train_npy_file, num_shards, num_slices, shard_index, disabled_trj_ids):
    # slices = shard_slice_data(npy_file, num_shards=num_shards, num_slices=num_slices, shard_index=shard_index)
    # 加载分片数据
    shard_data = np.load(train_npy_file, allow_pickle=True).item()
    
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
    print("disabled_trj_ids:",disabled_trj_ids)
    print("slices:",slices)
    affected_slices = [i for i, slice_data in enumerate(slices, start=1) if any(trj_id in slice_data for trj_id in disabled_trj_ids)]
    
    if not affected_slices:
        print("No disabled trj_ids found in any slices.")
        return None, slices
    return affected_slices, slices

# 重新训练禁用 trj_id 影响的 slice
def retrain_after_disable(npy_file, slices, source_data, disabled_trj_ids, affected_slices, shard_index, num_shards, num_slices, epoch_shard_slice_training, batch_size, input_size, output_size, device, intermediate_model_save_path, final_model_save_path):
    model = LSTMModel(input_size, output_size).to(device)
    start_slice = min(affected_slices)
    
    # 加载最接近未受影响的 slice 模型
    if start_slice > 1:
        model_path = f"{intermediate_model_save_path}/model_shard-idx_{shard_index}_shard-num_{num_shards}_slice_{start_slice-1}_{num_slices}.pth"
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded unaffected model: {model_path}")
    else:
        print("Disabled data in the first slice, retraining all slices.")

    # 更新 slices 去除禁用 trj_id
    for idx in affected_slices:
        slices[idx - 1] = [trj_id for trj_id in slices[idx - 1] if trj_id not in disabled_trj_ids]

    # 从受影响的 slice 开始重新训练模型
    for i in range(start_slice, num_slices + 1):
        slice_data = source_data[source_data['trj_id_set'].isin(slices[i - 1])].copy()
        dataloader = DataLoader(MyDataset(slice_data), batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=True)

        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.97)
        
        for epoch in range(epoch_shard_slice_training * i):
            model.train()
            running_loss = 0.0
            for x_batch, y_batch, lengths in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_batch = y_batch.view(-1, output_size)

                outputs = model(x_batch, lengths)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()
            avg_loss = running_loss / len(dataloader)
            if (epoch + 1) % 10 == 0 or epoch == epoch_shard_slice_training * i - 1:
                print(f"Shard {shard_index}, Slice {i}, Epoch {epoch + 1} completed, Avg Loss: {avg_loss:.4f}")

        # 保存模型
        model_path = f"{intermediate_model_save_path}/retrained_model_shard-idx_{shard_index}_shard-num_{num_shards}_slice_{i}_{num_slices}_disable_id_{str(disabled_trj_ids[0])}_.pth"
        torch.save(model.state_dict(), model_path)
        if i == num_slices:
            final_model_path = f"{final_model_save_path}/final_model_A_shard-idx_{shard_index}_shard-num_{num_shards}_after_disable.pth"
            torch.save(model.state_dict(), final_model_path)
            print(f"Final retrained model for Shard {shard_index} saved to {final_model_path}")

    return model

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_index', type=int, help="需要处理shard的索引")
    parser.add_argument('--config', type=str, default='./config-toy.yaml', help="配置文件路径")
    parser.add_argument('--disabled id config', type=str, default='./config-toy.yaml', help="需要删除的id路径")
    args = parser.parse_args()
    config = load_config(args.config)
    if config is None:
        print("Config loading failed.")
        return

    disabled_trj_ids = load_disabled_trj_ids('./disabled_trj_ids.yaml')
    if disabled_trj_ids is None:
        print("Failed to load disabled trj_id values.")
        return


    source_data = pd.read_csv(config['source_file'])
    num_shards = config['num_shards']
    num_slices = config['num_slices']
    save_path = config['output_data_path']
    folder_name = f"Model_shard_{num_shards}_slice_{num_slices}"
    full_path = os.path.join(save_path, folder_name)
    print("trying to retrain model A, shard index:", int(args.shard_index))
    # 查找禁用 ID 是否影响当前 shard 的 slices
    affected_slices, slices = find_slices_for_disabled_ids(
        train_npy_file=full_path + '/npy_index_file/training_shard_random_assignment.npy',
        num_shards=config['num_shards'],
        num_slices=config['num_slices'],
        shard_index= int(args.shard_index),
        disabled_trj_ids=disabled_trj_ids
    )
    
    if affected_slices is None:
        print("No slices affected, exiting.")
        return  # No affected slices, exit

    # 重新训练受影响的 slices
    retrain_after_disable(
        npy_file=full_path + '/npy_index_file/training_shard_random_assignment.npy',
        slices=slices,
        source_data=source_data,
        disabled_trj_ids=disabled_trj_ids,
        affected_slices=affected_slices,
        shard_index=args.shard_index,
        num_shards=config['num_shards'],
        num_slices=config['num_slices'],
        epoch_shard_slice_training=config['SISA_Lstm_training_epoch'],
        batch_size=config['SISA_Lstm_training_batch_size'],
        input_size=config['input_size'],
        output_size=config['output_size'],
        device=config['device'],
        intermediate_model_save_path=full_path +'/3_1_SISA_Random_lstm/SISA_sub_model',
        final_model_save_path=full_path +'/3_1_SISA_Random_lstm/SISA_final_model'
    )

if __name__ == "__main__":
    main()