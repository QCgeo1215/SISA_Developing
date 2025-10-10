import numpy as np
import torch
import pandas as pd
from datetime import datetime
from lstm_model import LSTMModel
from my_dataset import MyDataset
from collate_functions import pad_collate_fn
from manhattan_distance import is_prediction_in_threshold_range
from torch.utils.data import DataLoader
import yaml
import sys
import argparse
import os

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

def calculate_accuracy(aggregated_predictions, true_values, thresholds):
    results = {threshold: {"correct": 0, "total": len(true_values)} for threshold in thresholds}
    for i in range(len(true_values)):
        true_value, pred_value = true_values[i], aggregated_predictions[i]
        for threshold in thresholds:
            if is_prediction_in_threshold_range(pred_value, true_value, threshold):
                results[threshold]["correct"] += 1
    return results

def save_results(results, output_file):
    with open(output_file, "w") as f:
        for threshold, data in results.items():
            correct_predictions, total_samples = data["correct"], data["total"]
            accuracy = correct_predictions / total_samples * 100 if total_samples > 0 else 0
            f.write(f"Threshold {threshold}:\nCorrect predictions: {correct_predictions}/{total_samples}\nAccuracy: {accuracy:.2f}%\n\n")
    print(f"Results saved to {output_file}")

def aggregate_predictions(test_npy_file, csv_path, final_models_A, final_models_B, batch_size, output_size, device, num_shards, num_slices, result_path, start_time):  
    test_trj_id_set = np.load(test_npy_file)
    source_data = pd.read_csv(csv_path)
    test_data = source_data[source_data['trj_id_set'].isin(test_trj_id_set)].copy()
    print("len(test_data)",len(test_data))
    test_dataset = MyDataset(test_data)
    set_global_seed(3407)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=False)
    all_reliabilities, all_predictions_A, true_values = [], [], []
    true_values_collected = False
    for shard_index, (model_A, model_B) in enumerate(zip(final_models_A, final_models_B), start=1):
        print(f"Predicting with shard model and Loss model from Shard {shard_index}")
        model_A.eval()
        model_B.eval()
        all_reliabilities_shard, all_predictions_A_shard = [], []
        # Collect true_values only once
        with torch.no_grad():
            #for x_batch, y_batch, lengths in test_dataloader:
            for x_batch, y_batch, lengths, first_pos_x, last_pos_x in test_dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.view(-1, output_size)
                last_pos_x = last_pos_x.to(device)
                y_pred_B = model_B(x_batch, lengths,last_pos_x)
                y_pred_A = model_A(x_batch, lengths,last_pos_x)
                all_reliabilities_shard.append(y_pred_B.cpu().numpy())
                all_predictions_A_shard.append(y_pred_A.cpu().numpy())
                # Collect true_values only once
                if not true_values_collected:
                    true_values.append(y_batch.cpu().numpy())
        all_reliabilities.append(np.concatenate(all_reliabilities_shard, axis=0))
        all_predictions_A.append(np.concatenate(all_predictions_A_shard, axis=0))
        true_values_collected = True  # Prevent further appends to true_values
    all_reliabilities, all_predictions_A = np.array(all_reliabilities), np.array(all_predictions_A)
    true_values = np.concatenate(true_values, axis=0)
    print("true_values.shape",true_values.shape)
    # Method 1: Weighted average based on model_B's predictions (existing approach)
    Z_prime = 1 / all_reliabilities
    weights = Z_prime / Z_prime.sum(axis=0, keepdims=True)
    print("weights.shape",weights.shape) 
    aggregated_predictions_weighted = np.sum(weights * all_predictions_A, axis=0)
    print("aggregated_predictions_weighted.shape",aggregated_predictions_weighted.shape) 
    # Method 2: Simple average of all model_A predictions
    aggregated_predictions_average = np.mean(all_predictions_A, axis=0)
    # Calculate accuracy for both methods
    thresholds = [1,2,3,4,5]
    results_weighted = calculate_accuracy(aggregated_predictions_weighted, true_values, thresholds)
    results_average = calculate_accuracy(aggregated_predictions_average, true_values, thresholds)
    # Save results for both methods
    output_file_weighted = f"{result_path}/weighted_aggregate_prediction_results_shards{num_shards}_slice_{num_slices}_{start_time}.txt"
    output_file_average = f"{result_path}/average_aggregate_prediction_results_shards{num_shards}_slice_{num_slices}_{start_time}.txt"
    save_results(results_weighted, output_file_weighted)
    save_results(results_average, output_file_average)
    return aggregated_predictions_weighted, aggregated_predictions_average

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config-scaling.yaml', help="配置文件路径")
    args = parser.parse_args()
    config = load_config(args.config)
    if config is None:
        print("Config loading failed.")
        return
    num_shards = config['num_shards']
    num_slices = config['num_slices']
    save_path = config['output_data_path']
    folder_name = f"Model_shard_{num_shards}_slice_{num_slices}"
    full_path = os.path.join(save_path, folder_name)

    final_models_A, final_models_B = [], []
    for i in range(config['num_shards']):
        model_A = LSTMModel(input_size=2, output_size=config['output_size']).to(config['device'])
        model_B = LSTMModel(input_size=2, output_size=config['output_size']).to(config['device'])
        model_A.load_state_dict(torch.load(f"{full_path +'/3_1_SISA_Random_lstm/SISA_final_model'}/final_model_shard-idx_{i+1}_shard-num_{config['num_shards']}.pth"))
        model_B.load_state_dict(torch.load(f"{full_path +'/3_1_SISA_Random_lstm/shard_loss_model'}/final_loss_model_shard-idx_{i+1}_shard-num_{config['num_shards']}.pth"))
        # model_A.load_state_dict(torch.load(f"{full_path +'/3_1_SISA_Random_lstm/SISA_final_model'}/final_model_shard-idx_1_shard-num_5.pth"))
        # model_B.load_state_dict(torch.load(f"{full_path +'/3_1_SISA_Random_lstm/shard_loss_model'}/final_loss_model_shard-idx_1_shard-num_5.pth"))
        final_models_A.append(model_A)
        final_models_B.append(model_B)

    aggregate_predictions(
        test_npy_file=full_path +'/npy_index_file/test_trj_id_sets.npy',
        csv_path=config['source_file'],
        final_models_A=final_models_A,
        final_models_B=final_models_B,
        batch_size=config['SISA_Lstm_training_batch_size'],
        output_size=config['output_size'],
        device=config['device'],
        num_shards=config['num_shards'],
        num_slices=config['num_slices'],
        result_path=full_path +'/3_1_SISA_Random_lstm/aggreate_result',
        start_time=datetime.now().strftime("%Y%m%d_%H%M%S")
    )

if __name__ == "__main__":
    main()
