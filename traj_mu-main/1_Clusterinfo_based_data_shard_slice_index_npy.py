import numpy as np
import pandas as pd
import yaml
import torch
import random
import sys
import argparse
import os

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None
    
def load_and_preprocess_data(csv_file, num_shards, test_ratio=0.1):
    # 读取CSV文件
    data = pd.read_csv(csv_file)
    
    # 获取唯一的 trj_id_set
    all_trj_id_set_unique = data['trj_id_set'].unique()

    # 划分为训练数据和测试数据
    split_index = int(len(all_trj_id_set_unique) * (1 - test_ratio))
    train_trj_id_set = all_trj_id_set_unique[:split_index]
    test_trj_id_set = all_trj_id_set_unique[split_index:]

    # 筛选训练数据
    train_data = data[data['trj_id_set'].isin(train_trj_id_set)]
    return train_data, train_trj_id_set, test_trj_id_set

def compute_cluster_counts(train_data):
    # 按 kmeans_cluster 统计每个类别的样本数
    cluster_counts = train_data.groupby('kmeans_cluster').size().to_dict()
    return cluster_counts

def sort_trj_id_sets(train_data, cluster_counts):
    # 获取每个 trj_id_set 的 cluster 分布
    trj_id_set_cluster_counts = train_data.groupby(['trj_id_set', 'kmeans_cluster']).size().unstack(fill_value=0)

    # 定义 cluster 排序优先级
    cluster_priority = sorted(cluster_counts.keys(), key=lambda c: cluster_counts[c], reverse=True)

    # 排序 trj_id_set
    sorted_trj_id_sets = sorted(
        trj_id_set_cluster_counts.index,
        key=lambda x: tuple(
            -trj_id_set_cluster_counts.loc[x, cluster] if cluster in trj_id_set_cluster_counts.columns else 0
            for cluster in cluster_priority
        )
    )
    return sorted_trj_id_sets, trj_id_set_cluster_counts


def assign_shards(sorted_trj_id_sets, trj_id_set_cluster_counts, num_shards):
    """
    将 trj_id_set 按 cluster 统计进行 shard 分配，并增加随机分配方案

    :param sorted_trj_id_sets: 按 cluster 排序后的 trj_id_set 列表
    :param trj_id_set_cluster_counts: 每个 trj_id_set 的 cluster 统计
    :param num_shards: 需要分配的 shard 数量
    :return: (基于 cluster 统计的分配方案, 随机分配方案, 各 shard 的 cluster 统计)
    """

    # **初始化 shard 分配（基于 cluster 排序）**
    shard_cluster_assignments = {shard + 1: [] for shard in range(num_shards)}  # 从 1 开始
    shard_cluster_counts = {shard + 1: {cluster: 0 for cluster in trj_id_set_cluster_counts.columns} for shard in range(num_shards)}

    # **初始化 shard 分配（随机）**
    shard_random_assignment = {shard + 1: [] for shard in range(num_shards)}
    shard_random_counts = {shard + 1: {cluster: 0 for cluster in trj_id_set_cluster_counts.columns} for shard in range(num_shards)}

    # **计算 cluster 排序分配**
    for i, trj_id_set in enumerate(sorted_trj_id_sets):
        shard_index = (i % num_shards) + 1  # 从 1 开始
        shard_cluster_assignments[shard_index].append(trj_id_set)

        # 更新 shard 中的 cluster 统计
        for cluster, count in trj_id_set_cluster_counts.loc[trj_id_set].items():
            shard_cluster_counts[shard_index][cluster] += count

    # **生成随机分配方案**
    random_seed=4
    random.seed(random_seed) 
    random_trj_id_sets = sorted_trj_id_sets.copy()
    random.shuffle(random_trj_id_sets)  # **随机打乱 trj_id_set**
    
    for i, trj_id_set in enumerate(random_trj_id_sets):
        shard_index = (i % num_shards) + 1  # 从 1 开始
        shard_random_assignment[shard_index].append(trj_id_set)

        # 更新随机 shard 中的 cluster 统计
        for cluster, count in trj_id_set_cluster_counts.loc[trj_id_set].items():
            shard_random_counts[shard_index][cluster] += count

    return shard_cluster_assignments, shard_random_assignment, shard_cluster_counts, shard_random_counts


def shard_slice_data(train_npy_file, num_shards, num_slices, shard_index):
    # 加载分片数据
    shard_data = np.load(train_npy_file, allow_pickle=True).item()
    trj_id_sets = shard_data["shards"].get(shard_index, [])

    # 根据切片数量划分数据
    slice_size = max(len(trj_id_sets) // num_slices, 1)
    slices = [
        trj_id_sets[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)
    ]

    return slices

def save_npy_files(shard_cluster_assignments,shard_random_assignment, train_trj_id_set,test_trj_id_set, train_npy_cluster_shard_save_path,train_npy_random_shard_save_path,train_npy_save_path, test_npy_save_path):
    # 保存 shard 分配结果
    np.save(train_npy_cluster_shard_save_path, {"shards": shard_cluster_assignments})
    np.save(train_npy_random_shard_save_path, {"shards": shard_random_assignment})

    # 保存测试数据
    np.save(train_npy_save_path, train_trj_id_set)
    np.save(test_npy_save_path, test_trj_id_set)

    print(f"训练分片已保存至: {train_npy_cluster_shard_save_path} 和 {train_npy_random_shard_save_path}")
    print(f"测试数据已保存至: {test_npy_save_path}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config-scaling.yaml', help="配置文件路径")
    args = parser.parse_args()
    config = load_config(args.config)
    if config is None:
        print("Config loading failed.")
        return
    # 配置参数 
    csv_file = config['source_file']
    num_shards = config['num_shards']
    num_slices = config['num_slices']
    save_path = config['output_data_path']
    # 生成文件夹名称
    folder_name = f"Model_shard_{num_shards}_slice_{num_slices}"
    # 拼接完整路径
    full_path = os.path.join(save_path, folder_name)

    train_npy_cluster_shard_save_path = full_path +'/npy_index_file/training_shard_random_assignment.npy'
    #cluster_shard_save_path = config['train_npy_file_clustering_sort']
    train_npy_random_shard_save_path = full_path +'/npy_index_file/training_shard_cluster_assignment.npy'
    #test_save_path = config['test_npy_file']
    train_npy_save_path = full_path + '/npy_index_file/train_trj_id_sets.npy'
    test_npy_save_path = full_path +'/npy_index_file/test_trj_id_sets.npy'

    # 加载和预处理数据
    train_data, train_trj_id_set, test_trj_id_set = load_and_preprocess_data(csv_file, num_shards)

    print(f"训练数据中 trj_id_set 的数量: {len(train_trj_id_set)}")
    print(f"测试数据中 trj_id_set 的数量: {len(test_trj_id_set)}")

    # 统计 cluster 分布
    cluster_counts = compute_cluster_counts(train_data)
    print(f"Cluster 样本统计: {cluster_counts}")

    # 排序 trj_id_set
    sorted_trj_id_sets, trj_id_set_cluster_counts = sort_trj_id_sets(train_data, cluster_counts)
    print(f"排序后的 trj_id_set: {sorted_trj_id_sets[:10]} (仅显示前10个)")

    # 分配 shard
    shard_cluster_assignments, shard_random_assignment, shard_cluster_counts, shard_random_counts = assign_shards(sorted_trj_id_sets, trj_id_set_cluster_counts, num_shards)

    # 打印每个 shard 的 cluster 分布
    print("\n每个聚类排序 shard 的 cluster 样本数量：")
    for shard, cluster_count in shard_cluster_counts.items():
        print(f"Shard {shard}: {cluster_count}")
    # 打印每个 shard 的 cluster 分布
    print("\n每个随机分组 shard 的 cluster 样本数量：")
    for shard, cluster_count in shard_random_counts.items():
        print(f"Shard {shard}: {cluster_count}")

    # 保存分片和测试数据结果
    save_npy_files(shard_cluster_assignments,shard_random_assignment, train_trj_id_set,test_trj_id_set, train_npy_cluster_shard_save_path,train_npy_random_shard_save_path,train_npy_save_path, test_npy_save_path)


if __name__ == "__main__":
    main()
