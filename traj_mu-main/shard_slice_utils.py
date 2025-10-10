# shard_slice_utils.py
import numpy as np

def shard_slice_data(npy_file, num_shards, num_slices, shard_index):
    """
    根据指定的 shard 编号将数据切成 num_slices 块
    :param npy_file: npy 文件路径
    :param num_shards: shard 总数
    :param num_slices: 每个 shard 切分的 slice 数量
    :param shard_index: 当前 shard 的索引 (从 1 开始)
    :return: 指定 shard 的切片结果（切成 num_slices 块的数组列表）
    """
    # 加载 trj_id_set 的数据
    # 加载.npy文件
    trj_id_set = np.load(npy_file)
    shard_size = len(trj_id_set) // num_shards

    # 计算 shard 的起始和结束索引
    start_idx = (shard_index - 1) * shard_size
    end_idx = len(trj_id_set) if shard_index == num_shards else shard_index * shard_size
    shard_data = trj_id_set[start_idx:end_idx]
    print("shard_data:",shard_data)
    
    # 将 shard 数据切分成指定数量的 slices
    slice_size = len(shard_data) // num_slices
    slices = [shard_data[i * slice_size: (i + 1) * slice_size] for i in range(num_slices - 1)]
    slices.append(shard_data[(num_slices - 1) * slice_size:])
    
    print("slices:",slices)
    
    # 打印每个 slice 的信息
    for i, s in enumerate(slices, start=3):
        print(f"Shard {shard_index} - Slice {i} contains {len(s)} trj_set.")
        print(f"Slice {i} 前三个 trj_id_set 值: {s[:3]}")

    return slices