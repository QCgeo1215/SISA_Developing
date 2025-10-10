# collate_functions.py

# import numpy as np
# import torch

# def pad_collate_fn(batch):
#     padded_x_data = list()
#     first_pos_x = list()  # first location
#     last_pos_x = list()   # last location

#     x_data, y_data = zip(*batch)
#     y_data = np.array(y_data)
#     y_data = y_data.tolist()
#     y_data = np.array(y_data)

#     lengths = [len(x[0]) for x in x_data]  # Length of the original sequence
#     max_len = max(lengths)

#     # 用于存储有效的 y_data 索引
#     valid_indices = []

#     # 记录跳过的样本数量
#     skipped_count = 0

#     for i, x in enumerate(x_data):
#         padded_sample = list()
#         x = list(x)
#         x = np.array(x)

#         # 检查 x 是否为空
#         if x.size == 0:
#             skipped_count += 1
#             continue  # 跳过空数据

#         # 记录有效样本的索引
#         valid_indices.append(i)

#         x = x.T
#         first_pos_x.append(x[0])
#         last_pos_x.append(x[-1])
#         x = x - x[0]

#         # 对 x 进行填充
#         for feature in x.T:  # Transpose to pad each feature
#             padding_length = max_len - len(feature)
#             padded_feature = list(feature) + [0] * padding_length
#             padded_sample.append(padded_feature)

#         padded_x_data.append(padded_sample)

#     # 如果跳过了样本，打印跳过信息
#     if skipped_count > 0:
#         print(f"跳过了 {skipped_count} 个空数据样本")

#     # 仅保留有效的 y_data
#     y_data = y_data[valid_indices]
#     padded_x_data = np.array(padded_x_data)
#     first_pos_x = np.array(first_pos_x)
#     last_pos_x = np.array(last_pos_x)
    
#     padded_x_data = torch.tensor(padded_x_data, dtype=torch.float32).permute(0, 2, 1)  # Adjust shape to (batch_size, seq_len, input_size)
#     y_data = torch.tensor(y_data, dtype=torch.float32)
#     first_pos_x = torch.tensor(first_pos_x, dtype=torch.float32)
#     first_pos_x = first_pos_x.unsqueeze(dim=1)
#     y_data = y_data - first_pos_x
#     last_pos_x = torch.tensor(last_pos_x, dtype=torch.float32)
    
#     return padded_x_data, y_data, lengths, first_pos_x, last_pos_x


import numpy as np
import torch

def pad_collate_fn(batch):
    padded_x_data = []
    first_pos_x = []
    last_pos_x = []
    
    x_data, y_data = zip(*batch)
    y_data = np.array(y_data).tolist()
    y_data = np.array(y_data)

    lengths = [len(x[0]) for x in x_data]
    max_len = max(lengths)

    valid_indices = []
    skipped_count = 0

    for i, x in enumerate(x_data):
        x = np.array(x)
        if x.size == 0:
            skipped_count += 1
            continue

        valid_indices.append(i)
        x = x.T  # (seq_len, input_size)

        # === 新处理逻辑 ===
        norm_val = x[0]              # shape: (input_size,)
        x_rest = x[1:]               # shape: (seq_len - 1, input_size)
        first_pos_x.append(x_rest[0])
        last_pos_x.append(x_rest[-1])
        x_rest = x_rest - x_rest[0]  # 相对位置
        x_new = np.vstack([norm_val, x_rest])  # 拼接归一化首位

        # === padding ===
        padded_sample = []
        for feature in x_new.T:
            padding_len = max_len - len(feature)
            padded_feature = list(feature) + [0] * padding_len
            padded_sample.append(padded_feature)

        padded_x_data.append(padded_sample)

    if skipped_count > 0:
        print(f"跳过了 {skipped_count} 个空数据样本")

    y_data = y_data[valid_indices]
    padded_x_data = np.array(padded_x_data)
    first_pos_x = np.array(first_pos_x)
    last_pos_x = np.array(last_pos_x)
    
        # ✅ 打印 padded_x_data 前5个样本的前5行
    print("=== 前 5 个 padded_x_data 的首5行序列（维度：时间步 × 特征） ===")
    for i in range(min(5, len(padded_x_data))):
        print(f"样本 {i}:")
        sample_array = padded_x_data[i]  # shape: (input_size, seq_len)
        sample_array = sample_array.T    # 转换为 (seq_len, input_size)
        print(sample_array[:5])          # 打印前5个时间步

    padded_x_data = torch.tensor(padded_x_data, dtype=torch.float32).permute(0, 2, 1)  # (B, T, D)
    y_data = torch.tensor(y_data, dtype=torch.float32)
    first_pos_x = torch.tensor(first_pos_x, dtype=torch.float32).unsqueeze(1)
    last_pos_x = torch.tensor(last_pos_x, dtype=torch.float32)

    y_data = y_data - first_pos_x

    return padded_x_data, y_data, lengths, first_pos_x, last_pos_x



# def pad_collate_fn_V2(batch):
#     padded_x_data = list()
#     first_pos_x = list()  # first location
#     last_pos_x = list()   # last location

#     x_data, y_data = zip(*batch)
#     y_data = np.array(y_data)
#     y_data = y_data.tolist()
#     y_data = np.array(y_data)

#     lengths = [len(x[0]) for x in x_data]  # Length of the original sequence
#     max_len = max(lengths)

#     # 用于存储有效的 y_data 索引
#     valid_indices = []

#     # 记录跳过的样本数量
#     skipped_count = 0

#     for i, x in enumerate(x_data):
#         padded_sample = list()
#         x = list(x)
#         x = np.array(x)

#         # 检查 x 是否为空
#         if x.size == 0:
#             skipped_count += 1
#             continue  # 跳过空数据

#         # 记录有效样本的索引
#         valid_indices.append(i)

#         x = x.T
#         first_pos_x.append(x[0])
#         last_pos_x.append(x[-1])
#         x = x - x[0]

#         # 对 x 进行填充
#         for feature in x.T:  # Transpose to pad each feature
#             padding_length = max_len - len(feature)
#             padded_feature = list(feature) + [0] * padding_length
#             padded_sample.append(padded_feature)

#         padded_x_data.append(padded_sample)

#     # 如果跳过了样本，打印跳过信息
#     if skipped_count > 0:
#         print(f"跳过了 {skipped_count} 个空数据样本")

#     # 仅保留有效的 y_data
#     y_data = y_data[valid_indices]
#     padded_x_data = np.array(padded_x_data)
#     first_pos_x = np.array(first_pos_x)
#     last_pos_x = np.array(last_pos_x)
    
#     padded_x_data = torch.tensor(padded_x_data, dtype=torch.float32).permute(0, 2, 1)  # Adjust shape to (batch_size, seq_len, input_size)
#     y_data = torch.tensor(y_data, dtype=torch.float32)
#     first_pos_x = torch.tensor(first_pos_x, dtype=torch.float32)
#     first_pos_x = first_pos_x.unsqueeze(dim=1)
#     #y_data = y_data - first_pos_x
#     last_pos_x = torch.tensor(last_pos_x, dtype=torch.float32)
    
#     return padded_x_data, y_data, lengths, first_pos_x, last_pos_x

import numpy as np
import torch

def pad_collate_fn(batch):
    padded_x_data = list()
    first_pos_x = list()  # first location
    last_pos_x = list()   # last location

    x_data, y_data = zip(*batch)
    y_data = np.array(y_data)
    y_data = y_data.tolist()
    y_data = np.array(y_data)

    lengths = [len(x[0]) for x in x_data]  # Length of the original sequence
    max_len = max(lengths)

    # 用于存储有效的 y_data 索引
    valid_indices = []

    # 记录跳过的样本数量
    skipped_count = 0

    for i, x in enumerate(x_data):
        padded_sample = list()
        x = list(x)
        x = np.array(x)

        # 检查 x 是否为空
        if x.size == 0:
            skipped_count += 1
            continue  # 跳过空数据

        # 记录有效样本的索引
        valid_indices.append(i)

        x = x.T  # (seq_len, input_size)

        # ===== ✅ 插入保留归一化首值的处理 =====
        norm_val = x[0]            # shape: (input_size,)
        x_rest = x[1:]             # shape: (seq_len - 1, input_size)
        first_pos_x.append(x_rest[0])
        last_pos_x.append(x_rest[-1])
        x_rest = x_rest - x_rest[0]
        x_new = np.vstack([norm_val, x_rest])  # 拼接归一化首值在第一行
        # =====================================

        # 对 x_new 进行填充
        for feature in x_new.T:  # Transpose to pad each feature
            padding_length = max_len - len(feature) + 1  # 注意加1是因为我们删掉了1行又加回来1行
            padded_feature = list(feature) + [0] * padding_length
            padded_sample.append(padded_feature)

        padded_x_data.append(padded_sample)

    # 如果跳过了样本，打印跳过信息
    if skipped_count > 0:
        print(f"跳过了 {skipped_count} 个空数据样本")

    # 仅保留有效的 y_data
    y_data = y_data[valid_indices]
    padded_x_data = np.array(padded_x_data)
    first_pos_x = np.array(first_pos_x)
    last_pos_x = np.array(last_pos_x)
    
    # ✅ 打印调试：输出 padded_x_data 前 5 个样本的前 5 个时间步
    print("=== 前 5 个 padded_x_data 的首5行序列（V2） ===")
    for i in range(min(5, len(padded_x_data))):
        print(f"样本 {i}:")
        sample_array = np.array(padded_x_data[i]).T  # 转成 (seq_len, input_size)
        print(sample_array[:5])

    # === 转 tensor ===
    padded_x_data = torch.tensor(padded_x_data, dtype=torch.float32).permute(0, 2, 1)  # (batch_size, seq_len, input_size)
    y_data = torch.tensor(y_data, dtype=torch.float32)
    first_pos_x = torch.tensor(first_pos_x, dtype=torch.float32)
    first_pos_x = first_pos_x.unsqueeze(dim=1)
    y_data = y_data - first_pos_x
    last_pos_x = torch.tensor(last_pos_x, dtype=torch.float32)

    return padded_x_data, y_data, lengths, first_pos_x, last_pos_x