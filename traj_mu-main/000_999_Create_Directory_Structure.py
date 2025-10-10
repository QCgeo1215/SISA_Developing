import os
import yaml  # 如果你使用YAML格式的config文件
import torch
import argparse

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config-scaling.yaml', help="配置文件路径")
    args = parser.parse_args()
    config = load_config(args.config)
    if config is None:
        print("Config loading failed.")

    # 获取参数
    output_path = config['output_data_path']
    num_shards = config['num_shards']
    num_slices = config['num_slices']

    # 生成文件夹名称
    folder_name = f"Model_shard_{num_shards}_slice_{num_slices}"

    # 拼接完整路径
    full_path = os.path.join(output_path, folder_name)

    # 创建目录
    os.makedirs(full_path, exist_ok=True)

    print(f"Directory created at: {full_path}")

    # 在主目录下创建 "npy_index_file" 目录
    npy_index_path = os.path.join(full_path, "npy_index_file")
    os.makedirs(npy_index_path, exist_ok=True)

    # 在主目录下创建 "2_1_Direct_lstm" 目录
    npy_index_path = os.path.join(full_path, "2_1_Direct_lstm")
    os.makedirs(npy_index_path, exist_ok=True)

    # 定义需要在主目录下创建的子目录
    sub_folders = [
        "3_1_SISA_Random_lstm",
        "4_1_SISA_Sort_lstm"
    ]

    # 定义每个子目录下需要创建的二级目录
    sub_sub_folders = [
        "SISA_sub_model",
        "SISA_final_model",
        "shard_loss_model",
        "aggreate_result"
    ]

    # 创建子目录及其二级目录
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(full_path, sub_folder)
        os.makedirs(sub_folder_path, exist_ok=True)

        for sub_sub_folder in sub_sub_folders:
            sub_sub_folder_path = os.path.join(sub_folder_path, sub_sub_folder)
            os.makedirs(sub_sub_folder_path, exist_ok=True)

    print(f"Directory structure created at: {full_path}")

if __name__ == "__main__":
    main()