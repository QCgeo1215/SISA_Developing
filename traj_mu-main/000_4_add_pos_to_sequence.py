import pandas as pd
import numpy as np
import ast

# 读取数据（注意确保 lat_grid 和 lng_grid 是列表格式，如果是字符串形式的列表，要先解析）
df_main = pd.read_csv('./taxi_data_Singapore/4-Slidng_Windows_Data_Clustering_Clean-sta-Cluster-Results-20250711.csv')

# 如果列是字符串形式的列表，需要先转为真正的列表
df_main['lat_grid'] = df_main['lat_grid'].apply(ast.literal_eval)
df_main['lng_grid'] = df_main['lng_grid'].apply(ast.literal_eval)

# 1. 统计 max 和 min
lat_all_values = [item for sublist in df_main['lat_grid'] for item in sublist]
lng_all_values = [item for sublist in df_main['lng_grid'] for item in sublist]

lat_min, lat_max = min(lat_all_values), max(lat_all_values)
lng_min, lng_max = min(lng_all_values), max(lng_all_values)

print(f"lat_grid 范围: min = {lat_min}, max = {lat_max}")
print(f"lng_grid 范围: min = {lng_min}, max = {lng_max}")

# 2. 添加新的字段：归一化首值
def extract_norm_first_val(grid_list):
    return (grid_list[0] - 1) / (1001 - 1)

df_main['lat_grid_first_norm'] = df_main['lat_grid'].apply(extract_norm_first_val)
df_main['lng_grid_first_norm'] = df_main['lng_grid'].apply(extract_norm_first_val)

# 显示前几行看看
print(df_main[['lat_grid_first_norm', 'lng_grid_first_norm']].head(4))

# 保存回原 CSV（推荐备份原始数据）
df_main.to_csv('./taxi_data_Singapore/4-Slidng_Windows_Data_Clustering_Clean-sta-Cluster-Results-20250711_with_norm.csv', index=False)