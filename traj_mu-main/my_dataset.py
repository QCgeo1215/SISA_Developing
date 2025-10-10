# my_dataset.py
import ast
import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, df):
        #self.x_data = df[['lat_grid', 'lng_grid']].applymap(eval).values # 'week_info', 'holiday_info', 'time_period_info',
        self.x_data = df[['lat_grid_norm  ', 'lng_grid_norm']].applymap(eval).values
        df['final_grid_label'] = df['final_grid_label'].apply(lambda x: eval(x))
        self.y_data = df[['final_grid_label']].values
        self.length = len(self.y_data)
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.length
    
    
    

# class MyDataset(Dataset):
#     def __init__(self, df):
#         """
#         初始化函数，接收 DataFrame 数据。
#         :param df: pandas DataFrame，包含标准化后的数据。
#         """
#         # 提取 lat_grid_sta 和 lng_grid_sta
#         self.x_data = df[['lat_grid_sta', 'lng_grid_sta']].applymap(ast.literal_eval).values
        
#         # 提取 final_grid_label_sta
#         self.y_data = df[['final_grid_label_sta']].applymap(ast.literal_eval).values
        
#         # 提取 training_sequence_origin_final_loc
#         self.training_sequence_origin_final_loc = df['training_sequence_origin_final_loc'].apply(ast.literal_eval).values
        
#         # 数据集长度
#         self.length = len(self.y_data)
        
#     def __getitem__(self, index):
#         """
#         返回一个样本的数据。
#         :param index: 样本索引。
#         :return: 包含 x_data、training_sequence_origin_final_loc 和 y_data 的元组。
#         """
#         # 获取 lat_grid_sta 和 lng_grid_sta
#         x = self.x_data[index]
        
#         # 获取 training_sequence_origin_final_loc
#         training_sequence_origin_final_loc = self.training_sequence_origin_final_loc[index]
        
#         # 获取 final_grid_label_sta
#         y = self.y_data[index]
        
#         # 返回 x、training_sequence_origin_final_loc 和 y
#         return x, training_sequence_origin_final_loc, y
    
#     def __len__(self):
#         """
#         返回数据集长度。
#         :return: 数据集长度。
#         """
#         return self.length