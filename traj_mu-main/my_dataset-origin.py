# my_dataset.py
import ast
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, df):
        self.x_data = df[['lat_grid', 'lng_grid']].applymap(ast.literal_eval).values # 'week_info', 'holiday_info', 'time_period_info',
        self.y_data = df[['final_grid_label']].applymap(ast.literal_eval).values
        self.length = len(self.y_data)
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]#[0]
    
    def __len__(self):
        return self.length
    
# class MyDataset(Dataset):
#     def __init__(self, df):
#         self.x_data = df[['lat_grid_sta', 'lng_grid_sta']].applymap(ast.literal_eval).values # 'week_info', 'holiday_info', 'time_period_info',
#         self.y_data = df[['final_grid_label_sta']].applymap(ast.literal_eval).values
#         self.length = len(self.y_data)
        
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]
    
#     def __len__(self):
#         return self.length