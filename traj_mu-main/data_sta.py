import ast
import pandas as pd

# class DataStandardizer:
#     def __init__(self, cumulative_data):
#         """
#         初始化函数，接收原始数据 cumulative_data。
#         :param cumulative_data: pandas DataFrame，包含原始数据。
#         """
#         self.cumulative_data = cumulative_data

#     def standardize_data(self):
#         """
#         对数据进行标准化处理。
#         :return: 返回标准化后的 DataFrame。
#         """
#         # 提取 lat_grid 和 lng_grid 并转换为列表形式
#         lat_lng_data = self.cumulative_data[['lat_grid', 'lng_grid']].applymap(ast.literal_eval)

#         # 初始化新的列
#         training_sequence_origin_final_loc = []  # 存储每个样本的最后一个 lat 和 lng
#         lat_grid_sta = []            # 存储标准化后的 lat 序列
#         lng_grid_sta = []            # 存储标准化后的 lng 序列
#         final_grid_label_sta = []    # 存储标准化后的 final_grid_label

#         # 遍历每一行数据
#         for index, row in lat_lng_data.iterrows():
#             # 提取 lat 和 lng 序列
#             lat_seq = row['lat_grid']
#             lng_seq = row['lng_grid']

#             # (1) 提取最后一个 lat 和 lng，保存为 training_sequence_origin_final_loc
#             training_sequence_origin_final_loc.append([lat_seq[-1], lng_seq[-1]])

#             # (2) 对 lat 和 lng 序列进行标准化
#             lat_first = lat_seq[0]
#             lng_first = lng_seq[0]
#             lat_seq_sta = [lat - lat_first for lat in lat_seq]
#             lng_seq_sta = [lng - lng_first for lng in lng_seq]
#             lat_grid_sta.append(lat_seq_sta)
#             lng_grid_sta.append(lng_seq_sta)

#             # (3) 对 final_grid_label 进行标准化
#             final_grid_label = ast.literal_eval(self.cumulative_data.loc[index, 'final_grid_label'])
#             final_grid_label_sta.append([final_grid_label[0] - lat_first, final_grid_label[1] - lng_first])

#         # 将处理后的数据添加到新的 DataFrame 中
#         self.cumulative_data_sta = self.cumulative_data.copy()
#         self.cumulative_data_sta['training_sequence_origin_final_loc'] = training_sequence_origin_final_loc
#         self.cumulative_data_sta['lat_grid_sta'] = lat_grid_sta
#         self.cumulative_data_sta['lng_grid_sta'] = lng_grid_sta
#         self.cumulative_data_sta['final_grid_label_sta'] = final_grid_label_sta

#         return self.cumulative_data_sta

    # def save_to_csv(self, file_path):
    #     """
    #     将标准化后的数据保存为 CSV 文件。
    #     :param file_path: 保存文件的路径。
    #     """
    #     if not hasattr(self, 'cumulative_data_sta'):
    #         raise ValueError("请先调用 standardize_data 方法生成标准化数据。")
    #     self.cumulative_data_sta.to_csv(file_path, index=False)


class DataStandardizer:
    def __init__(self, cumulative_data):
        """
        初始化函数，接收原始数据 cumulative_data。
        :param cumulative_data: pandas DataFrame，包含原始数据。
        """
        self.cumulative_data = cumulative_data

    def safe_literal_eval(self, x):
        """
        安全地解析字符串为 Python 对象。
        :param x: 输入的字符串或对象。
        :return: 解析后的 Python 对象。
        """
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError) as e:
                print(f"解析错误：{e}，数据：{x}")
                raise ValueError(f"无法解析的数据：{x}")
        return x

    def standardize_data(self):
        """
        对数据进行标准化处理。
        :return: 返回标准化后的 DataFrame。
        """
        # 提取 lat_grid 和 lng_grid 并转换为列表形式
        lat_lng_data = self.cumulative_data[['lat_grid', 'lng_grid']].applymap(self.safe_literal_eval)

        # 初始化新的列
#         training_sequence_origin_final_loc = []  # 存储每个样本的最后一个 lat 和 lng
#         lat_grid_sta = []                        # 存储标准化后的 lat 序列
#         lng_grid_sta = []                        # 存储标准化后的 lng 序列
#         final_grid_label_sta = []                # 存储标准化后的 final_grid_label

#         # 遍历每一行数据
#         for index, row in lat_lng_data.iterrows():
#             # 提取 lat 和 lng 序列
#             lat_seq = row['lat_grid']
#             lng_seq = row['lng_grid']
            
#             # 确保 lat_seq 和 lng_seq 是列表
#             if not isinstance(lat_seq, (list, tuple)) or not isinstance(lng_seq, (list, tuple)):
#                 raise ValueError(f"lat_grid 或 lng_grid 不是列表或元组：lat_seq={lat_seq}, lng_seq={lng_seq}")

#             # (1) 提取最后一个 lat 和 lng，保存为 training_sequence_origin_final_loc
#             training_sequence_origin_final_loc.append([lat_seq[-1], lng_seq[-1]])

#             # (2) 对 lat 和 lng 序列进行标准化
#             lat_first = lat_seq[0]
#             lng_first = lng_seq[0]
#             lat_seq_sta = [lat - lat_first for lat in lat_seq]
#             lng_seq_sta = [lng - lng_first for lng in lng_seq]
#             lat_grid_sta.append(lat_seq_sta)
#             lng_grid_sta.append(lng_seq_sta)

#             # (3) 对 final_grid_label 进行标准化
#             final_grid_label = self.cumulative_data.loc[index, 'final_grid_label']
#             final_grid_label = self.safe_literal_eval(final_grid_label)
#             #print(final_grid_label[0])
#             #print(type(final_grid_label[0]))
#             final_grid_label_sta.append([final_grid_label[0] - lat_first, final_grid_label[1] - lng_first])
            # if not isinstance(final_grid_label, (list, tuple)) or len(final_grid_label) != 2:
            #     raise ValueError(f"final_grid_label 不是长度为 2 的列表或元组：{final_grid_label}")
            # final_grid_label_sta.append([final_grid_label[0] - lat_first, final_grid_label[1] - lng_first])
        #print("finish all")
        # 将处理后的数据添加到新的 DataFrame 中
        self.cumulative_data_sta = self.cumulative_data.copy()
        #self.cumulative_data_sta['training_sequence_origin_final_loc'] = training_sequence_origin_final_loc
        #self.cumulative_data_sta['lat_grid_sta'] = lat_grid_sta
        #EWself.cumulative_data_sta['lng_grid_sta'] = lng_grid_sta
        #self.cumulative_data_sta['final_grid_label_sta'] = final_grid_label_sta
        #print("finish all 2")
        return self.cumulative_data_sta