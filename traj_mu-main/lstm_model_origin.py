# lstm_model.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, num_layers=4, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向LSTM输出的hidden_size是原来的两倍
        #self.fc = nn.Linear(hidden_size * 2 + 2, output_size)  # 双向LSTM输出的hidden_size是原来的两倍
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.init_weights()

    def init_weights(self):
        # 初始化LSTM权重

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # 初始化全连接层权重
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x, lengths):
        x = x.flip(1)
        packed_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(packed_out[:, -1, :])
        return out

#     def forward(self, x, lengths, training_sequence_origin_final_loc):
#         """
#         :param x: 输入数据，形状为 (batch_size, sequence_length, input_size)
#         :param lengths: 每个样本的实际长度（未使用）
#         :param training_sequence_origin_final_loc: 每个样本的最后一个 lat 和 lng，形状为 (batch_size, 2)
#         :return: 输出数据，形状为 (batch_size, output_size)
#         """
#         # 将 training_sequence_origin_final_loc 扩展到每个时间步
#         training_sequence_origin_final_loc = training_sequence_origin_final_loc.unsqueeze(1).expand(-1, x.size(1), -1)  # 形状为 (batch_size, sequence_length, 2)
        
#         # 将额外信息拼接到输入数据上
#         x = torch.cat([x, training_sequence_origin_final_loc], dim=2)  # 形状为 (batch_size, sequence_length, input_size + 2)
        
#         # LSTM 前向传播
#         x = x.flip(1)
#         packed_out, (h_n, c_n) = self.lstm(x)
        
#         # 取 LSTM 的最后一个时间步的输出
#         lstm_out = packed_out[:, -1, :]  # 形状为 (batch_size, hidden_size * 2)
        
#         # 通过全连接层
#         out = self.fc(lstm_out)  # 形状为 (batch_size, output_size)
#         return out