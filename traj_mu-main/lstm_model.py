# lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, num_layers=4, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2+2, output_size)  # 双向LSTM输出的hidden_size是原来的两倍
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
    
    def forward(self, x, lengths, last_pos_x=None):
        x = x.flip(1)
        packed_out, (h_n, c_n) = self.lstm(x)
        packed_out = F.tanh(packed_out)
        out = self.fc(torch.cat((packed_out[:, -1, :], last_pos_x), 1))
        #print("out",out)
        return out


# class LSTMModel(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size=512, num_layers=4, dropout=0.3):
#         super(LSTMModel, self).__init__()
#         # 添加BatchNorm层
#         self.input_bn = nn.BatchNorm1d(input_size)
        
#         # 修改LSTM结构
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
#                            batch_first=True, 
#                            dropout=dropout if num_layers > 1 else 0,  # 只有多层时才用dropout
#                            bidirectional=True)
        
#         # 添加注意力机制
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, 1, bias=False)
#         )
        
#         # 修改全连接层结构
#         self.fc = nn.Sequential(
#             nn.LayerNorm(hidden_size * 2 + 2),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size * 2 + 2, hidden_size),
#             nn.ReLU(),
#             nn.LayerNorm(hidden_size),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, output_size)
#         )
        
#         self.init_weights()

#     def init_weights(self):
#         # 更精细的初始化
#         for name, param in self.lstm.named_parameters():
#             if 'weight' in name:
#                 if 'ih' in name:
#                     nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='tanh')
#                 else:
#                     nn.init.orthogonal_(param.data)
#             elif 'bias' in name:
#                 nn.init.constant_(param[:param.shape[0]//4], 1)  # 遗忘门偏置初始化为1
#                 nn.init.constant_(param[param.shape[0]//4:], 0)
        
#         # 初始化注意力层
#         for layer in self.attention:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_normal_(layer.weight)
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, 0)
    
#     def forward(self, x, lengths, last_pos_x=None):
#         # 输入归一化
#         x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)
#         x = x.flip(1)
        
#         # LSTM处理
#         packed_out, (h_n, c_n) = self.lstm(x)
#         packed_out = F.tanh(packed_out)
        
#         # 注意力机制
#         attention_weights = F.softmax(self.attention(packed_out), dim=1)
#         context_vector = torch.sum(attention_weights * packed_out, dim=1)
        
#         # 拼接特征
#         out = self.fc(torch.cat((context_vector, last_pos_x), 1))
#         return out


# class LSTMModel(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size=512, num_layers=3, dropout=0.2):
#         super(LSTMModel, self).__init__()
#         # 输入处理
#         self.input_bn = nn.BatchNorm1d(input_size)
#         self.input_dropout = nn.Dropout(dropout/2)
        
#         # LSTM结构优化
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
#                            batch_first=True,
#                            dropout=dropout if num_layers > 1 else 0,
#                            bidirectional=True)
        
#         # 注意力机制优化
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.LeakyReLU(0.1),
#             nn.Linear(hidden_size, 1, bias=False)
#         )
        
#         # 更高效的分类头
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size * 2 + 2, hidden_size),
#             nn.LeakyReLU(0.1),
#             nn.LayerNorm(hidden_size),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, output_size)
#         )
        
#         self.init_weights()

#     def init_weights(self):
#         # LSTM初始化调整
#         for name, param in self.lstm.named_parameters():
#             if 'weight' in name:
#                 if 'ih' in name:
#                     nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
#                 else:
#                     nn.init.orthogonal_(param.data)
#             elif 'bias' in name:
#                 nn.init.constant_(param[:param.shape[0]//4], 1)
#                 nn.init.constant_(param[param.shape[0]//4:param.shape[0]//2], 0.1)  # 输入门偏置微调
#                 nn.init.constant_(param[param.shape[0]//2:], 0)
        
#         # 注意力层初始化
#         for layer in self.attention:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
    
#     def forward(self, x, lengths, last_pos_x=None):
#         # 输入处理
#         x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)
#         x = self.input_dropout(x)
#         x = x.flip(1)
        
#         # LSTM处理
#         packed_out, _ = self.lstm(x)
#         packed_out = torch.tanh(packed_out)
        
#         # 注意力机制
#         attention_scores = self.attention(packed_out)
#         attention_weights = F.softmax(attention_scores, dim=1)
#         context_vector = torch.sum(attention_weights * packed_out, dim=1)
        
#         # 输出处理
#         combined = torch.cat((context_vector, last_pos_x), 1)
#         return self.fc(combined)