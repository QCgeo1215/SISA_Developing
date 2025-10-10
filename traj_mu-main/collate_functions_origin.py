# collate_functions.py
import torch

def pad_collate_fn(batch):
    x_data, y_data = zip(*batch)
    lengths = [len(x[0]) for x in x_data]  # Length of the original sequence
    max_len = max(lengths)
    padded_x_data = []
    for x in x_data:
        padded_sample = []
        for feature in x.T:  # Transpose to pad each feature
            padding_length = max_len - len(feature)
            padded_feature = [0] * padding_length + list(feature)
            padded_sample.append(padded_feature)
        padded_x_data.append(padded_sample)
    
    padded_x_data = torch.tensor(padded_x_data, dtype=torch.float32).permute(0, 2, 1)  # Adjust shape to (batch_size, seq_len, input_size)
    y_data = torch.tensor(y_data, dtype=torch.float32)#.squeeze(1) 
    return padded_x_data, y_data, lengths