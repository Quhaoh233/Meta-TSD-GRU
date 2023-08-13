import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import torch
import networks
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from tqdm import tqdm


def read_data(train_rate=0.6, valid_rate=0.2):
    if train_rate + valid_rate > 1:
        print('Input Error. train_rate + valid > 1.')
        sys.exit()

    occ = pd.read_csv('dataset/occupancy.csv', header=0, index_col=None)
    inf = pd.read_csv('dataset/inf.csv', header=0, index_col=None)
    capability = np.array(inf['CAPACITY'])
    occ.iloc[:, 1:] = occ.iloc[:, 1:] / capability  # occupancy rate
    sample_num = occ.shape[0]
    train_index = int(sample_num * train_rate)
    valid_index = int(sample_num * (train_rate + valid_rate))
    occ_train = occ.iloc[:train_index, 1:]
    occ_valid = occ.iloc[train_index:valid_index, 1:]
    occ_test = occ.iloc[valid_index:, 1:]
    return occ_train, occ_valid, occ_test


def create_rnn_data(data, lookback, predict_time):
    x = []
    y = []
    for i in range(len(data) - lookback - predict_time):
        x.append(data[i:i + lookback])
        y.append(data[i + lookback + predict_time])

    return np.array(x), np.array(y)


class CreateDataset(Dataset):
    def __init__(self, occ, lb, pt):
        occ = np.array(occ).astype(dtype=float)
        occ, label = create_rnn_data(occ, lb, pt)
        occ = torch.Tensor(occ)
        label = torch.Tensor(label)
        self.occ = torch.unsqueeze(occ, dim=2)  # (sample_num, seq_length, feature_num)
        self.label = torch.unsqueeze(label, dim=1)  # (sample_num, feature_num)

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):
        return self.occ[idx, :, :], self.label[idx, :]
    

def get_dataloader(train, valid, test, batch_size, lookback, predict_length):
    dataset_dict = dict()
    data_loader_dict = dict()
    columns = train.columns
    for i in range(train.shape[1]):
        dataset_dict['train', columns[i]] = CreateDataset(train.iloc[:, i], lookback, predict_length)
        data_loader_dict['train', columns[i]] = DataLoader(dataset_dict['train', columns[i]], shuffle=True, batch_size=batch_size)
        dataset_dict['valid', columns[i]] = CreateDataset(valid.iloc[:, i], lookback, predict_length)
        data_loader_dict['valid', columns[i]] = DataLoader(dataset_dict['valid', columns[i]], shuffle=False, batch_size=valid.shape[0])
        dataset_dict['test', columns[i]] = CreateDataset(test.iloc[:, i], lookback, predict_length)
        data_loader_dict['test', columns[i]] = DataLoader(dataset_dict['test', columns[i]], shuffle=False, batch_size=test.shape[0])
    return dataset_dict, data_loader_dict


def get_model_optim(headlines, model_name, lookback, in_s=1, hid_s=3, out_s=1, n_layer=1, weight_decay=0.00001):
    model_dict = dict()
    optimizer_dict = dict()

    for i in range(len(headlines)):
        if model_name == 'lstm':
            model_dict[model_name, headlines[i]] = networks.LSTM(in_s, hid_s, out_s, lookback, n_layer)
            optimizer_dict[model_name, headlines[i]] = torch.optim.Adam(model_dict[model_name, headlines[i]].parameters(), weight_decay=weight_decay)
        elif model_name == 'gru':
            model_dict[model_name, headlines[i]] = networks.GRU(in_s, hid_s, out_s, lookback, n_layer)
            optimizer_dict[model_name, headlines[i]] = torch.optim.Adam(model_dict[model_name, headlines[i]].parameters(), weight_decay=weight_decay)
        else:  # MLP
            model_dict[model_name, headlines[i]] = networks.MLP(in_s, hid_s, out_s, lookback, dropout=0.5)
            optimizer_dict[model_name, headlines[i]] = torch.optim.Adam(model_dict[model_name, headlines[i]].parameters(), weight_decay=weight_decay)

    return model_dict, optimizer_dict


def calculate_metrics(predict, label):
    mse = mean_squared_error(label, predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(label, predict)
    mape = mean_absolute_percentage_error(label, predict)
    r2 = r2_score(label, predict)
    return mse, rmse, mae, mape, r2


