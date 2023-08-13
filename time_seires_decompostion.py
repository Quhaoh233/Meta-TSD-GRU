import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import functions as fn
import networks


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return x


def to_fourier(x, signal_num, period):
    fourier_x = x
    for i in range(signal_num):
        n = i + 1
        cos_1 = np.cos(n * 2 * np.pi * x / period)
        sin_1 = np.sin(n * 2 * np.pi * x / period)
        fourier_x = np.concatenate((fourier_x, cos_1, sin_1), axis=1)
    fourier_x = fourier_x[:, 1:]  # drop x, shape = [time_steps, 2*N]
    return fourier_x


def tsd_pre_training(train, valid, out_time_stamps, signal_num=30, period=2016, lr=0.05, epoch=500):
    model_dict = dict()
    optim_dict = dict()
    loss_function = torch.nn.MSELoss()
    park_names = train.columns

    # Fourier
    x = np.array(np.arange(train.shape[0] + valid.shape[0])).reshape(-1, 1)  # (train+valid, 1)
    fourier_x = to_fourier(x, signal_num, period)  # shape = [time_steps, 2*N]
    fourier_x_tensor = torch.Tensor(fourier_x)
    y = pd.concat((train, valid), axis=0)

    cyc_list = []

    for i in tqdm(range(len(park_names)), desc='PROCESS = Time Series Decomposition'):
        label = np.array(y[park_names[i]]).reshape(-1, 1)
        label = torch.Tensor(label)

        model_dict[park_names[i]] = Linear(2 * signal_num, 1)
        optim_dict[park_names[i]] = torch.optim.SGD(model_dict[park_names[i]].parameters(), lr=lr)

        # cycle term
        model_dict[park_names[i]].train()
        for e in range(epoch):
            optim_dict[park_names[i]].zero_grad()
            predict = model_dict[park_names[i]](fourier_x_tensor)
            loss = loss_function(predict, label)
            loss.backward()
            optim_dict[park_names[i]].step()

        model_dict[park_names[i]].eval()
        out_x = np.array(np.arange(out_time_stamps)).reshape(-1, 1)
        out_fourier_x = to_fourier(out_x, signal_num, period)
        out_fourier_x_tensor = torch.Tensor(out_fourier_x)
        cyc = model_dict[park_names[i]](out_fourier_x_tensor)  # (time_stamps, 1)
        cyc = cyc.detach().numpy()
        cyc_list.append(cyc)

    cyc_out = np.stack(cyc_list, axis=1).squeeze(axis=2)
    cyc_out = pd.DataFrame(cyc_out, columns=park_names)

    return cyc_out


def create_cyc_data(data, lookback, predict_time):
    x = []
    for i in range(len(data) - lookback - predict_time):
        x.append(data[i+lookback:i+lookback+predict_time])
    return np.array(x)


class CreateDataset(Dataset):
    def __init__(self, occ, cyc, lb, threshold=0):
        pt = lb
        occ = np.array(occ).astype(dtype=float)
        occ, label = fn.create_rnn_data(occ, lb, pt)  # shape (:, lookback)

        cyc = np.array(cyc).astype(dtype=float)
        cyc = create_cyc_data(cyc, lb, pt)  # shape (:, lookback)

        eps = 0.0001
        eff = (occ[:, -1] - cyc[:, 0]) / (cyc[:, 0] + eps)
        eff[np.where(abs(eff) <= threshold)] = 0  # ignore the small effects

        occ = torch.Tensor(occ)
        cyc = torch.Tensor(cyc)
        eff = torch.Tensor(eff)
        label = torch.Tensor(label)
        self.occ = torch.unsqueeze(occ, dim=2)  # (sample_num, seq_length, feature_num)
        self.cyc = torch.unsqueeze(cyc, dim=2)
        self.eff = torch.unsqueeze(eff, dim=1)
        self.label = torch.unsqueeze(label, dim=1)  # (sample_num, feature_num)

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):
        return self.occ[idx, :, :], self.cyc[idx, :, :], self.eff[idx, :], self.label[idx, :]


def get_dataloader(train, valid, test, cycle, batch_size, lookback, threshold):
    train_cycle = cycle.iloc[:train.shape[0], :]
    valid_cycle = cycle.iloc[train.shape[0]:train.shape[0]+valid.shape[0], :]
    test_cycle = cycle.iloc[train.shape[0]+valid.shape[0]:, :]
    dataset_dict = dict()
    data_loader_dict = dict()
    columns = train.columns
    for i in range(train.shape[1]):
        dataset_dict['train', columns[i]] = CreateDataset(train.iloc[:, i], train_cycle.iloc[:, i], lookback, threshold)
        data_loader_dict['train', columns[i]] = DataLoader(dataset_dict['train', columns[i]], shuffle=False, batch_size=batch_size)
        dataset_dict['valid', columns[i]] = CreateDataset(valid.iloc[:, i], valid_cycle.iloc[:, i], lookback, threshold)
        data_loader_dict['valid', columns[i]] = DataLoader(dataset_dict['valid', columns[i]], shuffle=False, batch_size=valid.shape[0])
        dataset_dict['test', columns[i]] = CreateDataset(test.iloc[:, i], test_cycle.iloc[:, i], lookback, threshold)
        data_loader_dict['test', columns[i]] = DataLoader(dataset_dict['test', columns[i]], shuffle=False, batch_size=test.shape[0])
    return dataset_dict, data_loader_dict


def get_metaloader(train, valid, test, cycle, lookback, threshold):
    train_cycle = cycle.iloc[:train.shape[0], :]
    valid_cycle = cycle.iloc[train.shape[0]:train.shape[0]+valid.shape[0], :]
    test_cycle = cycle.iloc[train.shape[0]+valid.shape[0]:, :]
    dataset_dict = dict()
    data_loader_dict = dict()
    columns = train.columns
    for i in range(train.shape[1]):
        dataset_dict['train', columns[i]] = CreateDataset(train.iloc[:, i], train_cycle.iloc[:, i], lookback, threshold)
        data_loader_dict['train', columns[i]] = DataLoader(dataset_dict['train', columns[i]], shuffle=False, batch_size=train.shape[0])
        dataset_dict['valid', columns[i]] = CreateDataset(valid.iloc[:, i], valid_cycle.iloc[:, i], lookback, threshold)
        data_loader_dict['valid', columns[i]] = DataLoader(dataset_dict['valid', columns[i]], shuffle=False, batch_size=valid.shape[0])
        dataset_dict['test', columns[i]] = CreateDataset(test.iloc[:, i], test_cycle.iloc[:, i], lookback, threshold)
        data_loader_dict['test', columns[i]] = DataLoader(dataset_dict['test', columns[i]], shuffle=False, batch_size=test.shape[0])
    return dataset_dict, data_loader_dict


def get_model_optim(headlines, lookback, in_s=1, hid_s=1, out_s=1, n_layer=1, weight_decay=0.00001):
    model_dict = dict()
    optimizer_dict = dict()

    for i in range(len(headlines)):
        model_dict[headlines[i]] = networks.TsdGru(in_s, hid_s, out_s, lookback, n_layer)
        optimizer_dict[headlines[i]] = torch.optim.Adam(model_dict[headlines[i]].parameters(), weight_decay=weight_decay)

    return model_dict, optimizer_dict
