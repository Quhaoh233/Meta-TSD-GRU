import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os


# baselines
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_l, num_layers):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size*seq_l, output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x, _ = self.rnn(x)  # shape [batch, seq, feature]
        x = self.flatten(x)
        x = self.linear(x)
        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_l, num_layers):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size*seq_l, output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x, _ = self.rnn(x)  # shape [batch, seq, feature]
        x = self.flatten(x)
        x = self.linear(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq, dropout):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(input_size*seq, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):  # shape [batch, seq, feature]
        x = self.flatten(x)
        x = self.l1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.l3(x)
        return x


class TsdGru(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_l, num_layers):
        super(TsdGru, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.l1 = nn.Linear(hidden_size*seq_l*2, hidden_size)
        self.l2 = nn.Linear(hidden_size+1, output_size)
        self.flatten = nn.Flatten()

    def forward(self, occ, cyc, eff):
        x = torch.concat((occ, cyc), dim=1)
        x, _ = self.rnn(x)  # shape [batch, seq, feature]
        x = self.flatten(x)
        x = self.l1(x)
        x = torch.concat((x, eff), dim=1)
        x = self.l2(x)
        return x
