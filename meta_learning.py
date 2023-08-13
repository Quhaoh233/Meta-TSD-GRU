import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networks
import time_seires_decompostion
from tqdm import tqdm


def pre_training(train, valid, target_park, lookback, meta_loader_dict, epoch=1000, lr=0.03):
    data = pd.concat((train, valid), axis=0)
    park_names = data.columns
    park_type = target_park[0]

    # select appropriate source domains
    idx = []
    for col in range(len(park_names)):
        if park_type in park_names[col] and target_park != park_names[col]:
            idx.append(col)
    idx = np.array(idx).astype(dtype=int)
    source_idx = np.random.randint(0, len(idx), 3)

    # init
    meta_model = networks.TsdGru(1, 1, 1, lookback, 1)
    meta_optim = torch.optim.Adam(meta_model.parameters())
    loss_function = nn.MSELoss()

    # outer loop
    for e in tqdm(range(epoch), desc='PROCESS = Meta-Learning Pre-Training'):
        # init gradients
        grad_dict = dict()
        for name, param in meta_model.named_parameters():
            grad_dict[name] = 0

        # inner loop
        for i in range(len(source_idx)):
            temp_model = copy.deepcopy(meta_model)
            temp_optim = torch.optim.Adam(temp_model.parameters())
            park_id = park_names[source_idx[i]]
            for j, batch in enumerate(meta_loader_dict['train', park_id]):
                occ, cyc, eff, label = batch
                temp_optim.zero_grad()
                predict = temp_model(occ, cyc, eff)
                loss = loss_function(predict, label)
                loss.backward()
                temp_optim.step()

            for j, batch in enumerate(meta_loader_dict['valid', park_id]):
                occ, cyc, eff, label = batch
                temp_optim.zero_grad()
                predict = temp_model(occ, cyc, eff)
                loss = loss_function(predict, label)
                loss.backward()

            for name, param in temp_model.named_parameters():
                grad_dict[name] += param.grad.detach()

        # update meta model
        for name, param in meta_model.named_parameters():
            param.data = param.data - lr * grad_dict[name] / len(source_idx)

    return meta_model

