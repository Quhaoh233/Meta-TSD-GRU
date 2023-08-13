import copy
import sys
import torch.nn as nn
import time_seires_decompostion as tsd
import functions as fn
import meta_learning


# hyper params
meta = True
model_name = 'TSD-GRU'
lookback = 6
predict_length = 6
batch_size = 1024
epoch = 1000

# input data
train, valid, test = fn.read_data()
time_stamps = train.shape[0] + valid.shape[0] + test.shape[0]
cycle = tsd.tsd_pre_training(train, valid, time_stamps)  # tsd
park_names = train.columns
_, meta_loader_dict = tsd.get_metaloader(train, valid, test, cycle, lookback, threshold=0)
dataset_dict, data_loader_dict = tsd.get_dataloader(train, valid, test, cycle, batch_size, lookback, threshold=0)
model_dict, optimizer_dict = tsd.get_model_optim(park_names, lookback)
loss_function = nn.MSELoss()

for i in range(len(park_names)):
    print('Training in', park_names[i], '| model =', model_name)

    # meta-learning pre-training
    if meta is True:
        model_dict[park_names[i]] = meta_learning.pre_training(train, valid, park_names[i], lookback, meta_loader_dict)

    test_th = 100
    test_model = copy.deepcopy(model_dict[park_names[i]])
    for e in range(epoch):
        # training
        model_dict[park_names[i]].train()
        for j, batch in enumerate(data_loader_dict['train', park_names[i]]):
            occupancy, cycle_term, effect_term, label = batch
            optimizer_dict[park_names[i]].zero_grad()
            predict = model_dict[park_names[i]](occupancy, cycle_term, effect_term)
            loss = loss_function(predict, label)
            loss.backward()
            optimizer_dict[park_names[i]].step()

        # valid
        model_dict[park_names[i]].eval()
        for j, batch in enumerate(data_loader_dict['valid', park_names[i]]):
            occupancy, cycle_term, effect_term, label = batch
            predict = model_dict[park_names[i]](occupancy, cycle_term, effect_term)
            valid_loss = loss_function(predict, label)

            if valid_loss < test_th:
                test_th = valid_loss
                test_model = copy.deepcopy(model_dict[park_names[i]])
        if (e+1) % 10 == 0:
            print('EPOCH = %s / %s' % (e+1, epoch), '| train loss =', loss.item(), '| valid loss =', valid_loss.item())

    # test
    for j, batch in enumerate(data_loader_dict['test', park_names[i]]):
        occupancy, cycle_term,effect_term, label = batch
        test_predict = test_model(occupancy, cycle_term, effect_term)
        test_loss = loss_function(test_predict, label)
        np_predict = test_predict.detach().numpy()
        np_label = label.detach().numpy()
        mse, rmse, mae, mape, r2 = fn.calculate_metrics(np_predict, np_label)
        print('Test in', park_names[i], '| model =', model_name)
        print('MSE = %s, RMSE = %s, MAE = %s, MAPE = %s, R-square = %s' % (mse, rmse, mae, mape, r2))
    sys.exit()



