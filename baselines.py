import copy
import torch.nn as nn
import functions as fn


# hyper params
model_name = 'lstm'  # lstm, gru, mlp.
lookback = 6
predict_length = 6
batch_size = 1024
epoch = 1000

# input data
train, valid, test = fn.read_data()
park_names = train.columns
dataset_dict, data_loader_dict = fn.get_dataloader(train, valid, test, batch_size, lookback, predict_length)
model_dict, optimizer_dict = fn.get_model_optim(park_names, model_name, lookback)
loss_function = nn.MSELoss()


for i in range(len(park_names)):
    print('Model =', model_name, 'Training in', park_names[i], '| model =', model_name)

    test_th = 100
    test_model = copy.deepcopy(model_dict[model_name, park_names[i]])
    for e in range(epoch):

        # training
        model_dict[model_name, park_names[i]].train()
        for j, batch in enumerate(data_loader_dict['train', park_names[i]]):
            sample, label = batch
            optimizer_dict[model_name, park_names[i]].zero_grad()
            predict = model_dict[model_name, park_names[i]](sample)
            loss = loss_function(predict, label)
            loss.backward()
            optimizer_dict[model_name, park_names[i]].step()

        # valid
        model_dict[model_name, park_names[i]].eval()
        for j, batch in enumerate(data_loader_dict['valid', park_names[i]]):
            sample, label = batch
            predict = model_dict[model_name, park_names[i]](sample)
            valid_loss = loss_function(predict, label)

            if valid_loss < test_th:
                test_th = valid_loss
                test_model = copy.deepcopy(model_dict[model_name, park_names[i]])

        if (e+1) % 10 == 0:
            print('EPOCH = %s / %s' % (e+1, epoch), '| train loss =', loss.item(), '| valid loss =', valid_loss.item())

    # test
    for j, batch in enumerate(data_loader_dict['test', park_names[i]]):
        sample, label = batch
        test_predict = test_model(sample)
        test_loss = loss_function(test_predict, label)
        np_predict = test_predict.detach().numpy()
        np_label = label.detach().numpy()
        mse, rmse, mae, mape, r2 = fn.calculate_metrics(np_predict, np_label)
        print('Test in', park_names[i], '| model =', model_name)
        print('MSE = %s, RMSE = %s, MAE = %s, MAPE = %s, R-square = %s' % (mse, rmse, mae, mape, r2))



