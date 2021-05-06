import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import sys
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tqdm
import sys
from model import *
from dataloader import *
import statistics
from pickle import load, dump
import seaborn as sns
import pandas as pd
from utils import *

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    print('using cpu')
    device = torch.device('cpu')

is_load = False
is_save = False

dataDir = 'D:/Documents/PythonTest/stockpredict/data/Stocks'

# def train(model, data, batch_size, epochs=100):
#     loss = nn.MSELoss(reduction='mean')
#     lr = 0.01
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     for i in range(epochs):
#         # h = model.init_hidden(batch_size)
        
#         for j, (inputs, labels) in enumerate(data):
#             loss_lst = []
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             for batch in range(batch_size):
#                 X, y = inputs[batch], labels[batch]
#                 output = model(X)
#                 #print(f'O: {output}\ny: {y}\n', flush=True)
#                 l = loss(output, y)
#                 loss_lst.append(l.item())
#                 optimizer.zero_grad()
#                 l.backward()
#                 optimizer.step()
#             if j % 100 == 99:
#                 print(f'{j}: {sum(loss_lst)/len(loss_lst)}', flush=True)
#     return model 

def train(model, data, batch_size, epochs=100):
    loss = nn.MSELoss(reduction='mean')
    lr = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        # h = model.init_hidden(batch_size)
        
        for j, (inputs, labels) in enumerate(data):
            loss_lst = []
            inputs = inputs.to(device)
            labels = labels.to(device)
            X, y = inputs, labels
        
            output = model(X)
            # print(f'O: {output}\ny: {y}\n', flush=True)
            l = loss(output, y)
            loss_lst.append(l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if i % 1 == 0:
            print(f'{i}: {sum(loss_lst)/len(loss_lst)}', flush=True)
    return model 

def test(model, data, is_lstm=False):
    loss = nn.MSELoss(reduction='mean')
    loss_lst = []
    baseline_lst = []
    for i, (inputs, labels) in enumerate(data):
        inputs = inputs.to(device)
        labels = labels.to(device)
        X, y = inputs, labels
        
        output = model(X)

        if is_lstm:
            baseline_y = X[:, -1, :]
            y = y[:, -1, :]
            output = output[:, -1, :]
        else:
            baseline_y = X[:, -5:]
        loss_lst.append(loss(output, y))
        baseline_lst.append(loss(baseline_y, y))

    print(f'loss: {sum(loss_lst)/len(loss_lst)}')
    print(f'baseline: {sum(baseline_lst)/len(baseline_lst)}')

def demo(model, data, y_scaler, exposure_period=1, is_lstm=False):
    pred = []
    actual = []
    input_stream = []
    output = None
    print('running demo ...', flush=True)
    i = 0
    for inputs, labels in tqdm(data):
        if i % exposure_period == 0:
            input_stream = inputs
        elif not is_lstm:
            input_stream = input_stream.tolist()
            input_stream = input_stream[:, 5:] + output.tolist()[:, :]
            input_stream = torch.tensor(input_stream).double()
        else:
            output = output.view(1,1, -1).to(device)
            input_stream = input_stream[:, 1:, :].to(device)
            input_stream = torch.cat((input_stream, output), dim=1)
        inputs = input_stream.to(device)
        labels = labels.to(device)
        X, y = inputs, labels
        output = model(X)
        output = torch.squeeze(output)
        y = torch.squeeze(y)
        if is_lstm:
            output = output[-1]
            y = y[-1]
        
        pred.append(output.to('cpu').tolist())
        actual.append(y.to('cpu').tolist())
        i += 1
    predict = pd.DataFrame(y_scaler.inverse_transform(pred))
    original = pd.DataFrame(y_scaler.inverse_transform(actual))
    # print(f'predict: {predict}\noriginal: {original}', flush=True)
    plot(original, predict)

# def demo(model, data, y_scaler, exposure_period=1):
#     pred = []
#     actual = []
#     input_stream = []
#     output = None
#     print('running demo ...', flush=True)
#     for i, (inputs, labels) in tqdm(enumerate(data)):
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         X, y = inputs[0], labels[0]
#         output = model(X)
#         #print(f'X: {X}\nO: {output}\n', flush=True)
#         pred.append(output.to('cpu').tolist())
#         actual.append(y.to('cpu').tolist())

#     # pred = torch.tensor(pred)
#     # actual = torch.tensor(actual)
#     # predict = pd.DataFrame(y_scaler.inverse_transform(pred.detach().numpy()))
#     # original = pd.DataFrame(y_scaler.inverse_transform(actual.detach().numpy()))
#     predict = pd.DataFrame(y_scaler.inverse_transform(pred))
#     original = pd.DataFrame(y_scaler.inverse_transform(actual))
#     # print(f'predict: {predict}\noriginal: {original}', flush=True)
#     plot(original, predict)

    


if __name__ == '__main__':
    window = 20 # the size of the window
    features = 5
    embedding = 5
    hiddenSize = 100
    layers = 2
    batch_size = 10

    # train a model for each speaker, and reserve data for testing
    # model = Linear(features * window, features, hiddenSize, layers)
    model = LSTM(features, hiddenSize, layers, features)
    model.double()
    model.to(device)
    if is_load:
        train_data = load(open('train_data.pkl', 'rb'))
        test_data = load(open('test_data.pkl', 'rb'))
        X_scaler = load(open('X_scaler.pkl', 'rb'))
        y_scaler = load(open('y_scaler.pkl', 'rb'))
    else:
        # train_data, test_data, X_scaler, y_scaler = get_data_window(dataDir, window, batch_size, 0.2)
        train_data, test_data, scaler_lst = get_data_timeseries(dataDir, window, batch_size, 0.2)
        y_scaler = scaler_lst[0]
    if is_save:
        dump(train_data, open('train_data.pkl', 'wb'))
        dump(test_data, open('test_data.pkl', 'wb'))
        dump(X_scaler, open('X_scaler.pkl', 'wb'))
        dump(y_scaler, open('y_scaler.pkl', 'wb'))

    print('training', flush=True)
    model = train(model, train_data, batch_size, epochs=100)   
    test(model, test_data, is_lstm=True)
    # data, X_scaler, y_scaler = get_data_window(dataDir, window, batch_size, 0.2, is_demo=True)

    data, scaler_lst = get_data_timeseries(dataDir, window, batch_size, 0, is_demo=True)
    y_scaler = scaler_lst[0]
    demo(model, data, y_scaler, exposure_period=1, is_lstm=True)
