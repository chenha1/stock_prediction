import numpy as np 
import os 
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def format_line(lines):
    lst = []
    for line in lines:
        line = line.split(',')

        # removing date and openInt
        line = line[1:-1]
        line = list(map(lambda x: float(x), line))
        lst.append(line)
    return lst

def fit_transform(data):
    '''
    scale data to 0 - 1 inclusive
    '''
    print('fit transforming the data', flush=True)
    all_X, all_y = [], []
    for x, y in data:
        all_X.append(x)
        all_y.append(y)
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    print('fitting X', flush=True)
    X = X_scaler.fit_transform(all_X)
    print('fitting Y', flush=True)
    y = y_scaler.fit_transform(all_y)
    return list(zip(X, y)), X_scaler, y_scaler
    


# def get_data_window(dataDir, window, batch_size, test_size, is_demo=False):
#     data = []
#     print('grabbing data ...', flush=True)
#     for root, dirs, files in os.walk(dataDir):
#         for name in tqdm(files):
#             fname = os.path.join(root, name)
#             with open(fname, 'r') as f:
#                 _ = f.readline()
#                 lines = f.readlines()
#                 lines = format_line(lines)
#                 x, y, tmp = [], [], []
      
#                 for i in range(len(lines) - window):
#                     tmp = lines[i:i+window]
#                     x.append([item for sublist in tmp for item in sublist])

#                     y.append(lines[i+window])
#                 for i in range(len(x)):
#                     data.append((x[i], y[i]))
#             break
    
#     data, X_scaler, y_scaler = fit_transform(data)
#     if is_demo:
#         return DataLoader(data), X_scaler, y_scaler
#     train, test = train_test_split(data, test_size=test_size, shuffle=True)
#     return DataLoader(train, batch_size=batch_size), DataLoader(test), X_scaler, y_scaler

def get_data_window(dataDir, window, batch_size, test_size, is_demo=False):
    data = []
    print('grabbing data ...', flush=True)
    for root, dirs, files in os.walk(dataDir):
        for name in ['aapl.us.txt']:
            fname = os.path.join(root, name)
            with open(fname, 'r') as f:
                _ = f.readline()
                lines = f.readlines()
                lines = format_line(lines)
                x, y, tmp = [], [], []

                offset = 0
                if not is_demo:
                    offset = int((len(lines) - window) / 1.5)
                for i in range(len(lines) - window - offset):
                    tmp = lines[i:i+window]
                    x.append([item for sublist in tmp for item in sublist])

                    y.append(lines[i+window])
                for i in range(len(x)):
                    data.append((x[i], y[i]))
    
    data, X_scaler, y_scaler = fit_transform(data)
    if is_demo:
        return DataLoader(data), X_scaler, y_scaler
    train, test = train_test_split(data, test_size=test_size, shuffle=True)
    return DataLoader(train, batch_size=batch_size), DataLoader(test), X_scaler, y_scaler

if __name__ == '__main__':
    train, test, X_scaler, y_scaler = get_data_window('D:/Documents/PythonTest/stockpredict/data/Stocks', 20, 100, 0.2)
    for x, y in test:
        print(f'x: {x}')
        print(f'y: {y}')
        pass
    
                