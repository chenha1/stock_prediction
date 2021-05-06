import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

if torch.cuda.is_available() and False:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 != None and c0 != None:
            out, (hn, cn) = self.lstm(x, (h0, c0))
        else:
            out, (hn, cn) = self.lstm(x)
        out = self.fc(out) 
        return out

class Linear(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Linear, self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.linears.extend([nn.Linear(hidden_size, hidden_size) for i in range(1, self.num_layers-1)])
        self.linears.append(nn.Linear(hidden_size, output_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        for l in self.linears:
            out = l(out)
            out = self.sigmoid(out)
        return out


