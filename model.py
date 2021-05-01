import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

if torch.cuda.is_available() and False:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# class Model(nn.Module):
#     def __init__(self, inputSize, outputSize, embeddingSize, hiddenSize, layers):
#         super(Model, self).__init__()
#         self.outputSize = outputSize
#         self.layers = layers
#         self.hiddenSize = hiddenSize

#         # feed data into an embedding matrix 
#         # self.embedding = nn.Embedding()

#         # feed embedded data into lstm module
#         # batch first = True ensures [batch, ...]
#         embeddingSize = inputSize
#         self.lstm = nn.LSTM(embeddingSize, hiddenSize, layers, batch_first=True)
#         self.fc = nn.Linear(hiddenSize, outputSize)
    
#     def forward(self, x, hidden):
#         batchSize, seqLen, features = x.shape
        
#         out, hidden = self.lstm(x, hidden)
#         out = self.fc(out)
#         out = out.view(batchSize, seqLen, features)
#         return out, hidden
    
#     def init_hidden(self, batchSize):
#         weight = next(self.parameters()).data 
#         hidden = (weight.new(self.layers, batchSize, self.hiddenSize).zero_().to(device),
#                     weight.new(self.layers, batchSize, self.hiddenSize).zero_().to(device))
#         return hidden

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


