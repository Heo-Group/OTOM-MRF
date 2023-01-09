
import torch
import torch.nn as nn
import numpy as np


class nnModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers,dropout, bidirectional,device):
    super().__init__()
    self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
    self.num_layers = num_layers
    self.num_directions = 2 if bidirectional else 1
    self.device = device

    self.lstm = nn.LSTM(input_size=self.input_size, hidden_size = self.hidden_size, num_layers=self.num_layers, bidirectional=bidirectional, batch_first=True).to(device)
    self.output_fc = nn.Linear(self.num_directions*self.hidden_size, self.output_size).to(device)

    self.relu = nn.ReLU()
    self.sig = nn.Sigmoid()

  def forward(self, x):

    self.batch_size = x.size(0)
    
    h0, c0 = self.init_hidden()
    h0, c0 = h0.to(self.device), c0.to(self.device)

    output, (ht,ct) = self.lstm(x, (h0, c0))

    if self.num_directions == 2:
      hidden = torch.cat((ht[-2,:,:], ht[-1,:,:]),1)
    else:
      hidden = ht[-1,:,:]
      
    output = self.output_fc(hidden)
    output = self.relu(output)


    return output

  def init_hidden(self):
    h0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.hidden_size)
    c0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.hidden_size)
    
    return h0, c0
