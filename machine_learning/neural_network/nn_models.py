import torch
from torch import nn, autograd
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, batch_size=64, num_layers=3, layers_dim=(223, 140, 70), lr=None, batch_norm=True,  drop_out=0.3,
                 l2_penalty=0.005, activation_func="relu"):
        super(NeuralNet, self).__init__()

        self._layers_dim = layers_dim
        self._l2_pen = l2_penalty
        self._batch_norm = batch_norm
        self._activation = activation_func
        self._drop_out = drop_out
        self._batch_size = batch_size
        self._num_layers = num_layers

        # create linear layers
        if num_layers != len(layers_dim):
            print("Error: number of layers of size " + str(num_layers) + "and layers dim of size " + str(len(layers_dim)))
            exit(1)

        self._layers_dim = list(layers_dim) + [1]
        self._linear_layer = nn.ModuleList()
        for i in range(num_layers):
            self._linear_layer.append(nn.Linear(self._layers_dim[i], self._layers_dim[i+1]))

        # set optimizer
        self.optimizer = self.set_optimizer(lr)

    # init optimizer with RMS_prop
    def set_optimizer(self, lr):
        lr = 0.001 if not lr else lr
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self._l2_pen)

    def _forward_layer(self, x, layer_num, last_layer=False):
        if self._batch_norm:
            nn.BatchNorm1d(self._layers_dim[layer_num])
        x = self._linear_layer[layer_num](x)
        if self._activation == "relu" and not last_layer:
            x = F.relu(x)
        elif self._activation == "elu" and not last_layer:
            x = F.elu(x)
        if self._drop_out and not last_layer:
            x = nn.Dropout(p=self._drop_out)
        return torch.sigmoid(x) if last_layer else x

    def forward(self, x):
        for i in range(self._num_layers - 1):
            x = self._forward_layer(x, i)
        return self._forward_layer(x, self._num_layers - 1, last_layer=True)


class NeuralNet3(nn.Module):
    def __init__(self, layers_size, lr):
        super(NeuralNet3, self).__init__()

        self._dim = layers_size
        # useful info in forward function
        self._in_dim = layers_size[0]

        self._input = nn.Linear(self._dim[0], self._dim[1])
        self._layer1 = nn.Linear(self._dim[1], self._dim[2])
        self._layer2 = nn.Linear(self._dim[2], 1)

        # set optimizer
        self.optimizer = self.set_optimizer(lr)

    # init optimizer with RMS_prop
    def set_optimizer(self, lr):
        return torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=0.002)

    def forward(self, x):
        x = x.view(-1, self._in_dim)
        x = self._input(x)
        nn.BatchNorm1d(self._dim[1], momentum=0.9)
        x = F.relu(x)
        nn.Dropout(p=0.3)
        x = self._layer1(x)
        nn.BatchNorm1d(self._dim[2])
        x = F.relu(x)
        nn.Dropout(p=0.2)
        x = self._layer2(x)
        x = torch.sigmoid(x)
        return x


class ActiveLearningModel(nn.Module):
    def __init__(self, layers_size, lr):
        super(ActiveLearningModel, self).__init__()

        self._dim = layers_size
        # useful info in forward function
        self._in_dim = layers_size[0]

        self._input = nn.Linear(self._dim[0], self._dim[1])
        self._layer1 = nn.Linear(self._dim[1], self._dim[2])
        self._layer2 = nn.Linear(self._dim[2], 1)

        # set optimizer
        self.optimizer = self.set_optimizer(lr)

    # init optimizer with RMS_prop
    def set_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.view(-1, self._in_dim)
        x = self._input(x)
        x = F.relu(x)
        x = self._layer1(x)
        x = F.relu(x)
        x = self._layer2(x)
        x = torch.sigmoid(x)
        return x

