import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, layers:list, structure:list, activation:str):
        super(Classifier, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.orthogonal_(layer.weight)
            self.layers.append(layer)
            res = f'Layer[{i+1}]: {layers[i]} - {layers[i+1]}'
            structure.append(res)
        
    def forward(self, x):
        if self.activation == "relu":
            activationFunction = F.relu
        elif self.activation == "tanh":
            activationFunction = F.tanh
        elif self.activation == "softmax":
            activationFunction = F.softmax
        elif self.activation == "LeakyReLU":
            activationFunction = F.leaky_relu(0.1)
    
        for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = activationFunction(x)
        return x