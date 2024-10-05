import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_layer_sizes[0]))
        layers.append(nn.SiLU())  # Adding ReLU activation function
        
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.3))
        
        layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model