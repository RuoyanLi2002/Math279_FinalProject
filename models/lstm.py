import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        out_dim = config.model.out_dim

        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.2
        )

        self.mean_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.var_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus() 
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        
        last_time_step = out[:, -1, :]

        mu = self.mean_mlp(last_time_step)
        var = self.var_mlp(last_time_step)

        return mu, var