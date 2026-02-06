import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        num_layers = config.model.num_layers
        out_dim = config.model.out_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        
        last_time_step = out[:, -1, :]

        mu = self.mean_mlp(last_time_step)
        var = self.var_mlp(last_time_step)

        return mu, var