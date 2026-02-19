import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t, h):
        return self.net(h)

class NeuralODE(nn.Module):
    def __init__(self, args, config):
        super(NeuralODE, self).__init__()
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        out_dim = config.model.out_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.ode_func = ODEFunc(hidden_dim)
        
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
        t = torch.tensor([0., 1.]).to(x.device)

        h0, (h_n, c_n) = self.lstm(x)
        h0 = h0[:, -1, :]
        out = odeint(self.ode_func, h0, t)
        # print(f"out: {out.shape}")

        out = out[1]
        
        mu = self.mean_mlp(out)
        var = self.var_mlp(out)

        # print(f"mu: {mu.shape}")
        # print(f"var: {var.shape}")

        return mu, var