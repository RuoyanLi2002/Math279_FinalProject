import math
import torch
import torch.nn as nn


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    return x*torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res

        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, n_layers=0, res=False, act=act)
        self.mlp_new = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, n_layers=0, res=False, act=act)

    def forward(self, fx):
        attn_output, attn_weights = self.Attn(self.ln_1(fx), self.ln_1(fx), self.ln_1(fx), need_weights=False)
        fx = attn_output + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        n_layers = config.model.n_layers
        input_dim = config.model.input_dim
        n_hidden = config.model.n_hidden
        act = config.model.act
        n_head = config.model.n_head
        mlp_ratio = config.model.mlp_ratio
        out_dim = config.model.out_dim
        dropout = config.model.dropout

        self.pos_encoder = nn.Parameter(torch.randn(1, config.model.seq_length, n_hidden))

        self.preprocess = MLP(input_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout, act=act,
                                                      mlp_ratio=mlp_ratio)
                                     for _ in range(n_layers)])
        
        self.out = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, out_dim)
        )

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(n_hidden,
                            n_hidden),
            torch.nn.Linear(n_hidden,
                            n_hidden),
        ])

        self.n_hidden = n_hidden



    def forward(self, x, t):
        temb = get_timestep_embedding(t, self.n_hidden)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        temb = temb.unsqueeze(1)

        x = self.preprocess(x)
        x = x + self.pos_encoder + temb

        for block in self.blocks:
            x = block(x)

        x = self.out(x)
        x = x[:, -1, :]

        return x
    
class Diffusion(nn.Module):
    def __init__(self, args, config):
        super(Diffusion, self).__init__()
        self.beta_min = config.model.beta_min
        self.beta_max = config.model.beta_max
        self.timesteps = config.model.timesteps

        self.betas = torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.timesteps)
        self.sqrt_betas = torch.sqrt(self.betas)

        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        
        self.model = Transformer(config)
            
        self.device = None

    def to(self, *args, **kwargs):
        """Override the default `to` method to ensure all tensors are moved to the specified device."""
        super(Diffusion, self).to(*args, **kwargs)

        device = next(self.parameters()).device
        self.betas = self.betas.to(device)
        self.sqrt_betas = self.sqrt_betas.to(device)
        self.alphas = self.alphas.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)

        self.model = self.model.to(device)

        self.device = device

        return self
    
    def extract(self, a, t):
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, 1, 1)

    def forward_diffusion(self, x_zeros, t): 
        epsilon = torch.randn_like(x_zeros).to(x_zeros.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t)
        
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar

        return noisy_sample, epsilon
    
    def forward(self, x, target, criterion):
        target = target.unsqueeze(1)
        t = torch.randint(low=0, high=self.timesteps, size=(target.shape[0],)).to(x.device)

        perturbed, epsilon = self.forward_diffusion(target, t)

        x = torch.concat([x, perturbed], dim = 1)
        pred_epsilon = self.model(x, t)
        
        loss = criterion(epsilon.squeeze(1), pred_epsilon)

        return pred_epsilon, loss
    
    def denoise_at_t(self, x_t, condition, t):
        timestep = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)

        if t > 0:
            alpha = self.extract(self.alphas, timestep)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep)
            sqrt_alpha = self.extract(self.sqrt_alphas, timestep)

            temp_x_t = torch.concat([x_t, condition], dim = 1)
            pred_epsilon = self.model(temp_x_t, timestep)

            x_t_minus_1 = 1/sqrt_alpha * (x_t - (1-alpha) / sqrt_one_minus_alpha_bar * pred_epsilon)

            return x_t_minus_1
        else:
            alpha = self.extract(self.alphas, timestep)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep)
            sqrt_alpha = self.extract(self.sqrt_alphas, timestep)

            temp_x_t = torch.concat([x_t, condition], dim = 1)
            pred_epsilon = self.model(temp_x_t, timestep)

            pred_x0 = 1/sqrt_alpha * (x_t - (1-alpha) / sqrt_one_minus_alpha_bar * pred_epsilon)

            return pred_x0
        
    
    def ddpm_sample(self, x, target):
        x_t = torch.randn(target.shape).to(target.device)

        for t in range(self.timesteps-1, 0, -1):
            x_t = self.denoise_at_t(x_t, x, t)

        return x_t

    def ddim_sample(self, x, target, grid, criterion=None):
        pass