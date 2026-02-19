import torch
import torch.nn as nn

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}



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

        self.condition_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim)
        )

        nn.init.zeros_(self.condition_proj[-1].weight)
        nn.init.zeros_(self.condition_proj[-1].bias)

    def forward(self, fx, cond_token):
        cond_params = self.condition_proj(cond_token)
        gamma_1, beta_1, gamma_2, beta_2 = cond_params.chunk(4, dim=-1)
        gamma_1, beta_1 = gamma_1.unsqueeze(1), beta_1.unsqueeze(1)
        gamma_2, beta_2 = gamma_2.unsqueeze(1), beta_2.unsqueeze(1)

        norm_fx_1 = self.ln_1(fx) * (1 + gamma_1) + beta_1
        attn_output, _ = self.Attn(norm_fx_1, norm_fx_1, norm_fx_1, need_weights=False)
        fx = attn_output + fx

        norm_fx_2 = self.ln_2(fx) * (1 + gamma_2) + beta_2
        fx = self.mlp(norm_fx_2) + fx

        return fx


class RITransformer(nn.Module):
    def __init__(self, args, config):
        super(RITransformer, self).__init__()
        n_layers = config.model.n_layers
        input_dim = config.model.input_dim
        n_hidden = config.model.n_hidden
        act = config.model.act
        n_head = config.model.n_head
        mlp_ratio = config.model.mlp_ratio
        out_dim = config.model.out_dim
        dropout = config.model.dropout

        self.pos_encoder = nn.Parameter(torch.randn(1, args.seq_length, n_hidden))

        self.preprocess = MLP(input_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout, act=act,
                                                      mlp_ratio=mlp_ratio)
                                     for _ in range(n_layers)])
        
        self.mean_mlp = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, out_dim)
        )
        
        self.var_mlp = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, out_dim),
            nn.Softplus() 
        )

        self.mix = nn.MultiheadAttention(embed_dim=n_hidden, num_heads=1, batch_first=True)

    def regime_infer(self, x):
        batch_size, seq_len, n_hidden = x.shape

        random_token = torch.randn(batch_size, 1, n_hidden)
        random_token = random_token.to(x.device)
        x_extended = torch.cat((x, random_token), dim=1)

        x_extended, _ = self.mix(x_extended, x_extended, x_extended)
        regime_token = x_extended[:, -1, :]
        regime_token= self.gram_schmidt(x.mean(dim=1), regime_token)

        return regime_token

    def gram_schmidt(self, u, v):
        dot_uv = torch.sum(v * u, dim=1, keepdim=True)
        dot_uu = torch.sum(u * u, dim=1, keepdim=True)
        proj_u_v = (dot_uv / (dot_uu + 1e-8)) * u
        orthogonal_token = v - proj_u_v

        return orthogonal_token

    def forward(self, x):
        # x: (batch_size, seq_len, 5)
        x = self.preprocess(x)
        regime_token = self.regime_infer(x)
        x = x + self.pos_encoder

        for block in self.blocks:
            x = block(x, regime_token)

        x = x[:, -1, :]
        
        mu = self.mean_mlp(x)
        var = self.var_mlp(x)

        return mu, var