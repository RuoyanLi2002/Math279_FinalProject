import os

import numpy as np
import torch
from tqdm import tqdm

from utils import load_dataset
from models.diffusion import Diffusion
from models.ritdiffusion import RITDiffusion


def _load_checkpoint(args, model, device):
    if args.model_path and args.model_path.strip():
        model_path = args.model_path
    else:
        model_path = os.path.join(args.exp_name, "model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


def eval(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _load_checkpoint(args, model, device)

    test_loader = load_dataset(args, start_year=2021, end_year=2021, shuffle=False)

    is_diffusion = isinstance(model, Diffusion) or isinstance(model, RITDiffusion)

    total_se = 0.0
    total_count = 0

    all_strategy_returns = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Eval", total=len(test_loader), unit="batch"):
            close_t = x[:, -1, -1]
            close_t_plus_1 = y[:, -1]
            target_return = torch.log(close_t_plus_1 / close_t).unsqueeze(-1)

            x = x.to(device)
            target_return = target_return.to(device)

            if is_diffusion:
                # 使用专用于 eval 的完整采样（不改原 ddpm_sample/denoise_at_t 接口）
                shape = target_return.unsqueeze(1).shape  # (B, 1, out_dim)
                sampled = model.sample_for_eval(x, shape)
                if sampled.ndim == 3 and sampled.size(1) == 1:
                    pred = sampled[:, 0, :]
                else:
                    pred = sampled
            else:
                mu, _ = model(x)
                if mu.ndim == 2 and mu.size(-1) > 1 and target_return.size(-1) == 1:
                    pred = mu[:, -1:].contiguous()
                else:
                    pred = mu

            diff = pred - target_return
            se = torch.sum(diff ** 2).item()

            total_se += se
            total_count += target_return.numel()

            strategy_returns = torch.sign(pred.squeeze(-1)) * target_return.squeeze(-1)
            all_strategy_returns.append(strategy_returns.detach().cpu())

    mse = total_se / total_count if total_count > 0 else float("nan")

    pnl = float("nan")
    sharpe = float("nan")
    if all_strategy_returns:
        sr = torch.cat(all_strategy_returns).numpy()
        pnl = float(sr.sum())
        if sr.std() > 0:
            sharpe = float(sr.mean() / (sr.std() + 1e-8) * np.sqrt(252.0))

    print(f"Test MSE on 2021: {mse:.7f}")
    print(f"Simple long/short PnL on 2021: {pnl:.6f}")
    print(f"Simple long/short Sharpe on 2021: {sharpe:.4f}")

    results_path = os.path.join(args.exp_name, "test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Test MSE on 2021: {mse:.7f}\n")
        f.write(f"Simple long/short PnL on 2021: {pnl:.6f}\n")
        f.write(f"Simple long/short Sharpe on 2021: {sharpe:.4f}\n")

    return {"mse": mse, "pnl": pnl, "sharpe": sharpe}