import os
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


class GaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, mu, var, y):
        var = torch.clamp(var, min=self.eps)
        nll_element_wise = 0.5 * (torch.log(var) + (y - mu)**2 / var)

        if self.reduction == 'mean':
            return torch.mean(nll_element_wise)
        elif self.reduction == 'sum':
            return torch.sum(nll_element_wise)
        else:
            return nll_element_wise
        

def train(args, model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.end_learning_rate
    )

    loss_list = []
    runtime_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fnc = GaussianNLLLoss()

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for x, y in dataloader:
            if torch.isnan(x).any() or torch.isnan(y).any():
                print(f"FATAL: Found NaN")
                exit()
            if torch.isinf(x).any() or torch.isinf(y).any():
                print(f"FATAL: Found Inf")
                exit()
        
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            mu, var = model(x)

            loss = loss_fnc(mu, var, y)
            # print(f"loss: {loss}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_runtime = time.time() - start_time

        avg_loss = epoch_loss / len(dataloader)
        loss_list.append(avg_loss)
        runtime_list.append(epoch_runtime)

        scheduler.step()

        if (epoch) % args.save_freq == 0:
            model_save_path = os.path.join(args.exp_name, f"model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch} to {model_save_path}")
    
        print(f"Epoch {epoch}/{args.num_epochs} - Loss: {avg_loss:.7f}, Runtime: {epoch_runtime:.2f}s")

    results = {"loss": loss_list, "runtime": runtime_list}
    txt_path = f"{args.exp_name}/results.txt"
    pkl_path = f"{args.exp_name}/results.pkl"

    model_save_path = os.path.join(args.exp_name, f"model.pth")
    torch.save(model.state_dict(), model_save_path)

    with open(txt_path, "w") as f:
        for epoch, (loss, runtime) in enumerate(zip(loss_list, runtime_list)):
            f.write(f"Epoch {epoch+1}: Loss = {loss:.7f}, Runtime = {runtime:.2f}s\n")

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Training completed. Results saved to {txt_path} and {pkl_path}")



def train_diffusion(args, model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.end_learning_rate
    )

    loss_list = []
    runtime_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fnc = nn.MSELoss()

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for x, y in dataloader:
            if torch.isnan(x).any() or torch.isnan(y).any():
                print(f"FATAL: Found NaN")
                exit()
            if torch.isinf(x).any() or torch.isinf(y).any():
                print(f"FATAL: Found Inf")
                exit()
        
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, y, loss_fnc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_runtime = time.time() - start_time

        avg_loss = epoch_loss / len(dataloader)
        loss_list.append(avg_loss)
        runtime_list.append(epoch_runtime)

        scheduler.step()

        if (epoch) % args.save_freq == 0:
            model_save_path = os.path.join(args.exp_name, f"model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch} to {model_save_path}")
    
        print(f"Epoch {epoch}/{args.num_epochs} - Loss: {avg_loss:.7f}, Runtime: {epoch_runtime:.2f}s")

    results = {"loss": loss_list, "runtime": runtime_list}
    txt_path = f"{args.exp_name}/results.txt"
    pkl_path = f"{args.exp_name}/results.pkl"

    model_save_path = os.path.join(args.exp_name, f"model.pth")
    torch.save(model.state_dict(), model_save_path)

    with open(txt_path, "w") as f:
        for epoch, (loss, runtime) in enumerate(zip(loss_list, runtime_list)):
            f.write(f"Epoch {epoch+1}: Loss = {loss:.7f}, Runtime = {runtime:.2f}s\n")

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Training completed. Results saved to {txt_path} and {pkl_path}")