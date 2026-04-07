import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from molhiv.model import GCN
from tqdm import tqdm
from molhiv.utils import Metric

def train(model: GCN, loader: DataLoader, optimizer:torch.optim, criterion: nn.CrossEntropyLoss, max_grad_norm: float = 1.0, device: str="cpu"):
    model.train()
    loss = 0
    total_sample = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
        y = batch.y.float().squeeze().to(device)
        l = criterion(out, y)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        loss += (l.item() * batch.y.shape[0])
        total_sample += batch.y.shape[0]
    loss /= total_sample
    return loss

@torch.no_grad()
def val(model: GCN, loader: DataLoader, criterion: nn.CrossEntropyLoss, device: str="cpu"):
    loss = 0
    total_sample = 0
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
        y = batch.y.float().squeeze().to(device)
        l = criterion(out, y)
        loss += (l.item()*batch.y.shape[0])
        total_sample += batch.y.shape[0]
    loss /= total_sample
    return loss

@torch.no_grad()
def predict(model: GCN, dataloader: DataLoader, device:str = "cpu"):
    model.eval()
    outs = []
    ys = []
    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
        outs.append(out)
        y = batch.y.float().squeeze().to(device)
        ys.append(y)
    return torch.cat(outs, dim=0).cpu(), torch.cat(ys).cpu()

def train_val(model: GCN, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim, criterion: nn.CrossEntropyLoss, metrics: list[Metric], max_grad_norm: float=1.0, device: str="cpu"):
    model = model.to(device)
    train_loss = train(model, train_loader, optimizer, criterion, max_grad_norm, device)
    val_loss = val(model, val_loader, criterion, device)
    preds = {
        "train": predict(model, train_loader, device),
        "val": predict(model, val_loader, device)
        }
    result = {"train_loss": train_loss, "val_loss": val_loss}
    
    for m in metrics:
        outs, ys = preds[m.split]
        result[m.name] = m.compute(outs, ys)

    return result