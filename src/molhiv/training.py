import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from molhiv.model import GCN
from tqdm import tqdm
from molhiv.utils import Metric

def train(model: GCN, loader: DataLoader, optimizer:torch.optim, criterion: nn.CrossEntropyLoss, max_grad_norm: float = 1.0):
    model.train()
    loss = 0
    total_sample = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        l = criterion(out, batch.y.flatten())
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        loss += (l.item() * batch.y.shape[0])
        total_sample += batch.y.shape[0]
    loss /= total_sample
    return loss

@torch.no_grad()
def val(model: GCN, loader: DataLoader, criterion: nn.CrossEntropyLoss):
    loss = 0
    total_sample = 0
    model.eval()
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        l = criterion(out, batch.y.flatten())
        loss += (l.item()*batch.y.shape[0])
        total_sample += batch.y.shape[0]
    loss /= total_sample
    return loss

@torch.no_grad()
def predict(model: GCN, dataloader: DataLoader):
    model.eval()
    outs = []
    ys = []
    for loader in dataloader:
        out = model(loader.x, loader.edge_index, loader.batch)
        outs.append(out)
        ys.append(loader.y.flatten())
    return torch.cat(outs, dim=0), torch.cat(ys)

def train_val(model: GCN, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim, criterion: nn.CrossEntropyLoss, metrics: list[Metric], max_grad_norm: float=1.0):
    train_loss = train(model, train_loader, optimizer, criterion, max_grad_norm)
    val_loss = val(model, val_loader, criterion)
    preds = {
        "train": predict(model, train_loader),
        "val": predict(model, val_loader)
        }
    result = {"train_loss": train_loss, "val_loss": val_loss}
    
    for m in metrics:
        outs, ys = preds[m.split]
        result[m.name] = m.compute(outs, ys)

    return result
