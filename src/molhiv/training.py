import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from molhiv.model import GCN
from tqdm import tqdm
from molhiv.utils import Metric

<<<<<<< HEAD
def train(model: GCN, loader: DataLoader, optimizer:torch.optim, criterion: nn.CrossEntropyLoss, max_grad_norm: float = 1.0):
=======
def train(model: GCN, loader: DataLoader, optimizer:torch.optim, criterion: nn.CrossEntropyLoss, max_grad_norm: float = 1.0, device: str=None):
>>>>>>> 24945ca (Add core model, training functions, and utility methods for graph property prediction)
    model.train()
    loss = 0
    total_sample = 0
    for batch in loader:
        optimizer.zero_grad()
<<<<<<< HEAD
        out = model(batch.x, batch.edge_index, batch.batch)
        l = criterion(out, batch.y.flatten())
=======
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y.flatten().to(device)
        l = criterion(out, y)
>>>>>>> 24945ca (Add core model, training functions, and utility methods for graph property prediction)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        loss += (l.item() * batch.y.shape[0])
        total_sample += batch.y.shape[0]
    loss /= total_sample
    return loss

@torch.no_grad()
<<<<<<< HEAD
def val(model: GCN, loader: DataLoader, criterion: nn.CrossEntropyLoss):
=======
def val(model: GCN, loader: DataLoader, criterion: nn.CrossEntropyLoss, device: str=None):
>>>>>>> 24945ca (Add core model, training functions, and utility methods for graph property prediction)
    loss = 0
    total_sample = 0
    model.eval()
    for batch in loader:
<<<<<<< HEAD
        out = model(batch.x, batch.edge_index, batch.batch)
        l = criterion(out, batch.y.flatten())
=======
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y.flatten().to(device)
        l = criterion(out, y)
>>>>>>> 24945ca (Add core model, training functions, and utility methods for graph property prediction)
        loss += (l.item()*batch.y.shape[0])
        total_sample += batch.y.shape[0]
    loss /= total_sample
    return loss

@torch.no_grad()
<<<<<<< HEAD
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
=======
def predict(model: GCN, dataloader: DataLoader, device:str = None):
    model.eval()
    outs = []
    ys = []
    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        outs.append(out)
        y = batch.y.flatten().to(device)
        ys.append(y)
    return torch.cat(outs, dim=0), torch.cat(ys)

def train_val(model: GCN, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim, criterion: nn.CrossEntropyLoss, metrics: list[Metric], max_grad_norm: float=1.0, device: str=None):
    model = model.to(device)
    train_loss = train(model, train_loader, optimizer, criterion, max_grad_norm, device)
    val_loss = val(model, val_loader, criterion, device)
>>>>>>> 24945ca (Add core model, training functions, and utility methods for graph property prediction)
    preds = {
        "train": predict(model, train_loader),
        "val": predict(model, val_loader)
        }
    result = {"train_loss": train_loss, "val_loss": val_loss}
    
    for m in metrics:
        outs, ys = preds[m.split]
        result[m.name] = m.compute(outs, ys)

    return result
