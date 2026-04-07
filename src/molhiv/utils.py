
import torch
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, field
from typing import Callable

def download_graph_prop_pred_dataset(root="../data"):
    # Patch torch.load to use weights_only=False (safe for OGB's trusted data)
    _original_load = torch.load

    torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})
    dataset = PygGraphPropPredDataset(root=root, name="ogbg-molhiv")

    # Restore original behavior
    torch.load = _original_load
    return dataset

def calculate_label_imbalance(y: torch.Tensor):
    _, counts = np.unique(y, return_counts=True)
    class_weights = 1.0/(torch.Tensor(counts))
    class_weights = class_weights/class_weights.sum()
    sample_weights = class_weights[y]
    return class_weights, sample_weights

def acc(out:torch.Tensor, y: torch.Tensor):
    out = torch.argmax(out, dim=1)
    return (out == y).float().mean().item()

def prec(out: torch.Tensor, y: torch.Tensor):
    preds = torch.argmax(out, dim=1)
    pred_pos_mask = preds == 1
    if pred_pos_mask.float().sum() > 0:
        return (preds==y)[pred_pos_mask].float().mean().item()
    else:
        return 0

def rec(out: torch.Tensor, y: torch.Tensor):
    preds = torch.argmax(out, dim=1)
    pos_classes = y == 1
    return (preds[pos_classes] == 1).float().mean().item()

def roc_auc(outs: torch.Tensor, ys: torch.Tensor):
    y_score = torch.softmax(outs, dim=1)[:, 1]
    y_score = y_score.numpy()
    ys = ys.numpy()
    return roc_auc_score(ys, y_score)

@dataclass
class Metric:
    name: str
    fn: Callable
    split: str
    history: list = field(default_factory=list)

    def compute(self, outs: torch.Tensor, ys: torch.Tensor):
        value = self.fn(outs, ys)
        self.history.append(value)
        return value

