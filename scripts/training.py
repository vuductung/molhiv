import torch
from molhiv.utils import calculate_label_imbalance, prec, acc, rec, roc_auc, download_graph_prop_pred_dataset
from molhiv.model import GATNN
from molhiv.training import Metric, train_val
import torch.nn as nn
import os
import numpy as np
from molhiv.training import predict

params = {
    "batch_size": 64,
    "datasize": 99999,
    "in_channel": 32,
    "hidden_channel": 256,
    "out_channel": 1,
    "dropout": 0.5,
    "lr": 0.001,
    "weight_decay": 0.001,
    "epochs": 50,
    "datasetname": "ogbg-molhiv",
    "Conv": "GATConv",
    "n_heads": 4,
    "add_self_loops": True,
    "max_grad_norm":1,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
dataset = download_graph_prop_pred_dataset()

split_idx = dataset.get_idx_split()

from torch_geometric.loader import DataLoader
size = params["datasize"]

train_dataset = dataset[split_idx["train"]][:size]
val_dataset = dataset[split_idx["valid"]][:size]
test_dataset = dataset[split_idx["test"]]

sample_labels = [dataset[i].y.squeeze() for i in range(len(train_dataset))]
class_weights, sample_weights = calculate_label_imbalance(sample_labels)
class_weights = class_weights.to(device)

from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])

model = GATNN(
    params["in_channel"],
    params["hidden_channel"],
    params["out_channel"],
    params["n_heads"],
    params["dropout"],
    params["add_self_loops"]
)
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

metrics = [
    Metric("train_prec", fn=prec, split="train"),
    Metric("val_prec", fn=prec, split="val"),

    Metric("train_acc", fn=acc, split="train"),
    Metric("val_acc", fn=acc, split="val"),

    Metric("train_rec", fn=rec, split="train"),
    Metric("val_rec", fn=rec, split="val"),

    Metric("train_roc_auc", fn=roc_auc, split="train"),
    Metric("val_roc_auc", fn=roc_auc, split="val")
]
neg_counts, pos_counts = np.unique(sample_labels, return_counts=True)[1]
pos_weight = torch.tensor([neg_counts / pos_counts], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factore=0.5, patience=5, min_lr=1e-6
)

import mlflow
mlflow.set_tracking_uri(f"file://{os.path.expanduser('~')}/projects/molhiv/mlruns")
mlflow.set_experiment("Molhiv-GCN-HIV-binding")
with mlflow.start_run(run_name="Training-GAT-GPUDEV-BCELoss"):
    mlflow.log_params(params)
    for epoch in range(params["epochs"]):
        results = train_val(model, train_loader, val_loader, optimizer, criterion, metrics, params["max_grad_norm"], device)
        mlflow.log_metrics(results, step=epoch)
    
    scheduler.step(results["val_loss"])
    mlflow.log_param("lr_per_epoch", scheduler.optimizer.param_groups[0]["lr"], step="epoch")

    prob, y_true = predict(model, test_loader, "cpu")

    test_roc_auc = roc_auc(prob[:, 1], y_true)
    mlflow.log_metric("test_roc_auc", test_roc_auc)
