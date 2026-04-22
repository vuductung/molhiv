import torch
from molhiv.utils import calculate_label_imbalance, prec, acc, rec, roc_auc, download_graph_prop_pred_dataset
from molhiv.ginenn import GINENN
from molhiv.training import Metric, train_val
import torch.nn as nn
import numpy as np
from molhiv.training import predict
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="Model training")
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--datasize", type=int, default=None)
parser.add_argument("--model_name", type=str)
args = parser.parse_args()


SCRIPT_DIR = Path(__file__).resolve().parent
conifg_path = SCRIPT_DIR / "../configs/molhiv.yaml"
with open(conifg_path) as f:
    cfg = yaml.safe_load(f)

if args.epochs is not None:
    cfg["training"]["epochs"] = args.epochs
if args.datasize is not None:
    cfg["data"]["datasize"] = args.datasize

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
dataset = download_graph_prop_pred_dataset()

split_idx = dataset.get_idx_split()

from torch_geometric.loader import DataLoader
size = cfg["data"]["datasize"]

train_dataset = dataset[split_idx["train"]][:size]
val_dataset = dataset[split_idx["valid"]]
test_dataset = dataset[split_idx["test"]]

sample_labels = [train_dataset[i].y.squeeze() for i in range(len(train_dataset))]
class_weights, sample_weights = calculate_label_imbalance(sample_labels)
class_weights = class_weights.to(device)

from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=cfg["data"]["batch_size"], sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=cfg["data"]["batch_size"])

model = GINENN(**cfg["model"])
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])

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
criterion = nn.BCEWithLogitsLoss(pos_weight=None)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, **cfg["reduce_on_plateau_scheduler"]
)

import mlflow
if device == "cuda":
    mlflow.set_tracking_uri(f"file://{os.path.expanduser('~')}/projects/molhiv/mlruns")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

checkpoint_path = SCRIPT_DIR / f"../checkpoints/{args.model_name}.pt"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

mlflow.set_experiment("Molhiv-GCN-HIV-binding")
with mlflow.start_run(run_name=args.run_name):
    mlflow.log_params(cfg["data"])
    mlflow.log_params(cfg["model"])
    mlflow.log_params(cfg["optimizer"])
    mlflow.log_params(cfg["training"])

    best_val_roc_auc = 0
    patience_counter = 0

    for epoch in range(cfg["training"]["epochs"]):
        results = train_val(model, train_loader, val_loader, optimizer, criterion, metrics, cfg["training"]["max_grad_norm"], device)
        mlflow.log_metrics(results, step=epoch)
    
        scheduler.step(results["val_loss"])
        mlflow.log_metric("lr_per_epoch", scheduler.optimizer.param_groups[0]["lr"], step=epoch)

        if results["val_roc_auc"] > best_val_roc_auc:
            best_val_roc_auc = results["val_roc_auc"]
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_roc_auc": results["val_roc_auc"],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
                "config": cfg,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter == cfg["training"]["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    prob, y_true = predict(model, test_loader, device)
    test_roc_auc = roc_auc(prob, y_true)
    mlflow.log_metric("test_roc_auc", test_roc_auc)
