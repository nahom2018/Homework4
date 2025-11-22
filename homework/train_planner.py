import sys
import os

# Add parent directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.datasets.road_dataset import RoadDataset
from models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model



def get_model(name):
    if name == "mlp_planner":
        return MLPPlanner()
    elif name == "transformer_planner":
        return TransformerPlanner()
    elif name == "cnn_planner":
        return CNNPlanner()
    else:
        raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        if "image" in batch:
            # CNN Planner
            pred = model(batch["image"].to(device))
        else:
            # MLP or Transformer Planner
            pred = model(
                track_left=batch["track_left"].to(device),
                track_right=batch["track_right"].to(device),
            )

        loss = criterion(pred, batch["waypoints"].to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


from homework.datasets.road_dataset import load_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Training {args.model} for {args.epochs} epochs...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load training data
    train_loader = load_data(
        dataset_path="../drive_data/train",
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Load validation data (optional)
    val_loader = load_data(
        dataset_path="../drive_data/val",
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=False,
    )



if __name__ == "__main__":
    main()
