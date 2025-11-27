import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Fix Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from homework.datasets.road_dataset import load_data
from models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model


def get_model(name):
    if name == "mlp_planner":
        return MLPPlanner(n_track=10, n_waypoints=3)
    elif name == "transformer_planner":
        return TransformerPlanner(n_track=10, n_waypoints=3)
    elif name == "cnn_planner":
        return CNNPlanner(n_waypoints=3)
    else:
        raise ValueError(f"Unknown model {name}")


def masked_l1_loss(pred, target, mask):
    mask = mask.unsqueeze(-1)     # (B,3,1)
    return torch.sum(mask * torch.abs(pred - target)) / torch.sum(mask)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    is_cnn = isinstance(model, CNNPlanner)

    for batch in loader:
        optimizer.zero_grad()

        if is_cnn:
            preds = model(batch["image"].to(device))
        else:
            preds = model(
                track_left=batch["track_left"].to(device),
                track_right=batch["track_right"].to(device),
            )

        target = batch["waypoints"].to(device)
        mask = batch["waypoints_mask"].to(device)

        # Masked L1 loss
        mask = mask.unsqueeze(-1)
        loss = torch.sum(mask * torch.abs(preds - target)) / torch.sum(mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print(f"\nTraining {args.model} for {args.epochs} epochs...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = load_data(
        dataset_path=os.path.join(PARENT_DIR, "drive_data/train"),
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = load_data(
        dataset_path=os.path.join(PARENT_DIR, "drive_data/val"),
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = get_model(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}")

    path = save_model(model)
    print("\nMODEL SAVED AT:", path)
    print("Training completed.\n")


if __name__ == "__main__":
    main()
