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

from homework.datasets.road_dataset import load_planner_data, load_image_data




def get_model(name):
    if name == "mlp_planner":
        return MLPPlanner()
    elif name == "transformer_planner":
        return TransformerPlanner()
    elif name == "cnn_planner":
        return CNNPlanner()
    else:
        raise ValueError(f"Unknown model {name}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    use_cnn = isinstance(model, CNNPlanner)

    for batch in loader:
        optimizer.zero_grad()

        if use_cnn:
            preds = model(batch["image"].to(device))
        else:
            preds = model(
                track_left=batch["track_left"].to(device),
                track_right=batch["track_right"].to(device),
            )

        loss = criterion(preds, batch["waypoints"].to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print(f"\nTraining {args.model} for {args.epochs} epochs...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    if args.model in ["mlp_planner", "transformer_planner"]:
        # Use PLANNER loader (EgoTrackProcessor)
        train_loader = load_planner_data(
            dataset_path="../drive_data/train",
            batch_size=args.batch_size,
            shuffle=True,
        )

        val_loader = load_planner_data(
            dataset_path="../drive_data/val",
            batch_size=args.batch_size,
            shuffle=False,
        )

    elif args.model == "cnn_planner":
        # Use IMAGE loader (ImageLoader + normalization)
        train_loader = load_image_data(
            dataset_path="../drive_data/train",
            batch_size=args.batch_size,
            shuffle=True,
        )

        val_loader = load_image_data(
            dataset_path="../drive_data/val",
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        raise ValueError(f"Unknown model {args.model}")

    # Model setup
    model = get_model(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}")

    # Save model
    save_model(model)
    print("\nModel saved successfully!\n")


if __name__ == "__main__":
    main()