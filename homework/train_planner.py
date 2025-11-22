import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.datasets.road_dataset import RoadDataset
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Training {args.model} for {args.epochs} epochs...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_ds = RoadDataset(split="train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Build model
    model = get_model(args.model).to(device)

    # Optimizer / Loss
    criterion = nn.L1Loss()  # Smooth L1 or L1 works best for waypoint regression
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")

    # Save weights as required by grader
    save_model(model)
    print(f"Model saved!")


if __name__ == "__main__":
    main()
