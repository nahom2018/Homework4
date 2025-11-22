from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_sizes=(128, 256, 256, 128),
    ):
        """
        Simple but fairly strong MLP that predicts waypoints from the
        left and right track boundaries.

        Args:
            n_track (int): number of points on each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_sizes (tuple): sizes of hidden layers
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = 2 * n_track * 2  # left + right, each with (x, y)
        output_dim = n_waypoints * 2

        layers = []
        dim_in = input_dim
        for dim_out in hidden_sizes:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate left and right boundaries along the "point" dimension
        x = torch.cat([track_left, track_right], dim=1)  # (B, 2*n_track, 2)

        # Flatten per sample
        B = x.shape[0]
        x = x.view(B, -1)  # (B, 2*n_track*2)

        # MLP prediction
        out = self.mlp(x)  # (B, n_waypoints*2)

        # Reshape back to (B, n_waypoints, 2)
        out = out.view(B, self.n_waypoints, 2)
        return out


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        Transformer planner that uses learned waypoint queries (Perceiver-style)
        attending over encoded lane boundary points.
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Encode 2D boundary points into d_model
        self.input_proj = nn.Linear(2, d_model)

        # Learned latent queries: one per waypoint
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Positional embeddings for lane points (left+right -> 2 * n_track tokens)
        self.track_pos_embed = nn.Embedding(2 * n_track, d_model)

        # Transformer decoder (cross-attention over encoded lane features)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # we use (S, B, E) as expected by default
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.layernorm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate left and right track points => (B, 2*n_track, 2)
        x = torch.cat([track_left, track_right], dim=1)
        B, num_points, _ = x.shape  # num_points == 2 * n_track

        # Project 2D coordinates into d_model
        x = self.input_proj(x)  # (B, 2*n_track, d_model)

        # Add learned positional embeddings for the track points
        # pos indices: 0 .. (2*n_track - 1)
        pos_idx = torch.arange(num_points, device=x.device)
        pos_emb = self.track_pos_embed(pos_idx)[None, :, :]  # (1, 2*n_track, d_model)
        x = x + pos_emb  # (B, 2*n_track, d_model)

        # Transformer expects (S, B, E)
        memory = x.transpose(0, 1)  # (2*n_track, B, d_model)

        # Prepare learned waypoint queries as target sequence (T, B, E)
        # query_embed.weight: (n_waypoints, d_model)
        queries = self.query_embed.weight  # (n_waypoints, d_model)
        tgt = queries.unsqueeze(1).repeat(1, B, 1)  # (n_waypoints, B, d_model)

        # Cross-attention: queries attend over "memory" (lane boundaries)
        out = self.transformer_decoder(tgt=tgt, memory=memory)  # (n_waypoints, B, d_model)

        # Back to (B, n_waypoints, d_model)
        out = out.transpose(0, 1)  # (B, n_waypoints, d_model)
        out = self.layernorm(out)

        # Predict 2D waypoint coordinates for each latent
        waypoints = self.output_proj(out)  # (B, n_waypoints, 2)
        return waypoints


class ResidualBlock(nn.Module):
    """
    Basic ResNet-style residual block used in CNNPlanner.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        base_channels: int = 32,
    ):
        """
        CNN-based planner that predicts waypoints directly from the RGB image.

        Args:
            n_waypoints (int): number of waypoints to predict
            base_channels (int): base number of channels in the CNN backbone
        """
        super().__init__()

        self.n_waypoints = n_waypoints

        # Normalization stats for input images
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # ResNet-ish backbone tailored for 96x128 images
        # Output feature map is pooled to a global vector
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),

            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2),

            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4, base_channels * 4),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        cnn_out_dim = base_channels * 4
        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)
        """
        x = image

        # Normalize to zero mean / unit variance per channel
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # CNN backbone
        x = self.features(x)              # (B, C, H', W')
        x = self.global_pool(x)           # (B, C, 1, 1)
        x = x.flatten(1)                  # (B, C)

        # Fully connected head to predict flattened waypoints
        x = self.head(x)                  # (B, n_waypoints*2)

        # Reshape to (B, n_waypoints, 2)
        x = x.view(-1, self.n_waypoints, 2)
        return x


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
