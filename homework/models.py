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
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        """
        Transformer planner using learned waypoint queries that attend
        over encoded lane boundary points (left + right + center).

        Each lane token is encoded as:
            [coord0, coord1, is_left, is_right, is_center]  (5D)

        This is axis-agnostic: it doesn't matter whether coords are (x,y) or (x,z).
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # We encode each point as 2 coords + 3 type flags = 5D
        self.point_proj = nn.Linear(5, d_model)

        # Positional embeddings for all tokens: left + right + center = 3 * n_track
        self.track_pos_embed = nn.Embedding(3 * n_track, d_model)

        # Learned queries (one per waypoint)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Decoder: queries attend to lane memory (Perceiver-style cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, S, E)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        # MLP head to map from latent to 2D waypoint coordinates
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            track_left:  (B, n_track, 2)
            track_right: (B, n_track, 2)

        Returns:
            waypoints: (B, n_waypoints, 2)
        """
        B, T, C = track_left.shape  # T = n_track, C = 2
        device = track_left.device
        dtype = track_left.dtype

        # Centerline between left and right
        center = 0.5 * (track_left + track_right)  # (B, T, 2)

        # Type encodings: [is_left, is_right, is_center]
        left_type = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 3).expand(B, T, 3)
        right_type = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 1, 3).expand(B, T, 3)
        center_type = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 3).expand(B, T, 3)

        # Build tokens: (coords + type flags)
        left_tokens = torch.cat([track_left, left_type], dim=-1)    # (B, T, 5)
        right_tokens = torch.cat([track_right, right_type], dim=-1) # (B, T, 5)
        center_tokens = torch.cat([center, center_type], dim=-1)    # (B, T, 5)

        # Concatenate along sequence dimension: (B, 3T, 5)
        tokens = torch.cat([left_tokens, right_tokens, center_tokens], dim=1)

        # Project to d_model
        x = self.point_proj(tokens)  # (B, 3T, d_model)

        # Positional embeddings for lane tokens
        seq_len = 3 * T
        pos_idx = torch.arange(seq_len, device=device)
        pos_emb = self.track_pos_embed(pos_idx)[None, :, :]  # (1, 3T, d_model)
        memory = x + pos_emb  # (B, 3T, d_model)

        # Learned queries (waypoint latents)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, n_waypoints, d_model)

        # Decoder: queries attend over memory
        out = self.decoder(tgt=queries, memory=memory)  # (B, n_waypoints, d_model)
        out = self.norm(out)

        # Predict (coord0, coord1) for each waypoint
        waypoints = self.output_head(out)  # (B, n_waypoints, 2)
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
