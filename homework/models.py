from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(self, n_track=10, n_waypoints=3, hidden_sizes=(128,128,128)):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = 2 * n_track * 2      # (left+right) × (x,z)
        output_dim = n_waypoints * 2

        layers = []
        dim_in = input_dim
        for dim_out in hidden_sizes:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU(inplace=True))
            dim_in = dim_out

        layers.append(nn.Linear(dim_in, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, track_left, track_right, **kwargs):
        # Concatenate boundary points → shape (B, 20, 2)
        x = torch.cat([track_left, track_right], dim=1)

        # Flatten → shape (B, 40)
        B = x.shape[0]
        x = x.view(B, -1)

        # (B, 6)
        out = self.mlp(x)

        # reshape to (B, 3, 2)
        return out.view(B, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 96,               # higher capacity
        nhead: int = 4,
        num_layers: int = 1,             # 1 layer is more stable
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        FINAL version: adds geometry deltas + left/right/center tags + pre-LN.
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Each point: (x, z, dx, dz, left, right, center) = 7 dimensions
        self.point_proj = nn.Linear(7, d_model)

        # Positional embedding for 3T tokens
        self.pos_embed = nn.Embedding(3 * n_track, d_model)

        # Waypoint queries
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Pre-LN TransformerDecoderLayer (much more stable)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # ★ KEY FIX
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(d_model)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, track_left, track_right, **kwargs):
        B, T, C = track_left.shape
        device = track_left.device
        dtype = track_left.dtype

        # Compute centerline
        center = 0.5 * (track_left + track_right)

        # Compute deltas (local geometry)
        def deltas(t):
            dt = torch.zeros_like(t)
            dt[:, 1:] = t[:, 1:] - t[:, :-1]
            return dt

        left_d = deltas(track_left)
        right_d = deltas(track_right)
        center_d = deltas(center)

        # Type tags
        left_tag = torch.tensor([1,0,0], device=device, dtype=dtype).view(1,1,3).expand(B, T, 3)
        right_tag = torch.tensor([0,1,0], device=device, dtype=dtype).view(1,1,3).expand(B, T, 3)
        center_tag = torch.tensor([0,0,1], device=device, dtype=dtype).view(1,1,3).expand(B, T, 3)

        # Build tokens for each lane
        L = torch.cat([track_left, left_d, left_tag], dim=-1)
        R = torch.cat([track_right, right_d, right_tag], dim=-1)
        C = torch.cat([center, center_d, center_tag], dim=-1)

        # Sequence: L + R + C
        tokens = torch.cat([L, R, C], dim=1)   # (B, 3T, 7)

        x = self.point_proj(tokens)
        x = F.dropout(x, p=0.1, training=self.training)
        x = x + self.pos_embed(idx)[None, :, :]
        x = F.layer_norm(x, (self.d_model,))

        # Queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = F.layer_norm(queries, (self.d_model,))

        # Decoder
        out = self.decoder(queries, x)
        out = self.final_norm(out)

        return self.output_head(out)


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