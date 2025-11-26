#!/usr/bin/env python
# coding: utf-8

import os
import math
import time
from dataclasses import dataclass
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import imageio.v2 as imageio

import results

# Optional: scikit-fem for H1-smoothness penalty
try:
    import skfem as fem  # noqa: F401
    from skfem.helpers import dot, grad  # noqa: F401
    HAVE_SKFEM = True
except Exception:
    HAVE_SKFEM = False

try:
    from pysensors.reconstruction import SSPOR
    from pysensors.basis import SVD
    from pysensors.optimizers import QR
    HAVE_PYSENSORS = True
except Exception:
    HAVE_PYSENSORS = False


torch.set_float32_matmul_precision('high') if hasattr(torch, "set_float32_matmul_precision") else None
try:
    torch.set_num_threads(1)
except AttributeError:
    pass

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

FLOW_COMPONENTS = ("pressure", "velocity_x", "velocity_y")
DEFAULT_SIGMA = (0.02, 0.01, 0.01)
DEFAULT_TIME_WINDOW = (0.0, 4.0)


@dataclass
class FlowScenario:
    centers: np.ndarray           # (K, 2)
    velocities: np.ndarray        # (K, 2)
    amplitudes: np.ndarray        # (K,)
    widths: np.ndarray            # (K,)
    velocity_scale: float
    base_flow: np.ndarray         # (2,)
    background_gradient: np.ndarray  # (2,)
    base_level: float
    temporal_trend: float
    shear_strength: float
    time_window: tuple[float, float]


@dataclass
class PrecomputedDataset:
    fields: np.ndarray
    grid_x: np.ndarray | None
    grid_y: np.ndarray | None
    times: np.ndarray | None
    channel_names: tuple[str, ...]
    source_path: str


def make_grid(N: int) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.linspace(0.0, 1.0, N, dtype=np.float32)
    x1, x2 = np.meshgrid(x, x)
    spacing = float(x[1] - x[0]) if N > 1 else 1.0
    return x1.astype(np.float32), x2.astype(np.float32), spacing


def load_precomputed_dataset(path: str) -> PrecomputedDataset:
    with np.load(path) as data:
        fields = data["fields"].astype(np.float32)
        if fields.ndim == 3:
            fields = fields[:, np.newaxis, ...]
        grid_x = data.get("grid_x")
        grid_y = data.get("grid_y")
        times = data.get("times")
        if grid_x is not None:
            grid_x = grid_x.astype(np.float32)
        if grid_y is not None:
            grid_y = grid_y.astype(np.float32)
        if times is not None:
            times = times.astype(np.float32)
        base_name = "field"
        if "field_name" in data:
            base_name = str(data["field_name"])
        include_gradients = bool(int(data.get("include_gradients", 0)))
    channel_count = fields.shape[1]
    if include_gradients and channel_count >= 3:
        channel_names = (
            base_name,
            f"d{base_name}/dx",
            f"d{base_name}/dy",
        )
    elif channel_count == 1:
        channel_names = (base_name,)
    else:
        channel_names = tuple(f"{base_name}_{i}" for i in range(channel_count))
    return PrecomputedDataset(
        fields=fields,
        grid_x=grid_x,
        grid_y=grid_y,
        times=times,
        channel_names=channel_names,
        source_path=path,
    )


def parse_sigma(env_value: str | None, default: Sequence[float], channels: int) -> tuple[float, ...]:
    if env_value is None or env_value.strip() == "":
        values = list(default)
    else:
        values = [float(x.strip()) for x in env_value.split(",") if x.strip()]
    if len(values) == 1 and channels > 1:
        values = values * channels
    if len(values) != channels:
        raise ValueError(f"Sigma specification length {len(values)} does not match channel count {channels}.")
    return tuple(values)


def sample_flow_scenario(
    rng: np.random.Generator,
    num_sources_range: tuple[int, int] = (2, 4),
    time_window: tuple[float, float] = DEFAULT_TIME_WINDOW,
) -> FlowScenario:
    num_sources = int(rng.integers(num_sources_range[0], num_sources_range[1] + 1))
    centers = rng.uniform(0.2, 0.8, size=(num_sources, 2)).astype(np.float32)
    velocities = rng.uniform(-0.25, 0.25, size=(num_sources, 2)).astype(np.float32)
    amplitudes = rng.uniform(0.4, 1.2, size=num_sources).astype(np.float32)
    widths = rng.uniform(0.05, 0.15, size=num_sources).astype(np.float32)
    velocity_scale = float(rng.uniform(0.02, 0.08))
    base_flow = rng.uniform(-0.08, 0.08, size=2).astype(np.float32)
    background_gradient = rng.uniform(-0.25, 0.25, size=2).astype(np.float32)
    base_level = float(rng.uniform(-0.2, 0.2))
    temporal_trend = float(rng.uniform(-0.6, 0.6))
    shear_strength = float(rng.uniform(-0.3, 0.3))
    return FlowScenario(
        centers=centers,
        velocities=velocities,
        amplitudes=amplitudes,
        widths=widths,
        velocity_scale=velocity_scale,
        base_flow=base_flow,
        background_gradient=background_gradient,
        base_level=base_level,
        temporal_trend=temporal_trend,
        shear_strength=shear_strength,
        time_window=time_window,
    )


def _periodic_delta(grid: np.ndarray, center: float) -> np.ndarray:
    delta = grid - center
    return (delta + 0.5).astype(np.float32) % 1.0 - 0.5


def evaluate_flow_field(
    grid: tuple[np.ndarray, np.ndarray, float],
    scenario: FlowScenario,
    t: float,
) -> np.ndarray:
    x1, x2, spacing = grid
    pressure = np.zeros_like(x1, dtype=np.float32)
    t0, t1 = scenario.time_window
    t_norm = 0.0 if t1 == t0 else (t - t0) / max(t1 - t0, 1e-6)
    amp_scale = 1.0 + scenario.temporal_trend * (t_norm - 0.5)

    for idx in range(scenario.centers.shape[0]):
        center = scenario.centers[idx] + scenario.velocities[idx] * t
        center = np.mod(center, 1.0).astype(np.float32)
        dx = _periodic_delta(x1, float(center[0]))
        dy = _periodic_delta(x2, float(center[1]))
        r2 = dx * dx + dy * dy
        width = max(float(scenario.widths[idx]), 1e-3)
        gaussian = np.exp(-0.5 * (r2 / (width ** 2))).astype(np.float32)
        pressure += scenario.amplitudes[idx] * amp_scale * gaussian
        pressure += scenario.shear_strength * dx * dy * gaussian

    pressure += scenario.base_level
    pressure += scenario.background_gradient[0] * (x1 - 0.5)
    pressure += scenario.background_gradient[1] * (x2 - 0.5)

    dp_dy, dp_dx = np.gradient(pressure, spacing, edge_order=2)
    velocity_x = scenario.base_flow[0] - scenario.velocity_scale * dp_dx
    velocity_y = scenario.base_flow[1] - scenario.velocity_scale * dp_dy

    fields = np.stack([pressure, velocity_x, velocity_y], axis=0).astype(np.float32)
    return fields


def gappy_sample_flow(
    fields: np.ndarray,
    M: int,
    sigma: Sequence[float],
    rng: np.random.Generator,
    obs_idx: np.ndarray | None = None,
):
    channels, N, _ = fields.shape
    if obs_idx is None:
        obs_idx_flat = rng.choice(N * N, size=M, replace=False)
        obs_idx = np.array(np.unravel_index(obs_idx_flat, (N, N))).T
    else:
        obs_idx = obs_idx.astype(np.int64)
        obs_idx_flat = np.ravel_multi_index(obs_idx.T, (N, N))

    mask = np.zeros((N, N), dtype=np.float32)
    mask_flat = mask.reshape(-1)
    mask_flat[obs_idx_flat] = 1.0

    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.size == 1:
        sigma = np.repeat(sigma, channels)

    y_obs = np.zeros_like(fields, dtype=np.float32)
    for ch in range(channels):
        noise = rng.normal(0.0, float(sigma[ch]), size=M).astype(np.float32)
        channel_flat = fields[ch].reshape(-1)
        observed = channel_flat[obs_idx_flat] + noise
        y_obs[ch].reshape(-1)[obs_idx_flat] = observed.astype(np.float32)

    return y_obs, mask, obs_idx


def _compress_fields_for_sensor_selection(fields: np.ndarray, mode: str = "rss") -> np.ndarray:
    if mode == "pressure":
        return fields[0]
    if mode == "mean":
        return fields.mean(axis=0)
    if mode == "rss":
        return np.sqrt(np.sum(fields**2, axis=0))
    raise ValueError(f"Unknown aggregation mode '{mode}'.")


def _collect_sensor_snapshots(
    N: int,
    num_samples: int,
    seed: int,
    num_sources_range: tuple[int, int],
    time_window: tuple[float, float],
    aggregation_mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = make_grid(N)
    snapshots = np.zeros((num_samples, N * N), dtype=np.float32)
    for i in range(num_samples):
        scenario = sample_flow_scenario(rng, num_sources_range=num_sources_range, time_window=time_window)
        t = rng.uniform(time_window[0], time_window[1])
        fields = evaluate_flow_field(grid, scenario, t)
        compressed = _compress_fields_for_sensor_selection(fields, aggregation_mode)
        snapshots[i] = compressed.reshape(-1)
    return snapshots


def optimize_sensor_layout(
    N: int,
    candidate_sensor_counts: Sequence[int],
    num_train_snapshots: int,
    num_val_snapshots: int,
    aggregation_mode: str,
    n_basis_modes: int,
    seed: int,
    num_sources_range: tuple[int, int],
    time_window: tuple[float, float],
    precomputed_fields: np.ndarray | None = None,
):
    if not HAVE_PYSENSORS:
        raise RuntimeError("pysensors is required for sensor optimization but is not available.")
    counts = sorted({int(c) for c in candidate_sensor_counts if int(c) > 0})
    if not counts:
        raise ValueError("candidate_sensor_counts must contain positive integers.")
    if precomputed_fields is None:
        train_snapshots = _collect_sensor_snapshots(
            N=N,
            num_samples=num_train_snapshots,
            seed=seed,
            num_sources_range=num_sources_range,
            time_window=time_window,
            aggregation_mode=aggregation_mode,
        ).astype(np.float64)
        val_snapshots = _collect_sensor_snapshots(
            N=N,
            num_samples=num_val_snapshots,
            seed=seed + 1,
            num_sources_range=num_sources_range,
            time_window=time_window,
            aggregation_mode=aggregation_mode,
        ).astype(np.float64)
    else:
        if precomputed_fields.ndim != 4 or precomputed_fields.shape[2] != N or precomputed_fields.shape[3] != N:
            raise ValueError("precomputed_fields must have shape (samples, channels, N, N).")
        rng = np.random.default_rng(seed)
        def draw_snapshots(num_samples: int) -> np.ndarray:
            snaps = np.zeros((num_samples, N * N), dtype=np.float32)
            for i in range(num_samples):
                idx = rng.integers(0, precomputed_fields.shape[0])
                fields = precomputed_fields[idx]
                compressed = _compress_fields_for_sensor_selection(fields, aggregation_mode)
                snaps[i] = compressed.reshape(-1)
            return snaps
        train_snapshots = draw_snapshots(num_train_snapshots).astype(np.float64)
        rng = np.random.default_rng(seed + 1)
        val_snapshots = draw_snapshots(num_val_snapshots).astype(np.float64)
    mean_snapshot = train_snapshots.mean(axis=0, keepdims=True)
    train_centered = train_snapshots - mean_snapshot
    val_centered = val_snapshots - mean_snapshot
    max_basis = min(n_basis_modes, train_centered.shape[0], train_centered.shape[1])
    basis = SVD(n_basis_modes=max_basis)
    optimizer = QR()
    model = SSPOR(basis=basis, optimizer=optimizer)
    model.fit(train_centered, quiet=True)
    ranking = model.get_all_sensors()
    valid_counts = np.array([c for c in counts if c <= ranking.size], dtype=int)
    if valid_counts.size == 0:
        raise ValueError(f"No valid sensor counts. Max available sensors: {ranking.size}.")
    model.set_n_sensors(int(valid_counts.max()))
    errors = model.reconstruction_error(val_centered, sensor_range=valid_counts)
    best_idx = int(np.argmin(errors))
    best_count = int(valid_counts[best_idx])
    selected_flat = ranking[:best_count]
    sensors = np.column_stack(np.unravel_index(selected_flat, (N, N))).astype(np.int64)
    return sensors, ranking, valid_counts, errors, best_count


class FlowGappyDataset(Dataset):
    """Synthetic flow snapshots with sensor masks."""

    def __init__(
        self,
        N: int = 64,
        M: int = 256,
        sigma: Sequence[float] = DEFAULT_SIGMA,
        size: int = 10000,
        seed: int = 0,
        num_sources_range: tuple[int, int] = (2, 5),
        time_window: tuple[float, float] = DEFAULT_TIME_WINDOW,
        sensor_indices: np.ndarray | None = None,
    ):
        self.N = N
        self.size = size
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.num_sources_range = num_sources_range
        self.time_window = time_window
        self.rng = np.random.default_rng(seed)
        self.grid = make_grid(N)
        if sensor_indices is not None:
            sensor_indices = np.asarray(sensor_indices, dtype=np.int64)
            if sensor_indices.ndim != 2 or sensor_indices.shape[1] != 2:
                raise ValueError("sensor_indices must be of shape (M, 2)")
            self.sensor_indices = sensor_indices
            self.M = sensor_indices.shape[0]
        else:
            if M is None:
                raise ValueError("M must be provided when sensor_indices is None")
            self.sensor_indices = None
            self.M = int(M)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):  # noqa: ARG002
        t = self.rng.uniform(self.time_window[0], self.time_window[1])
        scenario = sample_flow_scenario(self.rng, self.num_sources_range, self.time_window)
        fields = evaluate_flow_field(self.grid, scenario, t)
        if self.sensor_indices is not None:
            y_obs, mask, _ = gappy_sample_flow(fields, self.M, self.sigma, self.rng, obs_idx=self.sensor_indices)
        else:
            y_obs, mask, _ = gappy_sample_flow(fields, self.M, self.sigma, self.rng)
        mask_channels = np.broadcast_to(mask, (fields.shape[0], self.N, self.N)).astype(np.float32)
        inp = np.concatenate([y_obs, mask_channels], axis=0).astype(np.float32)
        return (
            torch.from_numpy(inp),
            torch.from_numpy(fields),
            torch.from_numpy(mask.astype(np.float32)),
        )


# quick sanity
_tmp = FlowGappyDataset(N=64, M=256, sigma=DEFAULT_SIGMA, size=8, seed=SEED)
_tmp_inp, _tmp_y, _tmp_mask = _tmp[0]
logger.info(
    f"Flow dataset sample shapes — inp: {tuple(_tmp_inp.shape)}, y: {tuple(_tmp_y.shape)}, mask: {tuple(_tmp_mask.shape)}"
)


class PrecomputedFlowDataset(Dataset):
    """Dataset wrapper for precomputed MFEM fields."""

    def __init__(
        self,
        fields: np.ndarray,
        M: int,
        sigma: Sequence[float],
        size: int,
        seed: int,
        sensor_indices: np.ndarray | None = None,
    ):
        if fields.ndim != 4:
            raise ValueError("fields must have shape (samples, channels, N, N)")
        self.fields = fields.astype(np.float32)
        self.num_snapshots, self.channels, self.N, self._Ny = self.fields.shape
        if self.N != self._Ny:
            raise ValueError("Only square grids are supported for precomputed datasets.")
        self.size = int(size)
        self.sigma = np.asarray(sigma, dtype=np.float32)
        if self.sigma.size == 1:
            self.sigma = np.repeat(self.sigma, self.channels)
        if self.sigma.size != self.channels:
            raise ValueError(f"Sigma length {self.sigma.size} does not match channel count {self.channels}.")
        self.sensor_indices = None
        if sensor_indices is not None:
            sensor_indices = np.asarray(sensor_indices, dtype=np.int64)
            if sensor_indices.ndim != 2 or sensor_indices.shape[1] != 2:
                raise ValueError("sensor_indices must have shape (M, 2)")
            self.sensor_indices = sensor_indices
            self.M = sensor_indices.shape[0]
        else:
            self.M = int(M)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        field_idx = idx % self.num_snapshots
        fields = self.fields[field_idx]
        if self.sensor_indices is not None:
            y_obs, mask, _ = gappy_sample_flow(fields, self.M, self.sigma, self.rng, obs_idx=self.sensor_indices)
        else:
            y_obs, mask, _ = gappy_sample_flow(fields, self.M, self.sigma, self.rng)
        mask_channels = np.broadcast_to(mask, (self.channels, self.N, self.N)).astype(np.float32)
        inp = np.concatenate([y_obs, mask_channels], axis=0).astype(np.float32)
        return (
            torch.from_numpy(inp),
            torch.from_numpy(fields),
            torch.from_numpy(mask.astype(np.float32)),
        )


class GappyConvVAE2D(nn.Module):
    def __init__(
        self,
        N: int,
        latent_dim: int = 32,
        width: list[int] | tuple[int, ...] = (32, 64, 128),
        depth: int = 3,
        field_channels: int = 3,
    ):
        super().__init__()
        self.N = N
        self.latent_dim = latent_dim
        self.field_channels = field_channels
        in_channels = field_channels * 2
        out_channels = field_channels

        encoder_layers: list[nn.Module] = []
        current_channels = in_channels
        width = list(width)
        for i in range(depth):
            encoder_layers.append(nn.Conv2d(current_channels, width[i], kernel_size=5, stride=2, padding=2))
            encoder_layers.append(nn.ReLU())
            current_channels = width[i]
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, N, N)
            dummy_output = self.encoder(dummy_input)
            flattened_size = dummy_output.shape[1]

        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, flattened_size)

        decoder_layers: list[nn.Module] = []
        decoder_layers.append(nn.Unflatten(1, (width[-1], N // (2**depth), N // (2**depth))))
        current_channels = width[-1]
        for i in range(depth - 1, 0, -1):
            decoder_layers.append(
                nn.ConvTranspose2d(current_channels, width[i - 1], kernel_size=5, stride=2, padding=2, output_padding=1)
            )
            decoder_layers.append(nn.ReLU())
            current_channels = width[i - 1]
        decoder_layers.append(
            nn.ConvTranspose2d(current_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decode(z)
        return y_hat, mu, logvar


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_loaders(
    N: int = 64,
    M: int = 256,
    sigma: Sequence[float] = DEFAULT_SIGMA,
    train_size: int = 10000,
    val_size: int = 1000,
    bs: int = 128,
    seed: int = SEED,
    num_sources_range: tuple[int, int] = (2, 5),
    time_window: tuple[float, float] = DEFAULT_TIME_WINDOW,
    sensor_indices: np.ndarray | None = None,
    precomputed_fields: np.ndarray | None = None,
):
    dataset_M = sensor_indices.shape[0] if sensor_indices is not None else M
    if precomputed_fields is None:
        tr_dataset = FlowGappyDataset(
            N=N,
            M=dataset_M,
            sigma=sigma,
            size=train_size,
            seed=seed,
            num_sources_range=num_sources_range,
            time_window=time_window,
            sensor_indices=sensor_indices,
        )
        va_dataset = FlowGappyDataset(
            N=N,
            M=dataset_M,
            sigma=sigma,
            size=val_size,
            seed=seed + 1,
            num_sources_range=num_sources_range,
            time_window=time_window,
            sensor_indices=sensor_indices,
        )
    else:
        tr_dataset = PrecomputedFlowDataset(
            fields=precomputed_fields,
            M=dataset_M,
            sigma=sigma,
            size=train_size,
            seed=seed,
            sensor_indices=sensor_indices,
        )
        va_dataset = PrecomputedFlowDataset(
            fields=precomputed_fields,
            M=dataset_M,
            sigma=sigma,
            size=val_size,
            seed=seed + 1,
            sensor_indices=sensor_indices,
        )
    return (
        DataLoader(tr_dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=0),
        DataLoader(va_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=0),
    )


def train_gappy_conv_vae_2d(
    N: int = 64,
    M: int = 256,
    sigma: Sequence[float] = DEFAULT_SIGMA,
    epochs: int = 50,
    bs: int = 128,
    train_size: int = 10000,
    val_size: int = 1000,
    lr: float = 1e-3,
    latent_dim: int = 32,
    width: Sequence[int] = (32, 64, 128),
    depth: int = 3,
    obs_lambda: float = 1.0,
    kl_lambda: float = 1e-3,
    seed: int = SEED,
    num_sources_range: tuple[int, int] = (2, 5),
    time_window: tuple[float, float] = DEFAULT_TIME_WINDOW,
    sensor_indices: np.ndarray | None = None,
    field_channels: int = len(FLOW_COMPONENTS),
    precomputed_fields: np.ndarray | None = None,
):
    torch.manual_seed(seed)
    model = GappyConvVAE2D(
        N=N,
        latent_dim=latent_dim,
        width=list(width),
        depth=depth,
        field_channels=field_channels,
    ).to(DEVICE)
    logger.info(f"Trainable params: {count_params(model)}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader, va_loader = make_loaders(
        N,
        M,
        sigma,
        train_size,
        val_size,
        bs,
        seed,
        num_sources_range,
        time_window,
        sensor_indices=sensor_indices,
        precomputed_fields=precomputed_fields,
    )

    def run_epoch(loader, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()
        tot = 0.0
        count = 0
        with torch.set_grad_enabled(train_mode):
            for inp, y, mask in loader:
                inp = inp.to(DEVICE)
                y = y.to(DEVICE)
                mask = mask.to(DEVICE)

                y_hat, mu, logvar = model(inp)

                loss_full = nn.functional.mse_loss(y_hat, y)
                mask_expanded = mask.unsqueeze(1)
                loss_obs = nn.functional.mse_loss(y_hat * mask_expanded, y * mask_expanded)
                loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / inp.size(0)

                loss = loss_full + obs_lambda * loss_obs + kl_lambda * loss_kl

                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                tot += float(loss.detach().cpu()) * inp.size(0)
                count += inp.size(0)
        return tot / max(count, 1)

    best_val = math.inf
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = run_epoch(tr_loader, True)
        va = run_epoch(va_loader, False)
        best_val = min(best_val, va)
        if ep == 1 or ep % max(1, epochs // 10) == 0:
            logger.info(f"[ep {ep:03d}] train {tr:.6f}  val {va:.6f}  best {best_val:.6f}")
    logger.info(f"Wall-clock: {time.time()-t0:.2f}s")

    return model


def compute_flow_metrics(y_hat: np.ndarray, y_true: np.ndarray) -> dict:
    diff = y_hat - y_true
    mse_total = float(np.mean(diff ** 2))
    rmse_channels = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
    data_range = np.max(y_true) - np.min(y_true)
    rmse_total = math.sqrt(mse_total)
    psnr_global = float(20.0 * np.log10((data_range + 1e-8) / (rmse_total + 1e-8))) if data_range > 0 else float("inf")

    psnr_channels = []
    for ch in range(y_true.shape[0]):
        ch_range = np.max(y_true[ch]) - np.min(y_true[ch])
        ch_rmse = float(rmse_channels[ch])
        ch_psnr = float(20.0 * np.log10((ch_range + 1e-8) / (ch_rmse + 1e-8))) if ch_range > 0 else float("inf")
        psnr_channels.append(ch_psnr)

    return {
        "mse": mse_total,
        "rmse": rmse_total,
        "rmse_channels": rmse_channels,
        "psnr_global": psnr_global,
        "psnr_channels": psnr_channels,
    }


@torch.no_grad()
def demo_temporal(
    model: nn.Module,
    N: int = 64,
    M: int = 256,
    sigma: Sequence[float] = DEFAULT_SIGMA,
    seed: int = SEED + 999,
    outdir: str = "temporal_results/flow_demo",
    num_timesteps: int = 10,
    time_window: tuple[float, float] = DEFAULT_TIME_WINDOW,
    num_sources_range: tuple[int, int] = (2, 5),
    sensor_indices: np.ndarray | None = None,
    flow_components: Sequence[str] = FLOW_COMPONENTS,
    precomputed: PrecomputedDataset | None = None,
):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    frames = []

    if precomputed is not None:
        fields_array = precomputed.fields
        N = fields_array.shape[-1]
        times = precomputed.times if precomputed.times is not None else np.linspace(0.0, 1.0, fields_array.shape[0])
        grid_x = precomputed.grid_x
        grid_y = precomputed.grid_y
        if grid_x is not None and grid_y is not None:
            x1, x2 = np.meshgrid(grid_x.astype(np.float32), grid_y.astype(np.float32), indexing="xy")
            spacing = float(abs(grid_x[1] - grid_x[0])) if grid_x.size > 1 else 1.0
            grid = (x1, x2, spacing)
        else:
            grid = make_grid(N)
        sample_indices = np.linspace(0, fields_array.shape[0] - 1, num_timesteps, dtype=int)
    else:
        grid = make_grid(N)
        scenario = sample_flow_scenario(rng, num_sources_range=num_sources_range, time_window=time_window)

    if sensor_indices is not None:
        obs_idx = np.asarray(sensor_indices, dtype=np.int64)
        obs_idx_flat = np.ravel_multi_index(obs_idx.T, (N, N))
        effective_M = obs_idx.shape[0]
    else:
        obs_idx_flat = rng.choice(N * N, size=M, replace=False)
        obs_idx = np.array(np.unravel_index(obs_idx_flat, (N, N))).T
        effective_M = obs_idx.shape[0]

    for t_idx in range(num_timesteps):
        if precomputed is not None:
            sample_idx = int(sample_indices[t_idx])
            fields = fields_array[sample_idx]
            t = float(times[sample_idx]) if precomputed.times is not None else float(t_idx)
        else:
            t = np.linspace(time_window[0], time_window[1], num_timesteps)[t_idx]
            fields = evaluate_flow_field(grid, scenario, t)

        y_obs, mask, _ = gappy_sample_flow(fields, effective_M, sigma, rng, obs_idx=obs_idx)

        mask_channels = np.broadcast_to(mask, (fields.shape[0], N, N)).astype(np.float32)
        z = np.concatenate([y_obs, mask_channels], axis=0).astype(np.float32)
        z_t = torch.from_numpy(z).unsqueeze(0).to(DEVICE)

        y_hat, _, _ = model(z_t)
        y_hat = y_hat.squeeze(0).detach().cpu().numpy()

        metrics = compute_flow_metrics(y_hat, fields)
        metrics_str = ", ".join(
            f"{name}: {metrics['psnr_channels'][idx]:.2f} dB"
            for idx, name in enumerate(flow_components)
        )
        logger.info(
            f"Demo @ t={t:.2f} — MSE {metrics['mse']:.6e}, RMSE {metrics['rmse']:.6e}, "
            f"PSNR_global {metrics['psnr_global']:.2f} dB | {metrics_str}"
        )

        path = results.plot_flow_reconstruction(
            grid,
            fields,
            y_obs,
            y_hat,
            mask,
            obs_idx,
            flow_components,
            metrics,
            outdir=outdir,
            filename_suffix=f"_t_{t_idx:03d}",
        )
        frames.append(imageio.imread(path))

    gif_path = os.path.join(outdir, "gappy_flow_recon_temporal.gif")
    imageio.mimsave(gif_path, frames, fps=5)
    logger.info(f"Saved animation to {gif_path}")


def main():
    os.makedirs("log", exist_ok=True)
    logger.add("log/2d_temporal.log")

    dataset_path = os.environ.get("GAPPY_TEMPORAL_DATASET")
    precomputed_dataset: PrecomputedDataset | None = None
    flow_components = FLOW_COMPONENTS
    default_sigma = DEFAULT_SIGMA
    field_channels = len(flow_components)

    if dataset_path:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")
        precomputed_dataset = load_precomputed_dataset(dataset_path)
        field_channels = precomputed_dataset.fields.shape[1]
        default_sigma = tuple(0.01 for _ in range(field_channels))
        flow_components = precomputed_dataset.channel_names
        logger.info(
            f"Loaded MFEM dataset from {dataset_path} "
            f"with shape {precomputed_dataset.fields.shape} and channels {flow_components}"
        )

    if precomputed_dataset is not None:
        dataset_N = precomputed_dataset.fields.shape[-1]
        env_N = os.environ.get("GAPPY_TEMPORAL_N")
        if env_N is not None and int(env_N) != dataset_N:
            logger.warning(
                f"GAPPY_TEMPORAL_N={env_N} ignored; using dataset resolution N={dataset_N}."
            )
        N = dataset_N
    else:
        N = int(os.environ.get("GAPPY_TEMPORAL_N", 64))

    M = int(os.environ.get("GAPPY_TEMPORAL_M", 256))
    SIGMA = parse_sigma(os.environ.get("GAPPY_TEMPORAL_SIGMA"), default_sigma, field_channels)
    EPOCHS = int(os.environ.get("GAPPY_TEMPORAL_EPOCHS", 50))
    BS = int(os.environ.get("GAPPY_TEMPORAL_BATCH", 64))
    LR = float(os.environ.get("GAPPY_TEMPORAL_LR", 1e-3))
    OBS_LMB = float(os.environ.get("GAPPY_TEMPORAL_OBS_LMB", 1.0))
    KL_LMB = float(os.environ.get("GAPPY_TEMPORAL_KL_LMB", 1e-6))
    NUM_TIMESTEPS = int(os.environ.get("GAPPY_TEMPORAL_STEPS", 20))
    NUM_DEMOS = int(os.environ.get("GAPPY_TEMPORAL_DEMOS", 5))

    default_train = 10000 if precomputed_dataset is None else precomputed_dataset.fields.shape[0]
    default_val = 1000 if precomputed_dataset is None else max(1, precomputed_dataset.fields.shape[0] // 5)
    TRAIN_SIZE = int(os.environ.get("GAPPY_TEMPORAL_TRAIN", default_train))
    VAL_SIZE = int(os.environ.get("GAPPY_TEMPORAL_VAL", default_val))

    NUM_SOURCES_RANGE = (
        int(os.environ.get("GAPPY_TEMPORAL_MIN_SOURCES", 2)),
        int(os.environ.get("GAPPY_TEMPORAL_MAX_SOURCES", 6)),
    )
    TIME_WINDOW = (
        float(os.environ.get("GAPPY_TEMPORAL_TMIN", 0.0)),
        float(os.environ.get("GAPPY_TEMPORAL_TMAX", 5.0)),
    )

    logger.info(f"Device: {DEVICE}")

    use_pysensors_env = os.environ.get("GAPPY_TEMPORAL_USE_PYSENSORS")
    if use_pysensors_env is None:
        use_pysensors = precomputed_dataset is None
    else:
        use_pysensors = int(use_pysensors_env) != 0
    candidate_counts_env = os.environ.get("GAPPY_TEMPORAL_SENSOR_COUNTS", "32,64,96,128,192,256")
    sensor_counts = [int(x.strip()) for x in candidate_counts_env.split(",") if x.strip()]
    sensor_basis_modes = int(os.environ.get("GAPPY_TEMPORAL_SENSOR_BASIS", 48))
    sensor_train_snapshots = int(os.environ.get("GAPPY_TEMPORAL_SENSOR_TRAIN_SNAPSHOTS", 2000))
    sensor_val_snapshots = int(os.environ.get("GAPPY_TEMPORAL_SENSOR_VAL_SNAPSHOTS", 500))
    sensor_aggregation_mode = os.environ.get("GAPPY_TEMPORAL_SENSOR_MODE", "rss")
    sensor_seed = int(os.environ.get("GAPPY_TEMPORAL_SENSOR_SEED", SEED + 2048))

    sensor_indices = None
    sensor_error_curve = None
    if use_pysensors:
        if not HAVE_PYSENSORS:
            raise RuntimeError("pysensors not available; install it or disable GAPPY_TEMPORAL_USE_PYSENSORS.")
        logger.info(
            "Optimizing sensor layout with pysensors "
            f"(candidates={sensor_counts}, basis_modes={sensor_basis_modes}, aggregation={sensor_aggregation_mode})"
        )
        sensors, ranking, counts_arr, errors, best_count = optimize_sensor_layout(
            N=N,
            candidate_sensor_counts=sensor_counts,
            num_train_snapshots=sensor_train_snapshots,
            num_val_snapshots=sensor_val_snapshots,
            aggregation_mode=sensor_aggregation_mode,
            n_basis_modes=sensor_basis_modes,
            seed=sensor_seed,
            num_sources_range=NUM_SOURCES_RANGE,
            time_window=TIME_WINDOW,
            precomputed_fields=precomputed_dataset.fields if precomputed_dataset is not None else None,
        )
        sensor_indices = sensors
        sensor_error_curve = list(zip(counts_arr.tolist(), errors.tolist()))
        logger.info(
            "pysensors reconstruction error curve: "
            + ", ".join(f"M={int(c)} -> {err:.4e}" for c, err in sensor_error_curve)
        )
        logger.info(
            f"Selected {best_count} sensors via pysensors. Top 5 sensor coordinates: "
            f"{[tuple(map(int, sensor_indices[i])) for i in range(min(5, sensor_indices.shape[0]))]}"
        )
        M = best_count

    model = train_gappy_conv_vae_2d(
        N=N,
        M=M,
        sigma=SIGMA,
        epochs=EPOCHS,
        bs=BS,
        lr=LR,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        latent_dim=32,
        width=[32, 64, 128],
        depth=3,
        obs_lambda=OBS_LMB,
            kl_lambda=KL_LMB,
            seed=SEED,
            num_sources_range=NUM_SOURCES_RANGE,
            time_window=TIME_WINDOW,
            sensor_indices=sensor_indices,
            field_channels=field_channels,
            precomputed_fields=precomputed_dataset.fields if precomputed_dataset is not None else None,
        )

    base_outdir = "temporal_results"
    os.makedirs(base_outdir, exist_ok=True)
    if sensor_indices is not None:
        sensor_path = os.path.join(base_outdir, "optimized_sensors.npy")
        np.save(sensor_path, sensor_indices)
        logger.info(f"Saved optimized sensor locations to {sensor_path}")
        if sensor_error_curve is not None:
            curve_path = os.path.join(base_outdir, "sensor_error_curve.csv")
            with open(curve_path, "w", encoding="utf-8") as fh:
                fh.write("sensors,reconstruction_error\n")
                for count, err in sensor_error_curve:
                    fh.write(f"{int(count)},{err:.8e}\n")
            logger.info(f"Saved sensor error curve to {curve_path}")

    for i in range(NUM_DEMOS):
        demo_seed = SEED + 123 + i
        outdir = os.path.join(base_outdir, f"flow_run_{i:02d}_seed_{demo_seed}")
        logger.info(
            f"Running demo {i + 1}/{NUM_DEMOS} with seed {demo_seed} and output dir {outdir}"
        )
        demo_temporal(
            model,
            N=N,
            M=M,
            sigma=SIGMA,
            seed=demo_seed,
            outdir=outdir,
            num_timesteps=NUM_TIMESTEPS,
            time_window=TIME_WINDOW,
            num_sources_range=NUM_SOURCES_RANGE,
            sensor_indices=sensor_indices,
            flow_components=flow_components,
            precomputed=precomputed_dataset,
        )


if __name__ == "__main__":
    main()
