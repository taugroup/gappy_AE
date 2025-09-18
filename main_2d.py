
#!/usr/bin/env python
# coding: utf-8

import os, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger

import results

# Optional: scikit-fem for H1-smoothness penalty
try:
    import skfem as fem
    from skfem.helpers import dot, grad
    HAVE_SKFEM = True
except Exception:
    HAVE_SKFEM = False

torch.set_float32_matmul_precision('high') if hasattr(torch, "set_float32_matmul_precision") else None

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


# x-grid shape: (N,N)
# y-true shape: (N,N)

def sample_surface(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    y(x1, x2) = A * s(2π(f1*x1 + f2*x2) + φ); s ∈ {sin, cos}, A ∈ [0.8,1.2], f1,f2 ∈ {1,2,3}, φ ∼ U[0,2π)
    """
    x = np.linspace(0.0, 1.0, N, dtype=np.float32)
    x1, x2 = np.meshgrid(x, x)
    A   = rng.uniform(0.8, 1.2)
    f1  = rng.choice([1, 2, 3])
    f2  = rng.choice([1, 2, 3])
    phi = rng.uniform(0.0, 2*np.pi)
    if rng.random() < 0.5:
        y = A * np.sin(2*np.pi*(f1*x1 + f2*x2) + phi)
    else:
        y = A * np.cos(2*np.pi*(f1*x1 + f2*x2) + phi)
    return (x1, x2), y.astype(np.float32)

def gappy_sample(y: np.ndarray, M: int, sigma: float, rng: np.random.Generator):
    """
    Choose M observed indices; add N(0, σ^2) to those entries.
    Returns:
      y_obs_masked: (N,N)   (unobserved entries zero-filled)
      mask:         (N,N)   {0,1}
      obs_idx:      (M,2)
    """
    N = y.shape[0]
    obs_idx_flat = rng.choice(N*N, size=M, replace=False)
    obs_idx = np.array(np.unravel_index(obs_idx_flat, (N,N))).T
    mask = np.zeros((N,N), dtype=np.float32)
    mask[obs_idx[:,0], obs_idx[:,1]] = 1.0
    y_obs = y.copy()
    noise = rng.normal(0.0, sigma, size=M).astype(np.float32)
    y_obs[obs_idx[:,0], obs_idx[:,1]] += noise
    y_obs[mask == 0.0] = 0.0
    return y_obs.astype(np.float32), mask, obs_idx


class SurfaceGappyDataset(Dataset):
    """
    __getitem__ returns:
      inp:   concat([y_obs_masked, mask]) ∈ R^{2xNxN}   # shape (2,N,N)
      y:     full ground truth ∈ R^{NxN}
      mask:  {0,1}^{NxN} for observed positions ∈ R^{NxN}
    """
    def __init__(self, N=64, M=256, sigma=0.05, size=10000, seed=0):
        self.N, self.M, self.sigma, self.size = N, M, sigma, size
        self.rng = np.random.default_rng(seed)

    def __len__(self): return self.size

    def __getitem__(self, idx):
        _, y = sample_surface(self.N, self.rng)
        y_obs, mask, _ = gappy_sample(y, self.M, self.sigma, self.rng)
        inp = np.stack([y_obs, mask], axis=0).astype(np.float32)  # (2,N,N)
        return (
            torch.from_numpy(inp),   # (2,N,N)
            torch.from_numpy(y),     # (N,N)
            torch.from_numpy(mask)   # (N,N)
        )

# quick sanity
tmp = SurfaceGappyDataset(N=64, M=256, sigma=0.05, size=8, seed=SEED)
inp, y, mask = tmp[0]
logger.info(f"Shapes — inp: {tuple(inp.shape)}, y: {tuple(y.shape)}, mask: {tuple(mask.shape)}")


class GappyConvVAE2D(nn.Module):
    def __init__(self, N: int, latent_dim: int = 32, width: list[int] = [32, 64, 128], depth: int = 3):
        super().__init__()
        self.N = N
        self.latent_dim = latent_dim

        encoder_layers = []
        in_channels = 2
        for i in range(depth):
            encoder_layers.append(nn.Conv2d(in_channels, width[i], kernel_size=5, stride=2, padding=2))
            encoder_layers.append(nn.ReLU())
            in_channels = width[i]
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, N, N)
            dummy_output = self.encoder(dummy_input)
            flattened_size = dummy_output.shape[1]

        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, flattened_size)
        
        decoder_layers = []
        decoder_layers.append(nn.Unflatten(1, (width[-1], N // (2**depth), N // (2**depth))))
        in_channels = width[-1]
        for i in range(depth - 1, 0, -1):
            decoder_layers.append(nn.ConvTranspose2d(in_channels, width[i-1], kernel_size=5, stride=2, padding=2, output_padding=1))
            decoder_layers.append(nn.ReLU())
            in_channels = width[i-1]
        decoder_layers.append(nn.ConvTranspose2d(in_channels, 1, kernel_size=5, stride=2, padding=2, output_padding=1))
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

N = 64
model = GappyConvVAE2D(N=N, latent_dim=32, width=[32, 64, 128], depth=3).to(DEVICE)
logger.info(f"Trainable params: {count_params(model)}")


def make_loaders(N=64, M=256, sigma=0.05, train_size=10000, val_size=1000, bs=128, seed=SEED):
    tr = SurfaceGappyDataset(N=N, M=M, sigma=sigma, size=train_size, seed=seed)
    va = SurfaceGappyDataset(N=N, M=M, sigma=sigma, size=val_size,   seed=seed+1)
    return (
        DataLoader(tr, batch_size=bs, shuffle=True,  drop_last=True, num_workers=0),
        DataLoader(va, batch_size=bs, shuffle=False, drop_last=False, num_workers=0),
    )

def train_gappy_conv_vae_2d(
    N=64, M=256, sigma=0.05, epochs=50, bs=128, lr=1e-3,
    latent_dim=32, width=[32, 64, 128], depth=3, obs_lambda=1.0, kl_lambda=1e-3, seed=SEED,
):
    torch.manual_seed(seed)
    model = GappyConvVAE2D(N=N, latent_dim=latent_dim, width=width, depth=depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader, va_loader = make_loaders(N, M, sigma, 10000, 1000, bs, seed)

    def run_epoch(loader, train_mode: bool):
        if train_mode: model.train()
        else: model.eval()
        tot, count = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for inp, y, mask in loader:
                inp  = inp.to(DEVICE)     # (B,2,N,N)
                y    = y.to(DEVICE)       # (B,N,N)
                mask = mask.to(DEVICE)    # (B,N,N)

                y_hat, mu, logvar = model(inp)        # (B,1,N,N)
                y_hat = y_hat.squeeze(1)              # (B,N,N)

                # loss terms
                loss_full = nn.functional.mse_loss(y_hat, y)
                loss_obs  = nn.functional.mse_loss(y_hat*mask, y*mask)
                loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = loss_full + obs_lambda*loss_obs + kl_lambda*loss_kl

                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                tot += float(loss.detach().cpu()) * inp.size(0)
                count += inp.size(0)
        return tot / max(count, 1)

    best_val = math.inf
    t0 = time.time()
    for ep in range(1, epochs+1):
        tr = run_epoch(tr_loader, True)
        va = run_epoch(va_loader, False)
        best_val = min(best_val, va)
        if ep == 1 or ep % max(1, epochs//10) == 0:
            logger.info(f"[ep {ep:03d}] train {tr:.6f}  val {va:.6f}  best {best_val:.6f}")
    logger.info(f"Wall-clock: {time.time()-t0:.2f}s")

    return model

@torch.no_grad()
def demo_once(model: nn.Module, N=64, M=256, sigma=0.05, seed=SEED+999, outdir="outputs"):
    rng = np.random.default_rng(seed)
    (x1, x2), y = sample_surface(N, rng)
    y_obs, mask, obs_idx = gappy_sample(y, M, sigma, rng)

    z = np.stack([y_obs, mask], axis=0).astype(np.float32)
    z_t = torch.from_numpy(z).unsqueeze(0).to(DEVICE)
    
    y_hat, _, _ = model(z_t)
    y_hat = y_hat.squeeze(0).squeeze(0).detach().cpu().numpy()

    mse = float(np.mean((y_hat - y)**2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    logger.info(f"Demo — MSE: {mse:.6e} ; PSNR: {psnr:.2f} dB ; N={N}, M={M}, σ={sigma}")

    path = results.plot_reconstruction_2d(x1, x2, y, obs_idx, y_obs, y_hat, mse, psnr, N, M, sigma, outdir)
    logger.info(f"Saved reconstruction plot to {path}")
    return psnr


def main():
    logger.add("file_2d_{time}.log")

    # Config (edit here)
    N = 64
    M = 256
    SIGMA = 0.05
    EPOCHS = 50
    BS = 64
    LR = 1e-3
    OBS_LMB = 1.0
    KL_LMB = 1e-6

    logger.info(f"Device: {DEVICE}")
    
    observation_sweep_analysis()

def observation_sweep_analysis():
    """
    Train a model for different numbers of observations M and plot the results.
    """
    N = 64
    SIGMA = 0.05
    EPOCHS = 50
    BS = 64
    LR = 1e-3
    OBS_LMB = 1.0
    KL_LMB = 1e-6

    M_values = [16, 32, 64, 128, 256, 512]
    psnrs = []

    for M in M_values:
        logger.info(f"Running 2D VAE model with M={M}")
        model = train_gappy_conv_vae_2d(
            N=N, M=M, sigma=SIGMA, epochs=EPOCHS, bs=BS, lr=LR,
            latent_dim=32, width=[32, 64, 128], depth=3,
            obs_lambda=OBS_LMB, kl_lambda=KL_LMB, seed=SEED
        )
        psnr = demo_once(model, N=N, M=M, sigma=SIGMA, seed=SEED+123, outdir="outputs/2d_model")
        psnrs.append(psnr)

    results.plot_observation_sweep(M_values, psnrs, "outputs/observation_sweep_2d.png")



if __name__ == "__main__":
    main()
