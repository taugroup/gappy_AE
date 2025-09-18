
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


# x-grid shape: (N,)
# y-true shape: (N,)

def sample_curve(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    y(x) = A * s(2π f x + φ); s ∈ {sin, cos}, A ∈ [0.8,1.2], f ∈ {1,2,3}, φ ∼ U[0,2π)
    """
    x = np.linspace(0.0, 1.0, N, dtype=np.float32)
    A   = rng.uniform(0.8, 1.2)
    f   = rng.choice([1, 2, 3])
    phi = rng.uniform(0.0, 2*np.pi)
    if rng.random() < 0.5:
        y = A * np.sin(2*np.pi*f*x + phi)
    else:
        y = A * np.cos(2*np.pi*f*x + phi)
    return x, y.astype(np.float32)

def gappy_sample(y: np.ndarray, M: int, sigma: float, rng: np.random.Generator):
    """
    Choose M observed indices; add N(0, σ^2) to those entries.
    Returns:
      y_obs_masked: (N,)   (unobserved entries zero-filled)
      mask:         (N,)   {0,1}
      obs_idx:      (M,)
    """
    N = y.shape[0]
    obs_idx = rng.choice(N, size=M, replace=False)
    mask = np.zeros(N, dtype=np.float32)
    mask[obs_idx] = 1.0
    y_obs = y.copy()
    y_obs[obs_idx] = y_obs[obs_idx] + rng.normal(0.0, sigma, size=M).astype(np.float32)
    y_obs[mask == 0.0] = 0.0
    return y_obs.astype(np.float32), mask, obs_idx


class SineGappyDataset(Dataset):
    """
    __getitem__ returns:
      inp:   concat([y_obs_masked, mask]) ∈ R^{2N}   # shape (2N,)
      y:     full ground truth ∈ R^{N}
      mask:  {0,1}^N for observed positions ∈ R^{N}
    """
    def __init__(self, N=256, M=32, sigma=0.05, size=10000, seed=0):
        self.N, self.M, self.sigma, self.size = N, M, sigma, size
        self.rng = np.random.default_rng(seed)

    def __len__(self): return self.size

    def __getitem__(self, idx):
        x, y = sample_curve(self.N, self.rng)
        y_obs, mask, _ = gappy_sample(y, self.M, self.sigma, self.rng)
        inp = np.concatenate([y_obs, mask], axis=0).astype(np.float32)  # (2N,)
        return (
            torch.from_numpy(inp),   # (2N,)
            torch.from_numpy(y),     # (N,)
            torch.from_numpy(mask)   # (N,)
        )

# quick sanity
tmp = SineGappyDataset(N=256, M=32, sigma=0.05, size=8, seed=SEED)
inp, y, mask = tmp[0]
logger.info(f"Shapes — inp: {tuple(inp.shape)}, y: {tuple(y.shape)}, mask: {tuple(mask.shape)}")


class GappyVAE(nn.Module):
    def __init__(self, N: int, width: int = 256, depth: int = 3, latent_dim: int = 32):
        super().__init__()
        self.N = N
        self.latent_dim = latent_dim
        D_in = 2 * N

        # Encoder
        encoder_layers = []
        last = D_in
        for _ in range(depth):
            encoder_layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(width, latent_dim)
        self.fc_logvar = nn.Linear(width, latent_dim)

        # Decoder
        decoder_layers = []
        last = latent_dim
        for _ in range(depth):
            decoder_layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        decoder_layers += [nn.Linear(last, N)]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decode(z)
        return y_hat, mu, logvar

class GappyConvVAE(nn.Module):
    def __init__(self, N: int, latent_dim: int = 32, width: list[int] = [32, 64, 128], depth: int = 3):
        super().__init__()
        self.N = N
        self.latent_dim = latent_dim

        encoder_layers = []
        in_channels = 2
        for i in range(depth):
            encoder_layers.append(nn.Conv1d(in_channels, width[i], kernel_size=5, stride=2, padding=2))
            encoder_layers.append(nn.ReLU())
            in_channels = width[i]
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, N)
            dummy_output = self.encoder(dummy_input)
            flattened_size = dummy_output.shape[1]

        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, flattened_size)
        
        decoder_layers = []
        decoder_layers.append(nn.Unflatten(1, (width[-1], N // (2**depth))))
        in_channels = width[-1]
        for i in range(depth - 1, 0, -1):
            decoder_layers.append(nn.ConvTranspose1d(in_channels, width[i-1], kernel_size=5, stride=2, padding=2, output_padding=1))
            decoder_layers.append(nn.ReLU())
            in_channels = width[i-1]
        decoder_layers.append(nn.ConvTranspose1d(in_channels, 1, kernel_size=5, stride=2, padding=2, output_padding=1))
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

class GappyAE(nn.Module):
    def __init__(self, N: int, width: int = 256, depth: int = 3):
        super().__init__()
        D_in = 2 * N
        layers = []
        last = D_in
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        layers += [nn.Linear(last, N)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):  # z: (B, 2N)
        return self.net(z) # -> (B, N)

# quick param count
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

N = 256
model = GappyVAE(N=N, width=256, depth=3).to(DEVICE)
logger.info(f"Trainable params: {count_params(model)}")


def fem_stiffness_1d(N: int):
    """
    Assemble 1D stiffness matrix K (∫ grad u · grad v) on [0,1] with P1 elements.
    Returns dense np.ndarray (N,N) or None if scikit-fem is unavailable.
    """
    if not HAVE_SKFEM:
        return None
    mesh = fem.MeshLine(np.linspace(0.0, 1.0, N))
    Vh = fem.Basis(mesh, fem.ElementLineP1())

    @fem.BilinearForm
    def a(u, v, _):
        return dot(grad(u), grad(v))

    K = a.assemble(Vh).toarray().astype(np.float32)  # (N,N)
    return K

def smoothness_loss(y_hat: torch.Tensor, K_np: np.ndarray):
    """
    y_hat: (B,N), K_np: (N,N)
    penalty = mean_b  y_hat[b]^T K y_hat[b]
    """
    if K_np is None:
        return torch.tensor(0.0, device=y_hat.device)
    K = torch.from_numpy(K_np).to(y_hat.device)
    quad = (y_hat @ K) * y_hat
    return quad.sum(dim=1).mean()

K_np = fem_stiffness_1d(N)
logger.info(f"K assembled? {K_np is not None} ; shape: {None if K_np is None else K_np.shape}")


def make_loaders(N=256, M=32, sigma=0.05, train_size=10000, val_size=1000, bs=128, seed=SEED):
    tr = SineGappyDataset(N=N, M=M, sigma=sigma, size=train_size, seed=seed)
    va = SineGappyDataset(N=N, M=M, sigma=sigma, size=val_size,   seed=seed+1)
    return (
        DataLoader(tr, batch_size=bs, shuffle=True,  drop_last=True, num_workers=0),
        DataLoader(va, batch_size=bs, shuffle=False, drop_last=False, num_workers=0),
    )

def train_gappy_ae(
    N=256, M=32, sigma=0.05, epochs=50, bs=128, lr=1e-3,
    width=256, depth=3, obs_lambda=1.0, fem_lambda=1e-4, seed=SEED,
):
    torch.manual_seed(seed)
    model = GappyAE(N=N, width=width, depth=depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader, va_loader = make_loaders(N, M, sigma, 10000, 1000, bs, seed)
    K_local = fem_stiffness_1d(N) if fem_lambda > 0 else None

    def run_epoch(loader, train_mode: bool):
        if train_mode: model.train()
        else: model.eval()
        tot, count = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for inp, y, mask in loader:
                inp  = inp.to(DEVICE)     # (B,2N)
                y    = y.to(DEVICE)       # (B,N)
                mask = mask.to(DEVICE)    # (B,N)

                y_hat = model(inp)        # (B,N)

                # loss terms
                loss_full = nn.functional.mse_loss(y_hat, y)
                loss_obs  = nn.functional.mse_loss(y_hat*mask, y*mask)
                loss_smooth = smoothness_loss(y_hat, K_local) if K_local is not None else torch.tensor(0.0, device=DEVICE)

                loss = loss_full + obs_lambda*loss_obs + fem_lambda*loss_smooth

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

def train_gappy_vae(
    N=256, M=32, sigma=0.05, epochs=50, bs=128, lr=1e-3,
    width=256, depth=3, latent_dim=32, obs_lambda=1.0, fem_lambda=1e-4, kl_lambda=1e-3, seed=SEED,
):
    torch.manual_seed(seed)
    model = GappyVAE(N=N, width=width, depth=depth, latent_dim=latent_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader, va_loader = make_loaders(N, M, sigma, 10000, 1000, bs, seed)
    K_local = fem_stiffness_1d(N) if fem_lambda > 0 else None

    def run_epoch(loader, train_mode: bool):
        if train_mode: model.train()
        else: model.eval()
        tot, count = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for inp, y, mask in loader:
                inp  = inp.to(DEVICE)     # (B,2N)
                y    = y.to(DEVICE)       # (B,N)
                mask = mask.to(DEVICE)    # (B,N)

                y_hat, mu, logvar = model(inp)        # (B,N)

                # loss terms
                loss_full = nn.functional.mse_loss(y_hat, y)
                loss_obs  = nn.functional.mse_loss(y_hat*mask, y*mask)
                loss_smooth = smoothness_loss(y_hat, K_local) if K_local is not None else torch.tensor(0.0, device=DEVICE)
                loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = loss_full + obs_lambda*loss_obs + fem_lambda*loss_smooth + kl_lambda*loss_kl

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

def train_gappy_conv_vae(
    N=256, M=32, sigma=0.05, epochs=50, bs=128, lr=1e-3,
    latent_dim=32, width=[32, 64, 128], depth=3, obs_lambda=1.0, fem_lambda=1e-4, kl_lambda=1e-3, seed=SEED,
):
    torch.manual_seed(seed)
    model = GappyConvVAE(N=N, latent_dim=latent_dim, width=width, depth=depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader, va_loader = make_loaders(N, M, sigma, 10000, 1000, bs, seed)
    K_local = fem_stiffness_1d(N) if fem_lambda > 0 else None

    def run_epoch(loader, train_mode: bool):
        if train_mode: model.train()
        else: model.eval()
        tot, count = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for inp, y, mask in loader:
                inp  = inp.to(DEVICE)     # (B,2N)
                y    = y.to(DEVICE)       # (B,N)
                mask = mask.to(DEVICE)    # (B,N)

                # Reshape for Conv1d
                inp = inp.view(inp.size(0), 2, N)

                y_hat, mu, logvar = model(inp)        # (B,1,N)
                y_hat = y_hat.squeeze(1)              # (B,N)

                # loss terms
                loss_full = nn.functional.mse_loss(y_hat, y)
                loss_obs  = nn.functional.mse_loss(y_hat*mask, y*mask)
                loss_smooth = smoothness_loss(y_hat, K_local) if K_local is not None else torch.tensor(0.0, device=DEVICE)
                loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = loss_full + obs_lambda*loss_obs + fem_lambda*loss_smooth + kl_lambda*loss_kl

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
def demo_once(model: nn.Module, N=256, M=32, sigma=0.05, seed=SEED+999, outdir="outputs"):
    rng = np.random.default_rng(seed)
    x, y = sample_curve(N, rng)                           # (N,)
    y_obs, mask, obs_idx = gappy_sample(y, M, sigma, rng) # (N,), (N,), (M,)

    z = np.concatenate([y_obs, mask], axis=0).astype(np.float32)  # (2N,)
    z_t = torch.from_numpy(z).unsqueeze(0).to(DEVICE)             # (1,2N) 
    
    if isinstance(model, GappyVAE):
        y_hat, _, _ = model(z_t)
    elif isinstance(model, GappyConvVAE):
        z_t = z_t.view(z_t.size(0), 2, N)
        y_hat, _, _ = model(z_t)
        y_hat = y_hat.squeeze(1)
    else:
        y_hat = model(z_t)
    y_hat = y_hat.squeeze(0).detach().cpu().numpy()          # (N,)

    mse = float(np.mean((y_hat - y)**2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))  # signals ~ O(1)
    logger.info(f"Demo — MSE: {mse:.6e} ; PSNR: {psnr:.2f} dB ; N={N}, M={M}, σ={sigma}")

    path = results.plot_reconstruction(x, y, obs_idx, y_obs, y_hat, mse, psnr, N, M, sigma, outdir)
    logger.info(f"Saved reconstruction plot to {path}")
    return psnr

def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))

def make_sampler(mask_mode="random", block_frac=0.15, cluster_k=4):
    """
    Returns a function f(y, M, sigma, rng) -> (y_obs, mask, obs_idx)
    mask_mode:
      - "random": uniform indices
      - "block": one contiguous block observed; rest zero (M ignored, use block_frac)
      - "clusters": k clusters of equal size summing to M
    """
    def sampler(y, M, sigma, rng):
        N = y.shape[0]
        if mask_mode == "random":
            obs_idx = rng.choice(N, size=M, replace=False)
        elif mask_mode == "block":
            L = max(1, int(block_frac * N))
            start = int(rng.integers(0, max(1, N - L)))
            obs_idx = np.arange(start, start + L)
            obs_idx = np.clip(obs_idx, 0, N-1)
        elif mask_mode == "clusters":
            ks = max(1, int(cluster_k))
            base = M // ks
            rem = M - ks * base
            obs_idx = []
            for i in range(ks):
                sz = base + (1 if i < rem else 0)
                if sz == 0: 
                    continue
                center = int(rng.integers(0, N))
                left = max(0, center - sz//2)
                right = min(N, left + sz)
                obs_idx.extend(range(left, right))
            obs_idx = np.array(sorted(set(np.clip(obs_idx, 0, N-1))), dtype=int)
            # adjust if we overshot
            if obs_idx.size > M:
                obs_idx = rng.choice(obs_idx, size=M, replace=False)
        else:
            raise ValueError("unknown mask_mode")
        obs_idx = np.unique(obs_idx)
        mask = np.zeros(N, dtype=np.float32)
        mask[obs_idx] = 1.0
        y_obs = y.copy()
        y_obs[obs_idx] = y_obs[obs_idx] + rng.normal(0.0, sigma, size=obs_idx.size).astype(np.float32)
        y_obs[mask == 0.0] = 0.0
        return y_obs.astype(np.float32), mask, obs_idx
    return sampler

@torch.no_grad()
def eval_batch(model, N=256, M=32, sigma=0.05, n_trials=200, mask_mode="random", seed=12345, freq_choices=(1,2,3), only_sin=False, only_cos=False):
    rng = np.random.default_rng(seed)
    sampler = make_sampler(mask_mode)
    mses = []
    for _ in range(n_trials):
        # custom curve with optional distribution shift
        x = np.linspace(0.0, 1.0, N, dtype=np.float32)
        A   = rng.uniform(0.8, 1.2)
        f   = rng.choice(freq_choices)
        phi = rng.uniform(0.0, 2*np.pi)
        if only_sin:
            y = A * np.sin(2*np.pi*f*x + phi)
        elif only_cos:
            y = A * np.cos(2*np.pi*f*x + phi)
        else:
            y = A * (np.sin(2*np.pi*f*x + phi) if rng.random() < 0.5 else np.cos(2*np.pi*f*x + phi))
        y = y.astype(np.float32)

        y_obs, mask, _ = sampler(y, M, sigma, rng)
        z = np.concatenate([y_obs, mask], axis=0).astype(np.float32)  # (2N,)
        z_t = torch.from_numpy(z).unsqueeze(0).to(DEVICE)
        if isinstance(model, GappyVAE):
            y_hat, _, _ = model(z_t)
        elif isinstance(model, GappyConvVAE):
            z_t = z_t.view(z_t.size(0), 2, N)
            y_hat, _, _ = model(z_t)
            y_hat = y_hat.squeeze(1)
        else:
            y_hat = model(z_t)
        y_hat = y_hat.squeeze(0).cpu().numpy()
        mses.append(np.mean((y_hat - y)**2))
    mses = np.array(mses)
    return float(mses.mean()), float(mses.std()), float(psnr_from_mse(mses.mean()))

def main():
    logger.add("file_{time}.log")

    # Config (edit here)
    N = 256          # signal length
    M = 32           # observed samples
    SIGMA = 0.05     # noise std on observed entries
    EPOCHS = 50
    BS = 128
    LR = 1e-3
    OBS_LMB = 1.0
    FEM_LMB = 1e-3   # set 0.0 to disable smoothness penalty
    KL_LMB = 1e-6    # KL divergence weight

    logger.info(f"Device: {DEVICE}")
    
    logger.info("Running final model with best hyperparameters")
    model = train_gappy_conv_vae(
        N=N, M=M, sigma=SIGMA, epochs=EPOCHS, bs=BS, lr=LR,
        latent_dim=32, width=[64, 128, 256], depth=3,
        obs_lambda=OBS_LMB, fem_lambda=FEM_LMB, kl_lambda=KL_LMB, seed=SEED
    )
    demo_once(model, N=N, M=M, sigma=SIGMA, seed=SEED+123, outdir="outputs/final_model")


if __name__ == "__main__":
    main()
