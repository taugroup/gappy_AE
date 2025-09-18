
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_reconstruction(x, y, obs_idx, y_obs, y_hat, mse, psnr, N, M, sigma, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(x, y, lw=2, label="ground truth")
    plt.scatter(x[obs_idx], y_obs[obs_idx], s=18, label="noisy obs")
    plt.plot(x, y_hat, lw=2, ls='--', label="VAE recon")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.title("Gappy-VAE reconstruction")
    path = os.path.join(outdir, f"gappy_recon_N{N}_M{M}_sigma{sigma}.png")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_surface(ax, x1, x2, y, title, cmap='viridis'):
    ax.plot_surface(x1, x2, y, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")

def plot_reconstruction_2d(x1, x2, y, obs_idx, y_obs, y_hat, mse, psnr, N, M, sigma, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(18, 6))
    
    # Ground truth
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    plot_surface(ax1, x1, x2, y, "Ground Truth")

    # Observed
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    y_obs_plot = y_obs.copy()
    y_obs_plot[y_obs_plot == 0] = np.nan
    plot_surface(ax2, x1, x2, y_obs_plot, "Noisy Observations")
    ax2.scatter(x1[obs_idx[:,0], obs_idx[:,1]], x2[obs_idx[:,0], obs_idx[:,1]], y_obs[obs_idx[:,0], obs_idx[:,1]], color='r', s=5, label="Observed Points")

    # Reconstruction
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plot_surface(ax3, x1, x2, y_hat, f"VAE Reconstruction (PSNR: {psnr:.2f} dB)")

    plt.suptitle(f"Gappy-VAE 2D Reconstruction (N={N}x{N}, M={M}, σ={sigma})")
    path = os.path.join(outdir, f"gappy_recon_2d_N{N}_M{M}_sigma{sigma}.png")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_noise_sweep(sigmas, psnrs, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(sigmas, psnrs, marker='o')
    plt.xlabel("σ (noise std)"); plt.ylabel("PSNR (dB)")
    plt.title("Noise sweep at fixed M")
    plt.grid(True)
    path = os.path.join(outdir, "noise_sweep.png")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_observation_sweep(Ms, psnrs, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(Ms, psnrs, marker='o')
    plt.xlabel("M (observed points)"); plt.ylabel("PSNR (dB)")
    plt.title("Observation count sweep")
    plt.grid(True)
    path = os.path.join(outdir, "observation_sweep.png")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_mask_pattern_stress(scores, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.bar([s[0] for s in scores], [s[2] for s in scores])
    plt.ylabel("PSNR (dB)"); plt.title("Mask pattern stress @ fixed M, σ")
    path = os.path.join(outdir, "mask_pattern_stress.png")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_frequency_generalization(freq_grid, psnrs_f, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(freq_grid, psnrs_f, marker='o')
    plt.xlabel("frequency f"); plt.ylabel("PSNR (dB)")
    plt.title("Generalization to higher frequencies")
    plt.grid(True)
    path = os.path.join(outdir, "frequency_generalization.png")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_fem_ablation(lambdas, psnrs_l, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(lambdas, psnrs_l, marker='o')
    plt.xscale('log'); plt.xlabel("λ_FEM"); plt.ylabel("PSNR (dB)")
    plt.title("Ablation of FEM smoothness weight")
    plt.grid(True, which="both")
    path = os.path.join(outdir, "fem_ablation.png")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_reconstruction_3d(x1, x2, x3, y, obs_idx, y_obs, y_hat, mse, psnr, N, M, sigma, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(18, 6))
    
    # Ground truth
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    s1 = ax1.scatter(x1.flatten(), x2.flatten(), x3.flatten(), c=y.flatten(), cmap='viridis', s=1)
    ax1.set_title("Ground Truth")
    fig.colorbar(s1, ax=ax1, shrink=0.5)

    # Observed
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    y_obs_plot = y_obs.copy()
    y_obs_plot[y_obs_plot == 0] = np.nan
    s2 = ax2.scatter(x1[obs_idx[:,0], obs_idx[:,1], obs_idx[:,2]], x2[obs_idx[:,0], obs_idx[:,1], obs_idx[:,2]], x3[obs_idx[:,0], obs_idx[:,1], obs_idx[:,2]], c=y_obs[obs_idx[:,0], obs_idx[:,1], obs_idx[:,2]], cmap='viridis', s=5, label="Observed Points")
    ax2.set_title("Noisy Observations")
    fig.colorbar(s2, ax=ax2, shrink=0.5)

    # Reconstruction
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    s3 = ax3.scatter(x1.flatten(), x2.flatten(), x3.flatten(), c=y_hat.flatten(), cmap='viridis', s=1)
    ax3.set_title(f"VAE Reconstruction (PSNR: {psnr:.2f} dB)")
    fig.colorbar(s3, ax=ax3, shrink=0.5)

    plt.suptitle(f"Gappy-VAE 3D Reconstruction (N={N}x{N}x{N}, M={M}, σ={sigma})")
    path = os.path.join(outdir, f"gappy_recon_3d_N{N}_M{M}_sigma{sigma}.png")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.close()
    return path
