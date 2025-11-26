# MFEM Temporal Reconstruction Pipeline

This document explains how the MFEM-driven workflow is wired together, including
dataset generation, sensor placement, network architecture, and the overall
reconstruction flow implemented in `2d_temporal.py`, `2d_temporal_mfem.py`, and
the supporting utilities.

## 1. High-Level Flow

1. **Run MFEM Example 9** (`PyMFEM/examples/ex9.py`) to produce ParaView (`.vtu`)
   snapshots of a convected scalar field on a refined mesh.
2. **Convert the ParaView outputs** into regular grids plus optional gradients
   with `MFEM_Dataset/make_ex9_dataset.py`. The result is a `.npz` file
   containing `fields`, `grid_x/y`, and `times`.
3. **Train and evaluate the temporal reconstructor** via `2d_temporal.py`,
   optionally orchestrated end-to-end by `2d_temporal_mfem.py` (which handles
   running MFEM, conversion, logging, and demos in `MFEM_results/`).
4. **(Optional) Execute sweeps/experiments** using helper scripts like
   `sensor_sweep.py`, which loop over sensor counts, parse metrics, and plot
   trends.

All generated artifacts (ParaView dumps, datasets, logs, recon figures, sweep
results) are rooted in `MFEM_results/` so the repository stays organized.

## 2. Dataset Generation (`2d_temporal_mfem.py`, `make_ex9_dataset.py`)

`2d_temporal_mfem.py` wraps the three major steps:

1. **MFEM Simulation**: launches `PyMFEM/examples/ex9.py` with the requested mesh,
   refinement, time horizon, and time step. Passing `--paraview-datafiles`
   instructs MFEM to write `ParaView/Example9/Cycle*/proc*.vtu`.
2. **Grid Conversion**: calls `MFEM_Dataset/make_ex9_dataset.py`, which:
   - parses `Example9.pvd` to discover time steps,
   - loads each `proc*.vtu` group with `meshio`,
   - interpolates the unstructured point data onto a regular `resolution x resolution`
     grid (defaults to 64 but we often use 96),
   - optionally computes spatial gradients (`--include-gradients`),
   - stores arrays as `(snapshots, channels, N, N)` in `.npz`.
3. **Temporal Training**: sets `GAPPY_TEMPORAL_DATASET` to the new `.npz` file and
   executes `2d_temporal.py` with any extra environment overrides (epochs, batch
   size, demo length, sensor usage, etc.).

The script also supports `--skip-ex9`, `--skip-convert`, and `--only-generate`
flags, so you can reuse existing simulations or datasets without rerunning MFEM.

## 3. Sensor Placement & Sampling

### 3.1 Random / Fixed Sensors

Inside `2d_temporal.py`, `FlowGappyDataset` (synthetic) and
`PrecomputedFlowDataset` (for MFEM data) both call `gappy_sample_flow`. Given a
full field tensor `(channels, N, N)` and a sensor mask size `M`, this utility:

1. randomly selects `M` grid coordinates (unless a fixed `sensor_indices` array
   is supplied),
2. retrieves the true values at those coordinates,
3. adds Gaussian noise per channel (`sigma`),
4. creates a binary mask map so unobserved cells are zeroed out.

The network input concatenates the noisy sparse measurements and the binary
mask, producing a tensor of shape `(2 * channels, N, N)` that encodes both data
and sampling pattern.

### 3.2 pysensors Optimization

If `GAPPY_TEMPORAL_USE_PYSENSORS=1`, the script enables the pysensors pipeline:

1. **Snapshot pool**: flatten synthetic or MFEM fields into vectors and build two
   datasets (train/validation) for sensor ranking.
2. **Basis reduction**: apply SVD to derive POD bases up to
   `GAPPY_TEMPORAL_SENSOR_BASIS` modes.
3. **QR/SSPOR optimization**: run the SSPOR optimizer (pysensors) to rank grid
   points that best capture the basis variations.
4. **Selection**: evaluate reconstruction errors for each candidate sensor count
   (e.g., `[32,64,96,128,192,256]`) and pick the best number + coordinates.
5. **Reuse**: the chosen indices `sensor_indices` are passed to
   `PrecomputedFlowDataset`, ensuring every training sample uses the same sensor
   locations instead of random draws.

Sensor rankings and error curves are saved in
`MFEM_results/temporal_results/optimized_sensors.npy` and
`sensor_error_curve.csv` for traceability.

## 4. Network Architecture (`GappyConvVAE2D`)

`2d_temporal.py` defines a convolutional Variational Autoencoder tailored to
sparse-field reconstruction:

1. **Encoder**:
   - Input channels: `field_channels * 2` (observed values + mask per channel).
   - `depth` blocks of `Conv2d -> ReLU` with stride 2 and kernel 5 (default widths
     `[32,64,128]`).
   - Flattened output fed to two dense heads producing `mu` and `logvar` of latent
     dimension 32.
2. **Latent Sampling**: standard reparameterization `z = mu + eps * exp(0.5*logvar)`.
3. **Decoder**:
   - Dense layer maps `z` back to the flattened feature size.
   - `Unflatten -> ConvTranspose` blocks mirror the encoder to upsample back to
     `(field_channels, N, N)`.

The loss combines:

- **Full reconstruction MSE** over entire fields,
- **Observed reconstruction MSE** weighted by `GAPPY_TEMPORAL_OBS_LMB`,
- **KL divergence** scaled by `GAPPY_TEMPORAL_KL_LMB`.

This encourages faithful reconstructions globally while tightly matching the
observed measurements.

## 5. Reconstruction & Demos

After training, `demo_temporal` runs either synthetic scenarios or sweeps through
the MFEM time indices:

1. Create or reuse a scenario (for synthetic data) or loop through the
   precomputed fields with their timestamps.
2. Apply the sensor mask (random or optimized) to produce gappy observations.
3. Feed the concatenated observation+mask tensor through the VAE to obtain
   reconstructed fields.
4. Compute metrics (MSE, global PSNR, per-channel PSNR) and log them.
5. Use `results.plot_flow_reconstruction` to render each timestep, then assemble
   them into `gappy_flow_recon_temporal.gif` per demo run.

All demo outputs from the MFEM orchestration end up in
`MFEM_results/temporal_results/flow_run_XX_seed_YYY/`, making it easy to inspect
different seeds or sensor strategies.

## 6. Sensor Sweeps (`sensor_sweep.py`)

`sensor_sweep.py` automates experiments over a range of sensor counts:

1. Iterates over powers of two (configurable) and runs
   `2d_temporal_mfem.py --skip-ex9 --skip-convert` with a fixed dataset.
2. Captures each run's `log/` and `temporal_results/` into
   `MFEM_results/<output-root>/M_<count>/`.
3. Parses `2d_temporal.log` to extract best validation loss and PSNR metrics.
4. Writes `summary.csv` plus optional channel-wise PSNR tables.
5. Optional plotting utilities (the provided scripts) create PNG charts for loss
   vs. sensors and PSNR vs. sensors.

This infrastructure makes it easy to study how reconstruction quality scales
with observation density or training budget.

## 7. Repro Tips

- Activate the `gappy` conda environment (`conda activate gappy`) before running
  anything to ensure dependencies (PyTorch, PyMFEM, meshio, pysensors) are
  available.
- For fresh MFEM simulations, ensure write permissions inside `MFEM_results/`
  because ParaView output is redirected there.
- When sweeping or rerunning experiments, prefer `--skip-ex9`/`--skip-convert`
  to save time and pass environment overrides via `--gappy-env KEY=VALUE`.

With these components, the workflow handles everything from raw MFEM PDE solves
through sparse observation selection to autoencoder-based reconstructions and
hyperparameter studies.

