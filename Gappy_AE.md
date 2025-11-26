# Dataset and Simulation Setup
- The study uses three 2-D PDE benchmarks generated with **MFEM**:
  - Nonlinear diffusion: MFEM ex16
  - Radial advection: MFEM ex9
  - Wave equation: MFEM ex23
  :contentReference[oaicite:0]{index=0}
- All problems are posed on a uniform \(64 \times 64\) mesh. Time-stepping:
  - Diffusion: DIRK, \(\Delta t = 0.002\)
  - Advection: RK4, \(\Delta t = 0.005\)
  - Wave: generalized-\(\alpha\), \(\Delta t = 0.01\)
  :contentReference[oaicite:1]{index=1}
- Parameter sampling:
  - Training set: \(D_{\text{train}} = \{0.80, 0.85, \dots, 1.20\}\)
  - Test set: \(D_{\text{test}} = \{0.75 + i/100\}_{i=0}^{50}\)
  :contentReference[oaicite:2]{index=2}
- Sensing layouts:
  - Boundary sensors and interior sensors for diffusion and wave
  - Interior sensors for advection (boundary value fixed)
  :contentReference[oaicite:3]{index=3}
- Noise setting: uniform noise in \([-0.01, 0.01)\), i.e. \(\pm 1\%\) of the maximum field value. :contentReference[oaicite:4]{index=4}

# Method: Gappy Auto-Encoder (Gappy AE)
- The field \(x \in \mathbb{R}^{N_s}\) is represented on a nonlinear manifold via a decoder \(g : \mathbb{R}^{n_s} \rightarrow \mathbb{R}^{N_s}\), trained by an auto-encoder on solution snapshots:
  \[
  x \approx x_{\text{ref}} + g(\hat{x})
  \]
  where \(\hat{x}\) is a low-dimensional code. :contentReference[oaicite:5]{index=5}
- Given sparse measurements, a sampling operator \(Z^\top\) extracts the observed entries. The latent vector \(\hat{x}\) is recovered from
  \[
  \min_{\hat{x}} \big\| Z^\top \big(x - g(\hat{x}) \big) \big\|_2^2
  \]
  solved by Gauss–Newton updates that use the decoder Jacobian. This replaces the linear subspace in Gappy POD with a nonlinear decoder. :contentReference[oaicite:6]{index=6}
- Sensor placement is tested with four schemes:
  1. Uniform
  2. Latin Hypercube Sampling (LHS)
  3. Oversampled DEIM
  4. S-OPT
  For DEIM and S-OPT, a residual basis \(\Phi_r\) is built from training residuals so that sampling is carried out on the residual, which yields a residual-weighted normal equation for \(\hat{x}\). :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
- Auto-encoder training:
  - Sparse, shallow AE
  - Latent dimension \(n_s \in \{3,4,5,6\}\)
  - Encoder width = 2 × input
  - Block size \(b = 100\), shift \(\delta_b = 20\)
  - Batch size 240, 80/20 split
  - ADAM, initial LR \(10^{-3}\), LR drops on plateau
  - Up to 10k epochs, early stopping
  - Each experiment repeated 10 times
  :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

# Results
- Relative reconstruction error (%) — **noiseless** case, averaged over test parameters:

  - **Diffusion**
    - Boundary sensors:
      - Gappy AE: **0.0896**
      - Gappy POD: **9.7730**
    - Interior sensors:
      - Gappy AE: **0.0753**
      - Gappy POD: **2.6964**
  - **Radial advection** (interior sensors):
      - Gappy AE: **0.0699**
      - Gappy POD: **7.8696**
  - **Wave**
    - Boundary sensors:
      - Gappy AE: **0.4102**
      - Gappy POD: **2.5992**
    - Interior sensors:
      - Gappy AE: **0.3272**
      - Gappy POD: **1.6759**
  :contentReference[oaicite:13]{index=13}

- Projection-error experiments (full measurements) show that Gappy AE has reconstruction errors smaller than Gappy POD by one to two orders of magnitude, for clean and noisy data. :contentReference[oaicite:14]{index=14}

- CPU runtime per reconstruction step (12 sensors, \(n_s \in [3,6]\)):
  - Gappy AE: about 3.0–3.8 ms
  - Gappy POD: about 0.12 ms
  - Full FE solve: tens of ms, depends on problem and sensor layout
  :contentReference[oaicite:15]{index=15}

- Sensor-layout behavior:
  - Interior sensing often improves accuracy for diffusion and wave
  - LHS or S-OPT tend to be more stable in the noisy setting
  - DEIM can drop in accuracy near \(\mu \approx 0.8\) for advection
  :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}
