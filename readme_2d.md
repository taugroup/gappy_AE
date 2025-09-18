# Understanding `main_2d.py`

This document provides a detailed explanation of the `main_2d.py` script. The script's primary purpose is to train a Convolutional Variational Autoencoder (CVAE) to reconstruct a 2D surface from a sparse and noisy set of observations.

## 1. Dataset Generation

The dataset is synthetically generated. Each data sample represents a 2D surface.

### 1.1. Surface Sampling (`sample_surface` function)

- **Purpose:** To create a smooth 2D surface.
- **Method:** The surface is defined by the following function:
  `y(x1, x2) = A * s(2π(f1*x1 + f2*x2) + φ)`
  - `(x1, x2)` are coordinates on a 2D grid of size `N x N`, where `x1` and `x2` range from 0.0 to 1.0.
  - `s` is either a `sin` or `cos` function, chosen randomly.
  - `A` (amplitude) is a random value between 0.8 and 1.2.
  - `f1` and `f2` (frequencies) are random integers chosen from {1, 2, 3}.
  - `φ` (phase) is a random value between 0 and 2π.
- **Output:** A tuple containing the `(x1, x2)` grid and the corresponding `y` values of the surface.

### 1.2. Gappy Sampling (`gappy_sample` function)

- **Purpose:** To simulate a real-world scenario where we only have a limited number of noisy measurements of the surface.
- **Method:**
  1.  **Select Observation Points:** `M` random points are chosen from the `N x N` grid.
  2.  **Add Noise:** Gaussian noise with a mean of 0 and a standard deviation of `sigma` is added to the `y` values at these `M` observation points.
  3.  **Create Mask:** A binary mask of size `N x N` is created. It has a value of 1 at the `M` observed locations and 0 elsewhere.
  4.  **Zero-out Unobserved Points:** The `y` values at the unobserved locations are set to 0.
- **Output:**
  - `y_obs_masked`: The `y` values with noise added at observed points and zeros elsewhere.
  - `mask`: The binary mask indicating the observed points.
  - `obs_idx`: The indices of the `M` observed points.

### 1.3. `SurfaceGappyDataset` Class

- **Purpose:** To create a PyTorch `Dataset` that can be used with a `DataLoader`.
- **`__getitem__` method:** For each sample, this method:
  1.  Generates a surface using `sample_surface`.
  2.  Creates a gappy sample from the surface using `gappy_sample`.
  3.  Concatenates the `y_obs_masked` and the `mask` to form the input tensor of shape `(2, N, N)`.
- **Output of `__getitem__`:**
  - `inp`: The input tensor for the model, of shape `(2, N, N)`.
  - `y`: The ground truth (original, complete) surface, of shape `(N, N)`.
  - `mask`: The observation mask, of shape `(N, N)`.

## 2. Model Architecture (`GappyConvVAE2D` class)

The script uses a Convolutional Variational Autoencoder (CVAE) to learn a compressed representation of the surface and then reconstruct it.

### 2.1. Encoder

- **Input:** A tensor of shape `(B, 2, N, N)`, where `B` is the batch size, and the 2 channels are `y_obs_masked` and `mask`.
- **Architecture:**
  - A series of `depth` (default 3) convolutional layers (`nn.Conv2d`).
  - Each convolutional layer is followed by a ReLU activation function.
  - The number of filters in the convolutional layers is defined by the `width` list (default `[32, 64, 128]`).
  - After the convolutional layers, the output is flattened.
- **Output:** The flattened feature vector is passed through two separate fully connected layers (`nn.Linear`) to produce:
  - `mu`: The mean of the latent distribution.
  - `logvar`: The log variance of the latent distribution.

### 2.2. Reparameterization Trick

- **Purpose:** To allow gradients to flow back through the sampling process.
- **Method:** The latent vector `z` is sampled from the learned distribution using the reparameterization trick:
  `z = mu + eps * std`, where `std = exp(0.5 * logvar)` and `eps` is a random sample from a standard normal distribution.

### 2.3. Decoder

- **Input:** The latent vector `z` of size `latent_dim` (default 32).
- **Architecture:**
  - A fully connected layer to project the latent vector back to the flattened size of the last encoder layer.
  - An `nn.Unflatten` layer to reshape the vector into a 4D tensor.
  - A series of `depth` (default 3) transposed convolutional layers (`nn.ConvTranspose2d`) to upsample the feature maps back to the original `N x N` size.
  - Each transposed convolutional layer is followed by a ReLU activation function (except the last one).
- **Output:** A reconstructed surface `y_hat` of shape `(B, 1, N, N)`.

## 3. Training Process (`train_gappy_conv_vae_2d` function)

### 3.1. Loss Function

The total loss is a combination of three components:

1.  **Full Reconstruction Loss (`loss_full`):** The Mean Squared Error (MSE) between the reconstructed surface `y_hat` and the ground truth surface `y`. This encourages the model to reconstruct the entire surface accurately.
2.  **Observed Reconstruction Loss (`loss_obs`):** The MSE between the reconstructed values at the observed locations (`y_hat * mask`) and the ground truth values at those locations (`y * mask`). This puts extra emphasis on getting the observed points right.
3.  **Kullback-Leibler (KL) Divergence Loss (`loss_kl`):** This is a regularization term that forces the learned latent distribution to be close to a standard normal distribution. It is calculated as:
   `loss_kl = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))`

The final loss is a weighted sum of these three components:
`loss = loss_full + obs_lambda * loss_obs + kl_lambda * loss_kl`

- `obs_lambda` (default 1.0) and `kl_lambda` (default 1e-6) are hyperparameters that control the weight of the observed reconstruction loss and the KL divergence loss, respectively.

### 3.2. Optimizer

- The script uses the Adam optimizer (`torch.optim.Adam`) to update the model's weights.

### 3.3. Training Loop (`run_epoch` function)

- The training process iterates for a specified number of `epochs`.
- In each epoch, the script iterates through the training data, calculates the loss, and updates the model's parameters using backpropagation.
- After each epoch, the model is evaluated on a validation set to monitor its performance and prevent overfitting. The best validation loss is tracked.

## 4. Evaluation and Visualization

### 4.1. `demo_once` function

- **Purpose:** To demonstrate the model's performance on a single example.
- **Method:**
  1.  Generates a new surface and a gappy sample.
  2.  Passes the gappy sample through the trained model to get a reconstruction.
  3.  Calculates the Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) between the reconstructed surface and the ground truth.
  4.  Calls `results.plot_reconstruction_2d` to generate and save a plot showing the ground truth, the observed data, and the reconstructed surface.

### 4.2. `observation_sweep_analysis` function

- **Purpose:** To analyze how the model's performance changes with the number of observations (`M`).
- **Method:**
  1.  It trains a separate model for each value of `M` in a predefined list (`M_values`).
  2.  For each trained model, it calls `demo_once` to get the PSNR.
  3.  It then calls `results.plot_observation_sweep` to create and save a plot of PSNR vs. `M`.

## 5. Main Execution (`main` function)

- The `main` function is the entry point of the script.
- It sets up logging using the `loguru` library.
- It calls `observation_sweep_analysis` to run the main experiment.
- The hyperparameters for the training and the experiment are defined within the `main` and `observation_sweep_analysis` functions.
