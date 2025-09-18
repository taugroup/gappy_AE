# Gappy Signal Reconstruction with an Autoencoder

This project demonstrates how to use a neural network (a simple autoencoder) to reconstruct a smooth 1D signal from a small number of sparse, noisy observations. The model is built using PyTorch and includes an optional, novel smoothness penalty based on the Finite Element Method (FEM) to encourage more plausible reconstructions.

The primary script is `main.ipynb`, a Jupyter Notebook that contains all the code for data generation, model definition, training, and evaluation.

## Overview

The core problem is to reconstruct a full sine/cosine wave given only a small subset of its points, which have also been corrupted with random noise.

The approach uses a simple Multi-Layer Perceptron (MLP) based autoencoder:
1.  **Input**: The input to the model is a vector of size `2*N`, containing:
    *   The "gappy" signal of length `N`, where unobserved points are zero.
    *   A binary "mask" of length `N`, indicating which points were observed.
2.  **Model**: The autoencoder maps this input through several hidden layers to a dense output vector.
3.  **Output**: The output is the reconstructed full signal of length `N`.
4.  **Loss Function**: The model is trained to minimize a composite loss function:
    *   **Reconstruction Loss**: The Mean Squared Error (MSE) between the model's output and the original, clean signal.
    *   **Observation Loss**: An optional, weighted MSE calculated on only the observed points.
    *   **Smoothness Loss**: An optional, weighted penalty that discourages "wiggly" or non-smooth outputs, calculated using a 1D FEM stiffness matrix from `scikit-fem`.

## Dependencies

The project requires the following Python libraries:
- `numpy`
- `scipy`
- `torch`
- `matplotlib`
- `scikit-fem`

You can install all dependencies by running the first cell in the `main.ipynb` notebook, or by running the following command in your terminal:

```bash
pip install numpy scipy torch matplotlib "scikit-fem[all]"
```

## Running the Notebook

1.  Ensure you have Jupyter Notebook or JupyterLab installed.
2.  Launch the notebook server and open `main.ipynb`.
3.  You can run the cells sequentially ("Run All").

The notebook will:
1.  Install the required dependencies.
2.  Define all necessary functions and classes.
3.  Train the autoencoder model, printing the training and validation loss at each epoch.
4.  Display a plot (`gappy_recon_notebook.png`) showing the final reconstruction of a sample curve against the ground truth and the noisy input.
5.  Run several additional experiments to analyze the model's performance under different conditions.

## Code Structure (`main.ipynb`)

The notebook is organized into logical cells:

- **Cell 1: Installation**: Installs all required libraries using `pip`.
- **Cell 2: Imports and Setup**: Imports all modules, sets the random seeds for reproducibility, and configures the device (GPU or CPU) for PyTorch.
- **Cell 3: Data Generation Functions**:
    - `sample_curve`: Creates a clean, ground-truth sine or cosine wave with random parameters.
    - `gappy_sample`: Takes a clean curve and produces a sparse, noisy version to be used as the model's input.
- **Cell 4: PyTorch Dataset**:
    - `SineGappyDataset`: A custom `Dataset` class that wraps the data generation logic, making it compatible with PyTorch's `DataLoader`.
- **Cell 5: Model Definition**:
    - `GappyAE`: Defines the autoencoder architecture as a simple MLP with configurable depth and width.
- **Cell 6: Smoothness Loss (FEM)**:
    - `fem_stiffness_1d`: Assembles a 1D FEM stiffness matrix, which is used to measure the "smoothness" of a signal.
    - `smoothness_loss`: Calculates a penalty based on the output signal's lack of smoothness. This is a key feature for encouraging realistic reconstructions.
- **Cell 7: Training and Evaluation Functions**:
    - `train_gappy_ae`: The main training function that orchestrates the model initialization, training loop, loss calculation, and optimization.
    - `demo_once`: A function to perform inference on a single example, calculate performance metrics (MSE, PSNR), and generate a plot of the result.
- **Cell 8: Main Execution Block**:
    - This cell contains the main configuration parameters (signal length, noise level, epochs, etc.).
    - It calls `train_gappy_ae` to start training and then `demo_once` to visualize the result.
- **Cells 9-20: Experiments and Analysis**:
    - These cells conduct a series of "stress tests" and ablation studies to provide a deeper understanding of the model's capabilities:
        - **Robustness**: Testing with fewer observations and higher noise.
        - **Masking Patterns**: Evaluating performance on different kinds of "gaps" (random, block, clusters).
        - **Generalization**: Checking if the model can reconstruct signals with higher frequencies than seen during training.
        - **Ablation Study**: Comparing performance with and without the FEM smoothness penalty to demonstrate its impact.
