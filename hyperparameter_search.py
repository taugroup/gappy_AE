
import random
from loguru import logger

import main


def hyperparameter_search():
    logger.add("hyperparameter_search.log")

    hyperparameter_space = {
        'latent_dim': [16, 32, 64],
        'width': [128, 256, 512],
        'depth': [3, 4, 5],
        'lr': [1e-3, 1e-4],
        'kl_lambda': [1e-4, 1e-5, 1e-6]
    }

    for i in range(10):
        logger.info(f"Experiment {i+1}/10")

        config = {
            'latent_dim': random.choice(hyperparameter_space['latent_dim']),
            'width': random.choice(hyperparameter_space['width']),
            'depth': random.choice(hyperparameter_space['depth']),
            'lr': random.choice(hyperparameter_space['lr']),
            'kl_lambda': random.choice(hyperparameter_space['kl_lambda'])
        }

        logger.info(f"Config: {config}")

        model = main.train_gappy_vae(
            N=256, M=32, sigma=0.05, epochs=50, bs=128, lr=config['lr'],
            width=config['width'], depth=config['depth'], latent_dim=config['latent_dim'],
            obs_lambda=1.0, fem_lambda=1e-3, kl_lambda=config['kl_lambda'], seed=main.SEED
        )

        psnr = main.demo_once(model, N=256, M=32, sigma=0.05, seed=main.SEED+123)
        logger.info(f"Config: {config} | PSNR: {psnr}")

if __name__ == "__main__":
    hyperparameter_search()
