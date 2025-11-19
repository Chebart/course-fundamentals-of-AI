import logging
import shutil
import sys
import os

from sklearn.datasets import fetch_openml
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.utils import train_test_split, batch_split, plot_curves, pad_2d_data, save_mnist_grid
from core.losses import AbstractLoss, MSELoss, BCELoss
from core.optimizers import AbstractOptimizer, Adam
from core.models import AbstractModel, VAEEncoder, Generator, Discriminator
from core.data import Tensor
  
def train_fn(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    models: dict[AbstractModel],
    losses: dict[AbstractLoss],
    optimizers: dict[AbstractOptimizer]
)-> np.ndarray:
    train_stats = {"g_loss": [], "d_loss": [], "rec_loss": [], "kl_loss": []}
    for train_Xb, _ in batch_split(X_train, y_train, batch_size = BATCH_SIZE):
        # -------------------------
        # Train Discriminator
        # -------------------------
        real_preds = models["disc"](train_Xb)
        real_targets = Tensor.ones(real_preds.shape, dtype = real_preds.dtype, device = real_preds.device)
        loss_D_real = losses["d_loss"](real_preds, real_targets).mean().to_numpy()
        losses["d_loss"].backward(real_preds, real_targets)

        _, _, z = models["enc"](train_Xb)
        fake = models["gen"](z)
        fake_preds = models["disc"](fake)
        fake_targets = Tensor.zeros(fake_preds.shape, dtype = fake_preds.dtype, device = fake_preds.device)
        loss_D_fake = losses["d_loss"](fake_preds, fake_targets).mean().to_numpy()
        losses["d_loss"].backward(fake_preds, fake_targets)

        loss_D = loss_D_real + loss_D_fake
        train_stats["d_loss"].append(loss_D)
        optimizers["disc"].step()
        optimizers["disc"].zero_grad()

        # -------------------------
        # Train Encoder + Generator
        # -------------------------
        mu, sigma, z = models["enc"](train_Xb)
        recon = models["gen"](z)

        # Calculate reconstruction loss
        rec_loss = losses["rec_loss"](recon, train_Xb).mean().to_numpy()
        train_stats["rec_loss"].append(rec_loss)
        losses["rec_loss"].backward(recon, train_Xb)

        # Calculate generator loss
        preds = models["disc"](recon)
        targets = Tensor.ones(preds.shape, dtype = preds.dtype, device = preds.device)
        loss_G = losses["g_loss"](preds, targets).mean().to_numpy()
        train_stats["g_loss"].append(loss_G)
        dLdx_fake = losses["g_loss"].backward(preds, targets)
        dLdz_gen = models["gen"].backward(dLdx_fake)

        # Calculate KL loss
        kl_loss = -0.5 * (1 + sigma - mu**2 - sigma.exp()).mean().to_numpy()
        train_stats["kl_loss"].append(kl_loss)

        dKL_dmu = mu / mu.shape[0]
        dKL_dsigma = 0.5 * (sigma.exp() - 1) / sigma.shape[0]
        dLdz_gen_dmu = dLdz_gen
        dLdz_gen_dsigma = dLdz_gen * models["enc"].eps * (0.5 * models["enc"].std)
        dmu_total = dLdz_gen_dmu + dKL_dmu
        dsigma_total = dLdz_gen_dsigma + dKL_dsigma
        d_encoder_out = Tensor.concat(
            [dmu_total, dsigma_total], 
            axis = 1, 
            dtype = dmu_total.dtype, 
            device = dmu_total.device
        )
        models["enc"].backward(d_encoder_out)

        optimizers["enc"].step()
        optimizers["gen"].step()
        optimizers["enc"].zero_grad()
        optimizers["gen"].zero_grad()
        optimizers["disc"].zero_grad()

    logging.info(f"train g_loss: {np.array(train_stats['g_loss']).mean()}")
    logging.info(f"train d_loss: {np.array(train_stats['d_loss']).mean()}")
    logging.info(f"train rec_loss: {np.array(train_stats['rec_loss']).mean()}")
    logging.info(f"train kl_loss: {np.array(train_stats['kl_loss']).mean()}")

    return train_stats

def test_fn(
    X_test: np.ndarray, 
    y_test: np.ndarray,
    models: list[AbstractModel],
    losses: list[AbstractLoss]
)-> np.ndarray:
    test_stats = {"g_loss": [], "d_loss": [], "rec_loss": [], "kl_loss": []}
    for test_Xb, _ in batch_split(X_test, y_test, batch_size = BATCH_SIZE): 
        # -------------------------
        # Test Discriminator
        # -------------------------
        real_preds = models["disc"](test_Xb)
        real_targets = Tensor.ones(real_preds.shape, dtype = real_preds.dtype, device = real_preds.device)
        loss_D_real = losses["d_loss"](real_preds, real_targets).mean().to_numpy()

        _, _, z = models["enc"](test_Xb)
        fake = models["gen"](z)
        fake_preds = models["disc"](fake)
        fake_targets = Tensor.zeros(fake_preds.shape, dtype = fake_preds.dtype, device = fake_preds.device)
        loss_D_fake = losses["d_loss"](fake_preds, fake_targets).mean().to_numpy()

        loss_D = loss_D_real + loss_D_fake
        test_stats["d_loss"].append(loss_D)

        # -------------------------
        # Test Encoder + Generator
        # -------------------------
        mu, sigma, z = models["enc"](test_Xb)
        recon = models["gen"](z)

        # Calculate losses
        rec_loss = losses["rec_loss"](recon, test_Xb).mean().to_numpy()
        test_stats["rec_loss"].append(rec_loss)

        kl_loss = -0.5 * (1 + sigma - mu.pow(2) - sigma.exp()).mean().to_numpy()
        test_stats["kl_loss"].append(kl_loss)

        preds = models["disc"](recon)
        targets = Tensor.ones(preds.shape, dtype = preds.dtype, device = preds.device)
        loss_G = losses["g_loss"](preds, targets).mean().to_numpy()
        test_stats["g_loss"].append(loss_G)

        # -------------------------
        # Save restored images
        # -------------------------
        save_mnist_grid(recon, test_Xb)

    logging.info(f"test g_loss: {np.array(test_stats['g_loss']).mean()}")
    logging.info(f"test d_loss: {np.array(test_stats['d_loss']).mean()}")
    logging.info(f"test rec_loss: {np.array(test_stats['rec_loss']).mean()}")
    logging.info(f"test kl_loss: {np.array(test_stats['kl_loss']).mean()}")
    return test_stats

if __name__ == "__main__":
    # Init constants
    Z_DIM = 64
    TEST_SIZE = 0.3
    TEST_STEP = 3
    EPOCHS = 1
    BATCH_SIZE = 64
    LR = 1e-4
    DEVICE = "cpu"
    DTYPE = "fp32"

    # Create directory for results
    results_path = f"{os.getcwd()}/lab4/results"
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    # Setup logger
    logging.basicConfig(
        filename=f"{results_path}/main.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get dataset
    mnist = fetch_openml('mnist_784')
    X = mnist.data.astype('float16')
    y = mnist.target.astype('int')

    # convert to numpy
    X = Tensor(pad_2d_data(X.to_numpy().reshape(-1, 1, 28, 28), 2), dtype = DTYPE, device=DEVICE)
    y = Tensor(y.to_numpy(), dtype = DTYPE, device=DEVICE)
    # Split data on train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE)
    # init parts VAEEncoder, Generator, Discriminator
    models = {
        "enc": VAEEncoder(in_channels = 1, z_dim = Z_DIM).to_device(DEVICE),
        "gen": Generator(z_dim = Z_DIM, out_channels = 1).to_device(DEVICE),
        "disc": Discriminator(in_channels = 1, out_features = 1).to_device(DEVICE)
    }
    losses = {
        "rec_loss": MSELoss(model = models["gen"]),
        "g_loss": BCELoss(model = models["disc"]), # since we calculate g_loss grad through disc
        "d_loss": BCELoss(model = models["disc"])
    }
    optimizers = {
        "enc": Adam(model = models["enc"], lr = LR, reg_type = "l2"),
        "gen": Adam(model = models["gen"], lr = LR, reg_type = "l2"),
        "disc": Adam(model = models["disc"], lr = LR, reg_type = "l2")
    }

    # Train loop
    train_stats_by_epochs = {"g_loss": [], "d_loss": [], "rec_loss": [], "kl_loss": []}
    test_stats_by_epochs = {"g_loss": [], "d_loss": [], "rec_loss": [], "kl_loss": []}
    for epoch in range(EPOCHS):
        logging.info(f"epoch: {epoch+1}")
        train_stats = train_fn(X_train, y_train, models, losses, optimizers)
        # Add train statistics
        for key, val_lst in train_stats.items():
            if len(val_lst) == 0: continue
            val_lst = np.array(val_lst)
            train_stats_by_epochs[key].append(np.mean(val_lst))

        # Test model each TEST_STEP iteration
        if (epoch + 1) % TEST_STEP == 0:
            test_stats = test_fn(X_test, y_test, models, losses)
            # Add test statistics
            for key, val_lst in test_stats.items():
                if len(val_lst) == 0: continue
                val_lst = np.array(val_lst)
                test_stats_by_epochs[key].append(np.mean(val_lst))

    # draw training/testing stats
    plot_curves(
        np.arange(len(train_stats_by_epochs["g_loss"])),
        np.array(list(train_stats_by_epochs.values())),
        "Train loss plot",
        "epochs",
        "Train loss",
        f"{results_path}/train_loss.png"
    )
    plot_curves(
        np.arange(len(test_stats_by_epochs["g_loss"])),
        np.array(list(test_stats_by_epochs.values())),
        "Test loss plot",
        "epochs",
        "Test loss",
        f"{results_path}/test_loss.png"
    )