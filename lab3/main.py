import logging
import shutil
import sys
import os

from ucimlrepo import fetch_ucirepo 
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.metrics import mse, rmse, r2
from core.utils import timeseries_train_test_split, batch_split, plot_curves
from core.losses import AbstractLoss, MSELoss
from core.optimizers import AbstractOptimizer, Adam
from core.models import AbstractModel, RNNReg, LSTMReg, GRUReg
from core.data import Tensor
  
def train_fn(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    model: AbstractModel,
    loss_fn: AbstractLoss,
    optimizer: AbstractOptimizer
)-> np.ndarray:
    train_stats =  {"loss": [], "mse": [], "rmse": [], "r2": []}
    for train_Xb, train_yb in batch_split(X_train, y_train, batch_size = BATCH_SIZE): 
        if len(train_Xb) != BATCH_SIZE:
            continue

        # Do forward pass
        y_pred = model(train_Xb.transpose(1, 0, 2))
        # Calculate loss
        loss = loss_fn(y_pred, train_yb).to_numpy()
        train_stats["loss"].append(loss)

        # Do backward pass
        loss_fn.backward(y_pred, train_yb)
        # Update params
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad()

        # Convert results to numpy array
        y_pred = y_pred.to_numpy()
        train_yb = train_yb.to_numpy()

        # Calculate metrics
        train_stats["mse"].append(mse(y_pred, train_yb))
        train_stats["rmse"].append(rmse(y_pred, train_yb))
        train_stats["r2"].append(r2(y_pred, train_yb))

    logging.info(f"train_loss: {np.array(train_stats['loss']).mean()}")
    return train_stats

def test_fn(
    X_test: np.ndarray, 
    y_test: np.ndarray,
    model: AbstractModel,
    loss_fn: AbstractLoss,
)-> np.ndarray:
    test_stats =  {"loss": [], "mse": [], "rmse": [], "r2": []}
    for test_Xb, test_yb in batch_split(X_test, y_test, batch_size = BATCH_SIZE): 
        if len(test_Xb) != BATCH_SIZE:
            continue

        # Do forward pass
        y_pred = model(test_Xb.transpose(1, 0, 2))
        # Calculate loss
        loss = loss_fn(y_pred, test_yb).to_numpy()
        test_stats["loss"].append(loss)

        # Convert results to numpy array
        y_pred = y_pred.to_numpy()
        test_yb = test_yb.to_numpy()

        # Calculate metrics
        test_stats["mse"].append(mse(y_pred, test_yb))
        test_stats["rmse"].append(rmse(y_pred, test_yb))
        test_stats["r2"].append(r2(y_pred, test_yb))

    logging.info(f"test_loss: {np.array(test_stats['loss']).mean()}")
    return test_stats

if __name__ == "__main__":
    # Init constants
    WINDOW_SIZE = 6
    TEST_SIZE = 0.3
    TEST_STEP = 3
    EPOCHS = 25
    BATCH_SIZE = 64
    LR = 1e-4
    DEVICE = "cuda:0"
    DTYPE = "fp32"

    # Create directory for results
    results_path = f"{os.getcwd()}/lab3/results"
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
    dataset = fetch_ucirepo(id=851) 
    X = dataset.data.features.copy()

    # Preprocess dataset
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = (X[num_cols] - X[num_cols].min()) / (X[num_cols].max() - X[num_cols].min())
    cat_columns = X.select_dtypes(['object']).columns
    X[cat_columns] = X[cat_columns].astype('category')
    X[cat_columns] = X[cat_columns].apply(lambda col: col.cat.codes)
    y = X["Usage_kWh"]
    X = X.drop("Usage_kWh", axis=1)
    # Set needed precision for data
    X = X.astype('float32').to_numpy()
    y = y.astype('float32').to_numpy()

    # convert to numpy
    X = Tensor(X, dtype = DTYPE, device=DEVICE)
    y = Tensor(y, dtype = DTYPE, device=DEVICE)
    # Split data on train/test
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, TEST_SIZE, WINDOW_SIZE)

    # init models
    rnn = RNNReg(input_size = X_train.shape[-1], hidden_size = 16, num_layers = 4).to_device(DEVICE)
    lstm = LSTMReg(input_size = X_train.shape[-1], hidden_size = 16, num_layers = 4).to_device(DEVICE)
    gru = GRUReg(input_size = X_train.shape[-1], hidden_size = 16, num_layers = 4).to_device(DEVICE)

    # train RNN, LSTM, GRU
    for model_idx, model in enumerate([rnn, lstm, gru]):
        # update results path
        model_results_path = f"{results_path}/model{model_idx}"
        if os.path.exists(model_results_path):
            shutil.rmtree(model_results_path)
        os.makedirs(model_results_path)

        # init parts
        loss_fn = MSELoss(model = model)
        optimizer = Adam(model = model, lr = LR)

        # Train loop
        train_stats_by_epochs = {"loss": [], "mse": [], "rmse": [], "r2": []}
        test_stats_by_epochs = {"loss": [], "mse": [], "rmse": [], "r2": []}
        for epoch in range(EPOCHS):
            logging.info(f"epoch: {epoch+1}")
            train_stats = train_fn(X_train, y_train, model, loss_fn, optimizer)
            # Add train statistics
            for key, val_lst in train_stats.items():
                if len(val_lst) == 0: continue
                if key == "loss":
                    val_lst = np.mean(val_lst)
                else:
                    val_lst = np.array(val_lst)[-1]

                train_stats_by_epochs[key].append(val_lst)

            # Test model each TEST_STEP iteration
            if (epoch + 1) % TEST_STEP == 0:
                test_stats = test_fn(X_test, y_test, model, loss_fn)
                # Add test statistics
                for key, val_lst in test_stats.items():
                    if len(val_lst) == 0: continue
                    if key == "loss":
                        val_lst = np.mean(val_lst)
                    else:
                        val_lst = np.array(val_lst)[-1]

                    test_stats_by_epochs[key].append(val_lst)

        # draw training/testing stats
        plot_curves(
            np.arange(len(train_stats_by_epochs["loss"])),
            train_stats_by_epochs["loss"],
            "Train loss plot",
            "epochs",
            "Train loss",
            f"{model_results_path}/train_loss.png"
        )
        plot_curves(
            np.arange(len(test_stats_by_epochs["loss"])),
            test_stats_by_epochs["loss"],
            "Test loss plot",
            "epochs",
            "Test loss",
            f"{model_results_path}/test_loss.png"
        )

        # Calculate mean metrics
        file_paths = [f"{model_results_path}/train_metrics.txt", f"{model_results_path}/test_metrics.txt"]
        for stats, file_path in zip([train_stats_by_epochs, test_stats_by_epochs], file_paths):
            with open(file_path, "w", encoding = "utf-8") as f:
                for key, val in stats.items():
                    if key == "loss" or len(val) == 0: continue
                    f.write(f"{key}:\n")
                    f.write(f"{np.array(val).mean()}\n")
                    f.write('\n')

        logging.info('\n')