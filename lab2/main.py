from typing import Any
import logging
import shutil
import sys
import os

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.metrics import accuracy, precision, recall, f1, calculate_tpr_fpr_for_curve
from core.utils import train_test_split, batch_split, plot_curves, plot_tsne, pad_2d_data
from core.losses import AbstractLoss, CrossEntropyLoss
from core.optimizers import SGD
from core.models import AbstractModel, LeNet5
  
def train_fn(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    model: AbstractModel,
    loss_fn: AbstractLoss,
    optimizer: Any
)-> np.ndarray:
    train_stats =  {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    for train_Xb, train_yb in batch_split(X_train, y_train, batch_size = BATCH_SIZE): 
        # Do forward pass
        y_pred = model(train_Xb)
        # Calculate loss
        loss = loss_fn(y_pred, train_yb).mean()
        train_stats["loss"].append(loss)

        # Do backward pass
        loss_fn.backward(y_pred, train_yb, model = model)
        # Update params
        optimizer.step(model.parameters())
        # Reset gradients
        optimizer.zero_grad(model.parameters())

        # Calculate metrics
        num_cls = y_pred.shape[1]
        y_pred = np.argmax(y_pred, axis=1)
        train_stats["acc"].append(accuracy(y_pred, train_yb, num_cls))
        train_stats["prec"].append(precision(y_pred, train_yb, num_cls))
        train_stats["rec"].append(recall(y_pred, train_yb, num_cls))
        train_stats["f1"].append(f1(y_pred, train_yb, num_cls))

    logging.info(f"train_loss: {np.array(train_stats['loss']).mean()}")
    return train_stats

def test_fn(
    X_test: np.ndarray, 
    y_test: np.ndarray,
    model: AbstractModel,
    loss_fn: AbstractLoss,
)-> np.ndarray:
    test_stats =  {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    for test_Xb, test_yb in batch_split(X_test, y_test, batch_size = BATCH_SIZE): 
        # Do forward pass
        y_pred = model(test_Xb)
        # Calculate loss
        loss = loss_fn(y_pred, test_yb).mean()
        test_stats["loss"].append(loss)

        # Calculate metrics
        num_cls = y_pred.shape[1]
        y_pred = np.argmax(y_pred, axis=1)
        test_stats["acc"].append(accuracy(y_pred, test_yb, num_cls))
        test_stats["prec"].append(precision(y_pred, test_yb, num_cls))
        test_stats["rec"].append(recall(y_pred, test_yb, num_cls))
        test_stats["f1"].append(f1(y_pred, test_yb, num_cls))

    logging.info(f"test_loss: {np.array(test_stats['loss']).mean()}")
    return test_stats

if __name__ == "__main__":
    # Init constants
    TEST_SIZE = 0.3
    TEST_STEP = 2
    EPOCHS = 30
    BATCH_SIZE = 64
    LR = 1e-4
    # Create directory for results
    results_path = f"{os.getcwd()}/lab2/results"
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

    # apply t-sne to visualize mnist data
    n_samples = 3000
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_scaled = StandardScaler().fit_transform(X.iloc[indices])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)
    plot_tsne(
        X_embedded[:, 0],
        X_embedded[:, 1],
        y.iloc[indices],
        "MNIST Visualization using t-SNE",
        "t-SNE Component 1",
        "t-SNE Component 2",
        f"{results_path}/t-SNE.png"
    )

    # convert to numpy
    X = pad_2d_data(X.to_numpy().reshape(-1, 1, 28, 28), 2)
    y = y.to_numpy()
    # Split data on train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE)
    # init parts
    loss_fn = CrossEntropyLoss()
    model = LeNet5(in_channels = 1, out_channels = 10)
    optimizer = SGD(lr = LR)

    # Train loop
    train_stats_by_epochs = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    test_stats_by_epochs = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    for epoch in range(EPOCHS):
        logging.info(f"epoch: {epoch+1}")
        train_stats = train_fn(X_train, y_train, model, loss_fn, optimizer)
        # Add train statistics
        for key, val_lst in train_stats.items():
            if len(val_lst) == 0: continue
            val_lst = np.array(val_lst)
            train_stats_by_epochs[key].append(np.mean(val_lst, axis = 0))

        # Test model each TEST_STEP iteration
        if (epoch + 1) % TEST_STEP == 0:
            test_stats = test_fn(X_test, y_test, model, loss_fn)
            # Add test statistics
            for key, val_lst in test_stats.items():
                if len(val_lst) == 0: continue
                val_lst = np.array(val_lst)
                test_stats_by_epochs[key].append(np.mean(val_lst, axis = 0))

    # draw training/testing stats
    plot_curves(
        np.arange(len(train_stats_by_epochs["loss"])),
        train_stats_by_epochs["loss"],
        "Train loss plot",
        "epochs",
        "Train loss",
        f"{results_path}/train_loss.png"
    )
    plot_curves(
        np.arange(len(test_stats_by_epochs["loss"])),
        test_stats_by_epochs["loss"],
        "Test loss plot",
        "epochs",
        "Test loss",
        f"{results_path}/test_loss.png"
    )

    # Calculate mean metrics
    file_paths = [f"{results_path}/train_metrics.txt", f"{results_path}/test_metrics.txt"]
    for stats, file_path in zip([train_stats_by_epochs, test_stats_by_epochs], file_paths):
        with open(file_path, "w", encoding = "utf-8") as f:
            for key, val in stats.items():
                if key == "loss" or len(val) == 0: continue
                f.write(f"{key}:\n")
                val = np.array(val)
                for cls_idx in range(val.shape[1]):
                    f.write(f"{cls_idx}: {val[:, cls_idx].mean()}\n")
                f.write('\n')

    # calculate tpr/fpr for ROC
    y_probs = []
    y_labels = []
    for X_batch, y_batch in batch_split(X_test, y_test, batch_size = BATCH_SIZE):
        probs = model(X_batch)
        y_probs.append(probs)
        y_labels.append(y_batch)

    y_probs = np.concatenate(y_probs)
    y_labels = np.concatenate(y_labels)
    # convert labels to one hot
    n_classes = y_probs.shape[1]
    y_true_one_hot = label_binarize(y_labels, classes=np.arange(n_classes))

    for c in range(n_classes):
        tpr, fpr, auc = calculate_tpr_fpr_for_curve(y_true_one_hot[:, c].flatten(), y_probs[:, c].flatten())
        # draw ROC
        plot_curves(
            fpr,
            tpr,
            f"ROC for class: {c}, AUC: {auc}",
            "fpr",
            "tpr",
            f"{results_path}/roc_auc_cls_{c}.png"
        )