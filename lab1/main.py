import sys
import os

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.metrics import accuracy, precision, recall, f1, calculate_tpr_fpr_for_curve
from core.utils import train_test_split, batch_split, plot_curves
from core.losses import HingeLoss, BCELoss
from core.optimizers import SGD
from model import MLP
  

if __name__ == "__main__":
    # Init constants
    TEST_SIZE = 0.2
    TEST_STEP = 2
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 1e-4

    # Get dataset
    mushroom = fetch_ucirepo(id=73) 
    X = mushroom.data.features
    y = mushroom.data.targets.squeeze()
    # Explore data
    X = X.replace("?", pd.NA)
    print(f"class distribution:\n {y.value_counts()}")
    print(f"\nmissing values:\n {X.isna().mean() * 100}")
    print(f"\nunique values count:")
    for col in X.columns:
        print(f"{col}: {X[col].unique()}")
    # Convert categorical features to numeric
    X["stalk_root_missing"] = X["stalk-root"].isna().astype(int)
    X = X.drop(columns=["stalk-root", "veil-type"])
    X_enc = pd.get_dummies(X, drop_first=False).astype(int)
    y = y.map({'e': 0, 'p': 1})
    # Convert Dataframe to numpy
    X_enc = X_enc.to_numpy().astype(np.float64)
    y = y.to_numpy()[..., None]
    print(f"\nX_enc.shape: {X_enc.shape}")
    print(f"y.shape: {y.shape}")

    # Split data on train/test
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size = TEST_SIZE)
    # init parts
    loss_fn = BCELoss()
    model = MLP(in_features = X_enc.shape[1], out_features = 1)
    optimizer = SGD(lr = LR)

    # Train loop
    train_stats_by_epochs = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    test_stats_by_epochs = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    for epoch in range(EPOCHS):
        print(f"epoch: {epoch+1}")
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
            y_pred = (y_pred >= 0.5).astype(int)
            train_stats["acc"].append(accuracy(y_pred, train_yb))
            train_stats["prec"].append(precision(y_pred, train_yb))
            train_stats["rec"].append(recall(y_pred, train_yb))
            train_stats["f1"].append(f1(y_pred, train_yb))

        # Add train statistics
        for key, val_lst in train_stats.items():
            if len(val_lst) == 0: continue
            val_lst = np.array(val_lst)
            train_stats_by_epochs[key].append(np.mean(val_lst, axis = 0))

        # Test model each TEST_STEP iteration
        if (epoch + 1) % TEST_STEP == 0:
            test_stats =  {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
            for test_Xb, test_yb in batch_split(X_test, y_test, batch_size = BATCH_SIZE): 
                # Do forward pass
                y_pred = model(test_Xb)
                # Calculate loss
                loss = loss_fn(y_pred, test_yb).mean()
                test_stats["loss"].append(loss)

                # Calculate metrics
                y_pred = (y_pred >= 0.5).astype(int)
                test_stats["acc"].append(accuracy(y_pred, test_yb))
                test_stats["prec"].append(precision(y_pred, test_yb))
                test_stats["rec"].append(recall(y_pred, test_yb))
                test_stats["f1"].append(f1(y_pred, test_yb))
            
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
        "./lab1/results/train_loss.png"
    )
    plot_curves(
        np.arange(len(test_stats_by_epochs["loss"])),
        test_stats_by_epochs["loss"],
        "Test loss plot",
        "epochs",
        "Test loss",
        "./lab1/results/test_loss.png"
    )

    # Calculate mean metrics
    with open("./lab1/results/metrics.txt", "w", encoding = "utf-8") as f:
        f.write("Train metrics:\n")
        for key, val in train_stats_by_epochs.items():
            if key == "loss" or len(val) == 0: continue
            f.write(f"{key}:\n")
            val = np.array(val)
            for cls_idx in range(val.shape[1]):
                f.write(f"{cls_idx}: {val[:, cls_idx].mean()}\n")

        f.write("\nTest metrics:\n")
        for key, val in test_stats_by_epochs.items():
            if key == "loss" or len(val) == 0: continue
            f.write(f"{key}:\n")
            val = np.array(val)
            for cls_idx in range(val.shape[1]):
                f.write(f"{cls_idx}: {val[:, cls_idx].mean()}\n")

    # calculate tpr/fpr for ROC
    y_probs = []
    y_labels = []
    for X_batch, y_batch in batch_split(X_test, y_test, batch_size = BATCH_SIZE):
        probs = model(X_batch)
        y_probs.append(probs)
        y_labels.append(y_batch)

    y_probs = np.concatenate(y_probs).flatten()
    y_labels = np.concatenate(y_labels).flatten()
    tpr, fpr = calculate_tpr_fpr_for_curve(y_labels, y_probs)
    # draw ROC
    plot_curves(
        fpr,
        tpr,
        "ROC",
        "fpr",
        "tpr",
        "./lab1/results/roc.png"
    )