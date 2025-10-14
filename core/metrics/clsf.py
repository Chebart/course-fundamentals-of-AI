import numpy as np

def calculate_confusion_matrix_stats(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    num_cls: int
)-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build confusion matrix and get (tp, fp, fn, tn) per class"""
    # Check that labels is integers
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Build confusion matrix
    conf_matrix = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    # Calculate stats per class
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp
    tn = np.sum(conf_matrix) - (tp + fp + fn)

    return tp, fp, fn, tn

def accuracy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    num_cls: int,
    eps: float = 1e-8
)->np.ndarray:
    """(TP + TN) / (TP + TN + FP + FN)"""
    # Calculate pred stats
    (tp, fp, fn, tn) = calculate_confusion_matrix_stats(y_pred, y_true, num_cls)
    # Calculate metric
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return accuracy

def precision(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    num_cls: int,
    eps: float = 1e-8
)->np.ndarray:
    """TP / (TP + FP)"""
    # Calculate pred stats
    (tp, fp, _, _) = calculate_confusion_matrix_stats(y_pred, y_true, num_cls)
    # Calculate metric
    precision = (tp) / (tp + fp + eps)
    return precision

def recall(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    num_cls: int,
    eps: float = 1e-8
)->np.ndarray:
    """TP / (TP + FN)"""
    # Calculate pred stats
    (tp, _, fn, _) = calculate_confusion_matrix_stats(y_pred, y_true, num_cls)
    # Calculate metric
    recall = (tp) / (tp + fn + eps)
    return recall

def f1(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    num_cls: int,
    beta: float = 1,
    eps: float = 1e-8
)->np.ndarray:
    """(beta^2+1) * (presion * recall) / (beta * precision + recall)"""
    # Calculate metric
    prec = precision(y_pred, y_true, num_cls)
    rec = recall(y_pred, y_true, num_cls)
    f1 = (beta**2 + 1) * (prec * rec) / (beta * prec + rec + eps)
    return f1

def calculate_tpr_fpr_for_curve(
    y_true: np.ndarray,     
    y_score: np.ndarray,
)-> tuple[np.ndarray, np.ndarray]:
    """Calculate true and false positives per binary classification threshold"""
    # sort scores in descending order and reorder input data
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # get only consecutive elements where scores differs
    distinct_indices = np.where(np.diff(y_score))[0]
    end = np.array([y_true.size - 1])
    threshold_indices = np.hstack((distinct_indices, end))

    # calculate tps/fps
    tpr = np.cumsum(y_true)[threshold_indices]
    fpr = (1 + threshold_indices) - tpr

    return tpr, fpr


if __name__ == "__main__":
    # Test 1
    y_pred = np.array([1., 2., 0., 3., 0., 0., 1., 1.], dtype=np.float32)
    y_true = np.array([1., 2., 0., 3., 0., 0., 1., 1.], dtype=np.float32)
    print(f"accuracy: {accuracy(y_pred, y_true)}")
    print(f"recall: {recall(y_pred, y_true)}")
    print(f"precision: {precision(y_pred, y_true)}")
    print(f"f1: {f1(y_pred, y_true)}", end="\n\n")
    # Test 2
    y_pred = np.array([1., 0., 0., 1., 1., 0., 1., 0.], dtype=np.float32)
    y_true = np.array([1., 1., 0., 0., 1., 0., 1., 1.], dtype=np.float32)
    print(f"accuracy: {accuracy(y_pred, y_true)}")
    print(f"recall: {recall(y_pred, y_true)}")
    print(f"precision: {precision(y_pred, y_true)}")
    print(f"f1: {f1(y_pred, y_true)}", end="\n\n")
    # Test 3
    y_pred = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    print(f"accuracy: {accuracy(y_pred, y_true)}")
    print(f"recall: {recall(y_pred, y_true)}")
    print(f"precision: {precision(y_pred, y_true)}")
    print(f"f1: {f1(y_pred, y_true)}", end="\n\n")
    # Test 4
    y_score = np.array([0.3, 0.64, 0.48, 0.96, 0.79, 0.13, 0.92, 0.55])
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    print(f"calculate_tps_fps_for_curve: {calculate_tpr_fpr_for_curve(y_true, y_score)}", end="\n\n")