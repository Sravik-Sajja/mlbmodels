from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, average_precision_score
)
import torch
import numpy as np

# Evaluation helpers

def evaluate_nn_hit_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        probs = torch.sigmoid(logits).squeeze().numpy()
        preds = (probs >= 0.5).astype(int)

    y = y_test.squeeze().numpy().astype(int)

    f1 = f1_score(y, preds)
    roc_auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds, target_names=['Out', 'Hit'])

    return {
        'f1': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }


def evaluate_nn_bases_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    y = y_test.numpy()

    f1_macro = f1_score(y, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(y, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y, preds, labels=[0, 1, 2, 3, 4])
    report = classification_report(
        y, preds,
        labels=[0, 1, 2, 3, 4],
        target_names=['Out', 'Single', 'Double', 'Triple', 'HR'],
        zero_division=0
    )

    return {
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }

def evaluate_xgb_hit_model(model, x_test, y_test):
    probs = model.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    report = classification_report(y_test, preds, target_names=['Out', 'Hit'])
    return {
        'f1': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'classification_report': report,
    }


def evaluate_xgb_bases_model(model, x_test, y_test):
    preds = model.predict(x_test)
    f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, preds, average='weighted', zero_division=0)
    report = classification_report(
        y_test, preds,
        labels=[0, 1, 2, 3, 4],
        target_names=['Out', 'Single', 'Double', 'Triple', 'HR'],
        zero_division=0
    )
    return {
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'classification_report': report,
    }


def print_comparison(task, nn_metrics, xgb_metrics):
    print(f"  {task} — NN vs XGBoost")
    all_keys = sorted(set(nn_metrics) | set(xgb_metrics))
    numeric_keys = [k for k in all_keys if isinstance(nn_metrics.get(k), (int, float))]
    header = f"  {'Metric':<20} {'Neural Net':>12} {'XGBoost':>12}"
    print(header)
    for k in numeric_keys:
        nn_val = nn_metrics.get(k, '-')
        xgb_val = xgb_metrics.get(k, '-')
        print(f"  {k:<20} {nn_val:>12} {xgb_val:>12}")
    print("Classification report (Neural Net):")
    print(nn_metrics['classification_report'])
    print("Classification report (XGB):")
    print(xgb_metrics['classification_report'])