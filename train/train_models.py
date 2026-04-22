import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from xgboost import XGBClassifier 

import sys
sys.path.append('../data')
from prepare_data import get_prepared_data, get_weighted_split
from models import HitNN, BasesNN
from evaluate_models import evaluate_nn_hit_model, evaluate_nn_bases_model, evaluate_xgb_hit_model, evaluate_xgb_bases_model, print_comparison

os.makedirs('saved_models', exist_ok=True)

# Hit model (binary)
def train_hit_model(df, features):
    print("\n>>> Training Hit Model (binary: hit vs out)")

    (x_train, x_test, y_train, y_test,
     class_weights, scaler,
     x_train_np, x_test_np, y_train_np, y_test_np) = get_weighted_split(df, features, target_col='is_hit')

    # Neural Network
    model = HitNN(n_input_features=x_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())

    nn_metrics = evaluate_nn_hit_model(model, x_test, y_test)

    torch.save({
        'model_state_dict': model.state_dict(),
        'n_input_features': x_train.shape[1],
        'feature_names': features,
        'scaler': scaler,
        'metrics': nn_metrics,
    }, 'saved_models/hit_model.pth')

    # XGBoost baseline
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train_np == 0).sum() / max((y_train_np == 1).sum(), 1),
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(x_train_np, y_train_np, verbose=False)
    xgb_metrics = evaluate_xgb_hit_model(xgb, x_test_np, y_test_np)

    xgb.save_model('saved_models/hit_xgb.json')

    print_comparison("Hit Prediction", nn_metrics, xgb_metrics)

    return nn_metrics, xgb_metrics


# Bases model (multi-class)

def train_bases_model(df, features):
    print("\n>>> Training Bases Model (multi-class: 0–4 bases)")

    (x_train, x_test, y_train, y_test,
     class_weights, scaler,
     x_train_np, x_test_np, y_train_np, y_test_np) = get_weighted_split(
        df, features, target_col='num_bases'
    )

    # Neural Network
    model = BasesNN(n_input_features=x_train.shape[1], n_classes=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    num_epochs = 140
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss)

    nn_metrics = evaluate_nn_bases_model(model, x_test, y_test)

    torch.save({
        'model_state_dict': model.state_dict(),
        'n_input_features': x_train.shape[1],
        'n_classes': 5,
        'feature_names': features,
        'scaler': scaler,
        'metrics': nn_metrics,
    }, 'saved_models/bases_model.pth')

    # XGBoost
    sample_weights = np.array([class_weights[y].item() for y in y_train_np])

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=5,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(x_train_np, y_train_np, sample_weight=sample_weights, verbose=False)
    xgb_metrics = evaluate_xgb_bases_model(xgb, x_test_np, y_test_np)

    xgb.save_model('saved_models/bases_xgb.json')

    print_comparison("Bases Prediction", nn_metrics, xgb_metrics)

    return nn_metrics, xgb_metrics

def main():
    print("\nFetching and preparing data...")
    df, features = get_prepared_data()
    print(f"Dataset: {len(df):,} balls in play | {len(features)} features\n")

    hit_nn, hit_xgb = train_hit_model(df, features)
    bases_nn, bases_xgb = train_bases_model(df, features)

    print("Training complete. Models saved in saved_models/")


if __name__ == "__main__":
    main()