import pandas as pd
import numpy as np
import torch
from pybaseball import statcast
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def get_prepared_data():
    # Fetch data
    df = statcast(start_dt='2025-03-26', end_dt='2025-09-29')
    df = df[df['description'] == 'hit_into_play'].copy()

    # Game state features
    df['runner_on_1b'] = df['on_1b'].notna().astype(int)
    df['runner_on_2b'] = df['on_2b'].notna().astype(int)
    df['runner_on_3b'] = df['on_3b'].notna().astype(int)
    df['b_hits_left'] = (df['stand'] == 'L').astype(int)
    df['p_throws_left'] = (df['p_throws'] == 'L').astype(int)
    df['same_hand'] = (df['stand'] == df['p_throws']).astype(int)

    df['if_fielding_alignment'] = df['if_fielding_alignment'].fillna('Standard')
    df['of_fielding_alignment'] = df['of_fielding_alignment'].fillna('Standard')
    if_dummies = pd.get_dummies(df['if_fielding_alignment'], prefix='if_align')
    of_dummies = pd.get_dummies(df['of_fielding_alignment'], prefix='of_align')
    df = pd.concat([df, if_dummies, of_dummies], axis=1)

    # Featured columns
    feature_cols = [
        'hc_x', 
        'hc_y', 
        'hit_distance_sc', 
        'launch_speed', 
        'launch_angle',
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    bases_map = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    df['num_bases'] = df['events'].map(bases_map).fillna(0).astype(int)
    df['is_hit'] = (df['num_bases'] > 0).astype(int)

    df = df.dropna(subset=feature_cols + ['num_bases'])
    return df, feature_cols


def get_weighted_split(df, feature_cols, target_col='num_bases', test_size=0.2):
    #Split data, compute class weights, scale features, and return tensors
    x = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=y
    )

    # Class weights for imbalanced targets
    if target_col == 'num_bases':
        classes = np.array([0, 1, 2, 3, 4])
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weights = np.cbrt(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        class_weights = torch.tensor([pos_weight], dtype=torch.float32)

    # Scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Convert to Tensors
    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)

    if target_col == 'is_hit':
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

    return x_train_t, x_test_t, y_train_t, y_test_t, class_weights, scaler, x_train, x_test, y_train, y_test