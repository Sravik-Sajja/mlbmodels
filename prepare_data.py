import pandas as pd
import numpy as np
import torch
from pybaseball import statcast
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def get_prepared_data():
    # Fetch data
    df = statcast(start_dt='2025-03-26', end_dt='2025-09-29')
    df = df[df['description'] == 'hit_into_play'].copy()

    # Add features
    df['runner_on_1b'] = df['on_1b'].notna().astype(int)
    df['runner_on_2b'] = df['on_2b'].notna().astype(int)
    df['runner_on_3b'] = df['on_3b'].notna().astype(int)
    df['b_hits'] = (df['stand'] == 'L').astype(int)
    df['p_arm'] = (df['p_throws'] == 'L').astype(int)
    df['is_fly_ball'] = (df['bb_type'] == 'fly_ball').astype(int)
    df['is_line_drive'] = (df['bb_type'] == 'line_drive').astype(int)
    df['is_ground_ball'] = (df['bb_type'] == 'ground_ball').astype(int)

    # Featured columns
    feature_cols = [
        'hc_x', 'hc_y', 'hit_distance_sc', 'launch_speed', 'launch_angle', 
        'effective_speed', 'b_hits', 'release_speed',
        'runner_on_1b', 'runner_on_2b', 'runner_on_3b', 
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    bases_map = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    df['num_bases'] = df['events'].map(bases_map).fillna(0).astype(int)
    df['is_hit'] = (df['num_bases'] > 0).astype(int)

    df = df.dropna(subset=feature_cols + ['num_bases'])
    return df, feature_cols

def get_balanced_split(df, feature_cols, target_col='num_bases', test_size=0.2):
    # Split
    x = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Calculate class weights
    if target_col == 'num_bases':
        # Calculate inverse frequency weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array([0, 1, 2, 3, 4])
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        # For binary classification
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        class_weights = torch.tensor([pos_weight], dtype=torch.float32)
    
    # Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert to Tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    
    if target_col == 'is_hit':
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    else:
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

    return x_train, x_test, y_train, y_test, class_weights