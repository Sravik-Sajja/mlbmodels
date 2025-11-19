import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pybaseball import statcast
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample

df = statcast(start_dt='2025-03-26', end_dt='2025-9-29')

df = df[df['description'] == 'hit_into_play']

cols = [ 
    'batter', 'release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'events', 
    'hit_location', #'bb_type', 
    'hc_x', 'hc_y', 'hit_distance_sc', 'launch_speed', 
    'launch_angle', 'effective_speed', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
    'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'pitch_number',
    'pfx_x', 'pfx_z', 'release_spin_rate', 'sz_top', 'sz_bot',
    'stand', 'p_throws', 'release_extension',
    'balls', 'strikes', 'outs_when_up',
    'on_1b', 'on_2b', 'on_3b',
    'inning', 'inning_topbot', 'description'
]
df = df[cols]

df['runner_on_1b'] = df['on_1b'].notna().astype(int)
df['runner_on_2b'] = df['on_2b'].notna().astype(int)
df['runner_on_3b'] = df['on_3b'].notna().astype(int)
df = df.drop(columns=['on_1b', 'on_2b', 'on_3b'])

df['b_hits'] = (df['stand'] == 'L').astype(int)  # 1 if left-handed, 0 if right-handed
df['p_arm'] = (df['p_throws'] == 'L').astype(int)  # 1 if pitcher throws left, 0 if right
df['inning_top'] = (df['inning_topbot'] == 'Top').astype(int) # 1 if top of inning, 0 if bottom
df = df.drop(columns=['stand', 'p_throws', 'inning_topbot'])  # drop old columns


df['platoon_advantage'] = (df['b_hits'] == df['p_arm']).astype(int)


feature_cols = [
    'release_pos_y', 'hc_x', 'hc_y', 'hit_distance_sc', 
    'launch_speed', 'launch_angle', 'effective_speed', 'plate_z',
    'vy0','ay','az', 'pfx_z','release_spin_rate','sz_top','sz_bot','release_extension',
    'b_hits', 'runner_on_1b', 'runner_on_3b'
]


for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['is_hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)

df_majority = df[df['is_hit'] == 0]
df_minority = df[df['is_hit'] == 1]

df_majority_downsampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=int(len(df_minority) * 1),  # match minority class
                                    random_state=42)

# Combine minority class with downsampled majority class
df_balanced = pd.concat([df_majority_downsampled, df_minority])

df_balanced = df_balanced.dropna(subset=feature_cols)

x = df_balanced[feature_cols].values.astype(np.float32)
y = df_balanced['is_hit'].astype(int).values

scaler = StandardScaler()
# first 16 columns are continuous
x[:, :16] = scaler.fit_transform(x[:, :16])

# --- Train/test split ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Convert to PyTorch tensors ---
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class HitNN(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

model = HitNN(n_input_features=x_train.shape[1])

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

num_epochs = 200
for epoch in range(num_epochs):
    y_pred = model(x_train)

    loss = criterion(y_pred, y_train)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 20 == 0:
        print(f'epoch {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(x_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'acc = {acc:.4f}')