import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pybaseball import statcast
from pybaseball import playerid_lookup
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

df = statcast(start_dt='2025-06-01', end_dt='2025-9-29')

df = df[df['description'] == 'hit_into_play']

cols = [ 
    'batter', 'release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'events', 
    'hit_location', 'bb_type', 
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

df['is_fly_ball'] = (df['bb_type'] == 'fly_ball').astype(int)
df['is_line_drive'] = (df['bb_type'] == 'line_drive').astype(int)
df['is_ground_ball'] = (df['bb_type'] == 'ground_ball').astype(int)

feature_cols = [
    'hc_x', 'hc_y', 'hit_distance_sc', 
    'launch_speed', 'launch_angle', 'effective_speed', 'plate_z',
    'vy0','ay','az', 'pfx_z','release_spin_rate','sz_top','sz_bot',
    'b_hits', 'runner_on_3b', 'is_fly_ball', 'is_line_drive', 'is_ground_ball'
]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

bases_map = {
    'single': 1,
    'double': 2,
    'triple': 3,
    'home_run': 4
}

df['num_bases'] = df['events'].map(bases_map).fillna(0).astype(int)

df_outs = df[df['num_bases'] == 0]
df_singles = df[df['num_bases'] == 1]
df_doubles = df[df['num_bases'] == 2]
df_triples = df[df['num_bases'] == 3]
df_homers = df[df['num_bases'] == 4]

target_size = len(df_singles)

df_outs_resampled = resample(df_outs,
                                 replace=False if len(df_outs) > target_size else True,
                                 n_samples=target_size,
                                 random_state=42)

df_doubles_resampled = resample(df_doubles,
                                 replace=False if len(df_doubles) > target_size else True,
                                 n_samples=target_size,
                                 random_state=42)

df_triples_resampled = resample(df_triples,
                                 replace=True, 
                                 n_samples=target_size,
                                 random_state=42)

df_homers_resampled = resample(df_homers,
                                replace=False if len(df_homers) > target_size else True,
                                n_samples=target_size,
                                random_state=42)

df_balanced = pd.concat([
    df_outs_resampled,
    df_singles,
    df_doubles_resampled,
    df_triples_resampled,
    df_homers_resampled
])

# Shuffle the combined dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

df_balanced = df_balanced.dropna(subset=feature_cols)

x = df_balanced[feature_cols].values.astype(np.float32)
y = df_balanced['num_bases'].astype(int).values

scaler = StandardScaler()
x[:, :14] = scaler.fit_transform(x[:, :14])

# --- Train/test split ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Convert to PyTorch tensors ---
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

class BasesNN(nn.Module):
    def __init__(self, n_input_features, n_classes=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )   
    def forward(self, x):
        return self.layers(x)

model = BasesNN(n_input_features=x_train.shape[1], n_classes=5)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_x, batch_y in train_loader:  # Process in batches
        # Forward pass
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch+1) % 20 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss = {avg_loss:.4f}')

with torch.no_grad():
    y_pred = model(x_test)
    _, predicted = torch.max(y_pred, 1)
    probs = torch.softmax(y_pred, dim=1)

    #expected bases for each swing
    expected_bases = (
        0 * probs[:, 0] +  
        1 * probs[:, 1] +  
        2 * probs[:, 2] + 
        3 * probs[:, 3] +  
        4 * probs[:, 4]  
    )
    
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'\nAccuracy: {accuracy:.4f}')
    
    # Detailed Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), predicted.numpy(),
                                target_names=['Out (0)', 'Single (1)', 'Double (2)', 
                                            'Triple (3)', 'Home Run (4)']))
    
    # Prediction distribution
    print(f"\nPredictions breakdown:")
    for i in range(5):
        count = (predicted == i).sum().item()
        print(f"Predicted {i} bases: {count}")
    
    print(f"\nActual distribution:")
    for i in range(5):
        count = (y_test == i).sum().item()
        print(f"Actual {i} bases: {count}")