import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from prepare_data import get_prepared_data, get_balanced_split
from models import HitNN, BasesNN
import os

# Create directory for saved models
os.makedirs('saved_models', exist_ok=True)

def train_hit_model(df, features):
    print("Training Hit Model")
    
    x_train, x_test, y_train, y_test, class_weights = get_balanced_split(df, features, target_col='is_hit')
    
    # Initialize model
    model = HitNN(n_input_features=x_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred_prob = torch.sigmoid(y_pred)
        y_pred_cls = y_pred_prob.round()
        acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_input_features': x_train.shape[1],
        'feature_names': features
    }, 'saved_models/hit_model.pth')
    
    print(f"Hit Model trained - Accuracy: {acc:.4f}")
    return acc.item()

def train_bases_model(df, features):
    print("Training Bases Model")
    
    x_train, x_test, y_train, y_test, class_weights = get_balanced_split(df, features, target_col='num_bases')
    
    # Initialize model
    model = BasesNN(n_input_features=x_train.shape[1], n_classes=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training with DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    num_epochs = 140
    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        _, predicted = torch.max(y_pred, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_input_features': x_train.shape[1],
        'n_classes': 5,
        'feature_names': features
    }, 'saved_models/bases_model.pth')
    
    print(f"Bases Model trained - Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    print("\nTraining models...\n")
    
    df, features = get_prepared_data()
    # Train both models
    hit_acc = train_hit_model(df, features)
    bases_acc = train_bases_model(df, features)
    
    print(f"\nTraining complete: ")
    print(f"   Models saved in 'saved_models/' directory\n")
    

if __name__ == "__main__":
    main()
    