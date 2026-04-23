
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import pandas as pd
import torch.optim as optim # <-- THIS IS THE FIX


class Baseline_1D_CNN(nn.Module):
    '''A simple 1D-CNN to act as a baseline for time-series data.'''
    def __init__(self, input_dim, num_classes=1):
        super(Baseline_1D_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the flattened size
        self.flattened_size = 32 * (int(input_dim / 4))
        
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        x = x.view(-1, self.flattened_size)
        
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

class BaselineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def evaluate_baseline_cnn_subject_wise(X, y, pair_info_path, input_dim, device):
    '''Evaluates the simple 1D-CNN using subject-wise CV.'''
    print(f"--- Evaluating Baseline 1D-CNN (Input Dim: {input_dim}) on {device} ---")
    
    try:
        pair_info = pd.read_csv(pair_info_path)
        subject_ids = pair_info['voice_subject'].values
    except Exception as e:
        print(f"Error loading pair_info: {e}")
        return 0.0
        
    group_kfold = GroupKFold(n_splits=5)
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, subject_ids)):
        print(f"  Fold {fold+1}/5...")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        train_dataset = BaselineDataset(X_train, y_train)
        test_dataset = BaselineDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        model = Baseline_1D_CNN(input_dim=input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # This line will now work
        criterion = nn.BCELoss()
        
        for epoch in range(20): # 20 epochs for baseline
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                all_preds.extend((preds > 0.5).float().cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"    Fold {fold+1} Accuracy: {acc:.4f}")
        fold_accuracies.append(acc)
        
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"  Baseline 1D-CNN Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return mean_acc
