
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
import os
import copy

def _rbf_kernel(X, Y, sigma):
    '''Gaussian (RBF) kernel calculation'''
    # Calculate pairwise squared distances
    # X_sq_dist = (X.unsqueeze(1) - X.unsqueeze(0)).pow(2).sum(dim=2)
    # Y_sq_dist = (Y.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(dim=2)
    # XY_sq_dist = (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(dim=2)
    
    X_sq_dist = torch.cdist(X, X, p=2).pow(2)
    Y_sq_dist = torch.cdist(Y, Y, p=2).pow(2)
    XY_sq_dist = torch.cdist(X, Y, p=2).pow(2)

    sigma2 = 2.0 * (sigma ** 2)
    
    k_xx = torch.exp(-X_sq_dist / sigma2)
    k_yy = torch.exp(-Y_sq_dist / sigma2)
    k_xy = torch.exp(-XY_sq_dist / sigma2)
    
    return k_xx, k_yy, k_xy

def compute_mmd(X, Y, sigmas=[1.0, 5.0, 10.0, 20.0]):
    '''Computes the Maximum Mean Discrepancy (MMD) loss using multiple kernels'''
    mmd_loss = 0.0
    for sigma in sigmas:
        k_xx, k_yy, k_xy = _rbf_kernel(X, Y, sigma=sigma)
        
        # Calculate MMD for this kernel
        # We use the mean of the kernel matrices
        mmd_loss_sigma = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        mmd_loss += mmd_loss_sigma
        
    return mmd_loss / len(sigmas) # Average MMD across all kernels


class CMDANTrainer_MMD:
    '''
    Trains a CM-DAN model using MMD loss for domain alignment.
    '''
    def __init__(self, model, device, lambda_mmd=1.0, # Now lambda_mmd
                 weight_decay=1e-3, learning_rate=0.0005, results_dir="."):
        
        self.model = model
        self.device = device
        self.lambda_mmd = lambda_mmd # MMD loss weight
        self.results_dir = results_dir
        
        # Only one loss function needed for the task
        self.task_criterion = nn.BCELoss()
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        
        self.history = {'train_task_loss': [], 'train_mmd_loss': [], 'test_task_acc': []}
        
        print(f"✅ CM-DAN Trainer with MMD Initialized:")
        print(f"    λ_mmd: {lambda_mmd}, LR: {learning_rate}, Weight Decay: {weight_decay}")

    def train_epoch(self, train_loader, epoch, epochs):
        self.model.train()
        avg_task_loss = 0.0
        avg_mmd_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training (MMD)", leave=False)
        for batch_idx, (voice, gait, labels) in enumerate(pbar):
            voice, gait, labels = voice.to(self.device), gait.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # --- 1. Forward pass (no alpha needed) ---
            outputs = self.model(voice, gait)
            
            # --- 2. Calculate Task Loss ---
            task_loss = (self.task_criterion(outputs['voice_task'].squeeze(), labels) +
                         self.task_criterion(outputs['gait_task'].squeeze(), labels)) / 2
            
            # --- 3. Calculate MMD Loss ---
            mmd_loss = compute_mmd(outputs['voice_latent'], outputs['gait_latent'])
            
            # --- 4. Total Loss ---
            loss = task_loss + self.lambda_mmd * mmd_loss
            loss.backward()
            self.optimizer.step()
            
            avg_task_loss += task_loss.item()
            avg_mmd_loss += mmd_loss.item()
            pbar.set_postfix({'Task Loss': f"{task_loss.item():.4f}", 'MMD Loss': f"{mmd_loss.item():.4f}"})
        
        avg_task_loss /= len(train_loader)
        avg_mmd_loss /= len(train_loader)
        self.history['train_task_loss'].append(avg_task_loss)
        self.history['train_mmd_loss'].append(avg_mmd_loss)

    def evaluate(self, test_loader):
        self.model.eval()
        test_correct = 0
        total_samples = 0
        with torch.no_grad():
            for voice, gait, labels in test_loader:
                voice, gait, labels = voice.to(self.device), gait.to(self.device), labels.to(self.device)
                
                # No alpha needed
                outputs = self.model(voice, gait) 
                
                # Ensemble prediction: Average the predictions
                combined_preds = (outputs['voice_task'].squeeze() + outputs['gait_task'].squeeze()) / 2
                
                predicted = (combined_preds > 0.5).float()
                test_correct += (predicted == labels).sum().item()
                total_samples += len(labels)
        
        test_acc = 100. * test_correct / total_samples if total_samples > 0 else 0.0
        self.history['test_task_acc'].append(test_acc)
        return test_acc

    def train(self, train_loader, test_loader, epochs=100, patience=15, min_delta=0.1):
        # (This train loop is identical to the one in CMDANTrainer_Optimized,
        #  including Early Stopping and saving the best model)
        
        print(f"🚀 Starting Optimized MMD training for up to {epochs} epochs...")
        print(f"    Early Stopping: Patience={patience} epochs, Min Delta={min_delta}%")
        
        best_accuracy = 0.0
        epochs_no_improve = 0
        best_model_state = None
        best_epoch = 0
        
        best_model_path = os.path.join(self.results_dir, 'cm_dan_MMD_best_model.pth')

        for epoch in range(epochs):
            self.train_epoch(train_loader, epoch, epochs)
            current_accuracy = self.evaluate(test_loader)
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{epochs} | Test Accuracy: {current_accuracy:.2f}% ", end="")

            if current_accuracy > best_accuracy + min_delta:
                best_accuracy = current_accuracy
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1
                print(f"(New Best! 🎉 Saving model...)")
                torch.save(best_model_state, best_model_path)
            else:
                epochs_no_improve += 1
                print(f"(No improvement for {epochs_no_improve}/{patience} epochs)")
            
            if epochs_no_improve >= patience:
                print(f"\n✋ Early stopping triggered after {epoch+1} epochs.")
                break
        
        if best_model_state:
            print(f"\nLoading best model from Epoch {best_epoch} with Accuracy: {best_accuracy:.2f}%")
            self.model.load_state_dict(best_model_state)
            print(f"Best model state saved to: {best_model_path}")
        else:
             print("\nWarning: No improvement observed during training.")
             best_accuracy = self.history['test_task_acc'][-1] if self.history.get('test_task_acc') else 0.0
             final_model_path = os.path.join(self.results_dir, 'cm_dan_MMD_final_epoch_model.pth')
             torch.save(self.model.state_dict(), final_model_path)
             print(f"Saved final model state to: {final_model_path}")

        print(f"🏆 Training Finished. Best MMD Accuracy: {best_accuracy:.2f}%")
        return self.history, best_accuracy
