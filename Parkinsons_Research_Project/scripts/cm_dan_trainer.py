
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
import os
import copy

class CMDANTrainer_Optimized:
    def __init__(self, model, device, lambda_domain=0.7, weight_decay=1e-3, learning_rate=0.0005, results_dir="."):
        self.model = model; self.device = device; self.lambda_domain = lambda_domain; self.results_dir = results_dir
        self.task_criterion = nn.BCELoss(); self.domain_criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.history = {'train_task_loss': [], 'train_domain_loss': [], 'test_task_acc': []}
        print(f"✅ Optimized CM-DAN Trainer Initialized:")
        print(f"    λ_domain: {lambda_domain}, LR: {learning_rate}, Weight Decay: {weight_decay}")

    def train_epoch(self, train_loader, epoch, epochs):
        self.model.train(); avg_task_loss = 0.0; avg_domain_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        for batch_idx, (voice, gait, labels) in enumerate(pbar):
            voice, gait, labels = voice.to(self.device), gait.to(self.device), labels.to(self.device)
            p = float(batch_idx + epoch * len(train_loader)) / (epochs * len(train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            self.optimizer.zero_grad()
            outputs = self.model(voice, gait, alpha)
            task_loss = (self.task_criterion(outputs['voice_task'].squeeze(), labels) + self.task_criterion(outputs['gait_task'].squeeze(), labels)) / 2
            domain_labels_voice = torch.zeros(len(voice)).long().to(self.device)
            domain_labels_gait = torch.ones(len(gait)).long().to(self.device)
            domain_loss = (self.domain_criterion(outputs['voice_domain'], domain_labels_voice) + self.domain_criterion(outputs['gait_domain'], domain_labels_gait)) / 2
            loss = task_loss + self.lambda_domain * domain_loss
            loss.backward(); self.optimizer.step()
            avg_task_loss += task_loss.item(); avg_domain_loss += domain_loss.item()
            pbar.set_postfix({'Task Loss': f"{task_loss.item():.4f}", 'Domain Loss': f"{domain_loss.item():.4f}"})
        avg_task_loss /= (len(train_loader) if len(train_loader) > 0 else 1)
        avg_domain_loss /= (len(train_loader) if len(train_loader) > 0 else 1)
        self.history['train_task_loss'].append(avg_task_loss); self.history['train_domain_loss'].append(avg_domain_loss)

    def evaluate(self, test_loader):
        self.model.eval(); test_correct = 0; total_samples = 0
        with torch.no_grad():
            for voice, gait, labels in test_loader:
                voice, gait, labels = voice.to(self.device), gait.to(self.device), labels.to(self.device)
                outputs = self.model(voice, gait, alpha=0)
                combined_preds = (outputs['voice_task'].squeeze() + outputs['gait_task'].squeeze()) / 2
                predicted = (combined_preds > 0.5).float()
                test_correct += (predicted == labels).sum().item(); total_samples += len(labels)
        test_acc = 100. * test_correct / total_samples if total_samples > 0 else 0.0
        self.history['test_task_acc'].append(test_acc); return test_acc

    def train(self, train_loader, test_loader, epochs=100, patience=15, min_delta=0.1):
        print(f"🚀 Starting Optimized CM-DAN training for up to {epochs} epochs...")
        print(f"    Early Stopping: Patience={patience} epochs, Min Delta={min_delta}%")
        best_accuracy = 0.0; epochs_no_improve = 0; best_model_state = None; best_epoch = 0
        best_model_path = os.path.join(self.results_dir, 'cm_dan_best_model_earlystop.pth')
        for epoch in range(epochs):
            self.train_epoch(train_loader, epoch, epochs)
            current_accuracy = self.evaluate(test_loader)
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} | Test Accuracy: {current_accuracy:.2f}% ", end="")
            if current_accuracy > best_accuracy + min_delta:
                best_accuracy = current_accuracy; epochs_no_improve = 0
                best_model_state = copy.deepcopy(self.model.state_dict()); best_epoch = epoch + 1
                print(f"(New Best! 🎉 Saving model...)")
                torch.save(best_model_state, best_model_path)
            else:
                epochs_no_improve += 1
                print(f"(No improvement for {epochs_no_improve}/{patience} epochs)")
            if epochs_no_improve >= patience:
                print(f"\n✋ Early stopping triggered after {epoch+1} epochs."); break
        if best_model_state:
            print(f"\nLoading best model from Epoch {best_epoch} with Accuracy: {best_accuracy:.2f}%")
            self.model.load_state_dict(best_model_state); print(f"Best model state saved to: {best_model_path}")
        else:
             print("\nWarning: No improvement observed during training.")
             best_accuracy = self.history['test_task_acc'][-1] if self.history.get('test_task_acc') else 0.0
        print(f"🏆 Training Finished. Best Validated Test Accuracy: {best_accuracy:.2f}%")
        return self.history, best_accuracy
