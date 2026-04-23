
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
import os
import copy
import torch.nn.functional as F

class CoTeaching_CMDANTrainer:
    '''
    Trains two CM-DAN models simultaneously, using co-teaching
    to filter out noisy (mismatched) labels.
    '''
    def __init__(self, model_A, model_B, device, lambda_domain=0.7,
                 weight_decay=1e-3, learning_rate=0.0005, results_dir=".",
                 keep_rate=0.8): # Keep 80% (for 20% noise)

        self.model_A = model_A
        self.model_B = model_B
        self.device = device
        self.lambda_domain = lambda_domain
        self.results_dir = results_dir
        self.keep_rate = keep_rate

        # Loss functions
        # CRITICAL: Use reduction='none' to get per-sample losses for filtering
        self.task_criterion_per_sample = nn.BCELoss(reduction='none')
        # Use reduction='mean' for the final filtered batch update
        self.task_criterion_mean = nn.BCELoss(reduction='mean')
        self.domain_criterion = nn.NLLLoss()

        # Optimizers for each model
        self.optimizer_A = optim.Adam(model_A.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer_B = optim.Adam(model_B.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Schedulers for each optimizer
        self.scheduler_A = optim.lr_scheduler.StepLR(self.optimizer_A, step_size=30, gamma=0.5)
        self.scheduler_B = optim.lr_scheduler.StepLR(self.optimizer_B, step_size=30, gamma=0.5)

        self.history = {'train_task_loss': [], 'train_domain_loss': [], 'test_task_acc': []}

        print(f"✅ Co-Teaching CM-DAN Trainer Initialized:")
        print(f"    Keep Rate: {keep_rate*100:.0f}% (for {100-keep_rate*100:.0f}% assumed noise)")
        print(f"    λ_domain: {lambda_domain}, LR: {learning_rate}, Weight Decay: {weight_decay}")

    def train_epoch(self, train_loader, epoch, epochs):
        self.model_A.train()
        self.model_B.train()

        avg_task_loss = 0.0
        avg_domain_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Co-Teaching", leave=False)
        for batch_idx, (voice, gait, labels) in enumerate(pbar):
            voice, gait, labels = voice.to(self.device), gait.to(self.device), labels.to(self.device)

            p = float(batch_idx + epoch * len(train_loader)) / (epochs * len(train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # --- 1. Forward pass for BOTH models ---
            outputs_A = self.model_A(voice, gait, alpha)
            outputs_B = self.model_B(voice, gait, alpha)

            # --- 2. Calculate per-sample task losses ---
            # (voice_loss + gait_loss) / 2 for each sample
            loss_A_task_per_sample = (self.task_criterion_per_sample(outputs_A['voice_task'].squeeze(), labels) +
                                      self.task_criterion_per_sample(outputs_A['gait_task'].squeeze(), labels)) / 2
            loss_B_task_per_sample = (self.task_criterion_per_sample(outputs_B['voice_task'].squeeze(), labels) +
                                      self.task_criterion_per_sample(outputs_B['gait_task'].squeeze(), labels)) / 2

            # --- 3. Identify "clean" samples (lowest loss) ---
            num_keep = int(len(labels) * self.keep_rate)
            _, keep_indices_A = torch.topk(loss_A_task_per_sample, num_keep, largest=False)
            _, keep_indices_B = torch.topk(loss_B_task_per_sample, num_keep, largest=False)

            # --- 4. Cross-Update Model A (using Model B's clean set) ---
            self.optimizer_A.zero_grad()

            # Get outputs from Model A, but only for samples Model B found clean
            outputs_A_clean = self.model_A(voice[keep_indices_B], gait[keep_indices_B], alpha)
            labels_clean_B = labels[keep_indices_B] # Get the corresponding clean labels

            # Task Loss (on mean) for clean samples
            task_loss_A = (self.task_criterion_mean(outputs_A_clean['voice_task'].squeeze(), labels_clean_B) +
                           self.task_criterion_mean(outputs_A_clean['gait_task'].squeeze(), labels_clean_B)) / 2

            # Domain Loss for clean samples
            domain_labels_voice_A = torch.zeros(len(labels_clean_B)).long().to(self.device)
            domain_labels_gait_A = torch.ones(len(labels_clean_B)).long().to(self.device)
            domain_loss_A = (self.domain_criterion(outputs_A_clean['voice_domain'], domain_labels_voice_A) +
                             self.domain_criterion(outputs_A_clean['gait_domain'], domain_labels_gait_A)) / 2

            loss_A_total = task_loss_A + self.lambda_domain * domain_loss_A
            loss_A_total.backward()
            self.optimizer_A.step()

            # --- 5. Cross-Update Model B (using Model A's clean set) ---
            self.optimizer_B.zero_grad()

            # Get outputs from Model B, but only for samples Model A found clean
            outputs_B_clean = self.model_B(voice[keep_indices_A], gait[keep_indices_A], alpha)
            labels_clean_A = labels[keep_indices_A] # Get the corresponding clean labels

            # Task Loss (on mean) for clean samples
            task_loss_B = (self.task_criterion_mean(outputs_B_clean['voice_task'].squeeze(), labels_clean_A) +
                           self.task_criterion_mean(outputs_B_clean['gait_task'].squeeze(), labels_clean_A)) / 2

            # Domain Loss for clean samples
            domain_labels_voice_B = torch.zeros(len(labels_clean_A)).long().to(self.device)
            domain_labels_gait_B = torch.ones(len(labels_clean_A)).long().to(self.device)
            domain_loss_B = (self.domain_criterion(outputs_B_clean['voice_domain'], domain_labels_voice_B) +
                             self.domain_criterion(outputs_B_clean['gait_domain'], domain_labels_gait_B)) / 2

            loss_B_total = task_loss_B + self.lambda_domain * domain_loss_B
            loss_B_total.backward()
            self.optimizer_B.step()

            # --- 6. Record losses ---
            avg_task_loss += (task_loss_A.item() + task_loss_B.item()) / 2
            avg_domain_loss += (domain_loss_A.item() + domain_loss_B.item()) / 2

        avg_task_loss /= len(train_loader)
        avg_domain_loss /= len(train_loader)
        self.history['train_task_loss'].append(avg_task_loss)
        self.history['train_domain_loss'].append(avg_domain_loss)

    def evaluate(self, test_loader):
        self.model_A.eval()
        self.model_B.eval()
        test_correct = 0
        total_samples = 0
        with torch.no_grad():
            for voice, gait, labels in test_loader:
                voice, gait, labels = voice.to(self.device), gait.to(self.device), labels.to(self.device)

                # Get outputs from both models
                outputs_A = self.model_A(voice, gait, alpha=0)
                outputs_B = self.model_B(voice, gait, alpha=0)

                # Ensemble prediction: Average the predictions of both models
                pred_A_avg = (outputs_A['voice_task'].squeeze() + outputs_A['gait_task'].squeeze()) / 2
                pred_B_avg = (outputs_B['voice_task'].squeeze() + outputs_B['gait_task'].squeeze()) / 2

                combined_preds_avg = (pred_A_avg + pred_B_avg) / 2

                predicted = (combined_preds_avg > 0.5).float()
                test_correct += (predicted == labels).sum().item()
                total_samples += len(labels)

        test_acc = 100. * test_correct / total_samples if total_samples > 0 else 0.0
        self.history['test_task_acc'].append(test_acc)
        return test_acc

    def train(self, train_loader, test_loader, epochs=100, patience=15, min_delta=0.1):
        print(f"🚀 Starting Co-Teaching CM-DAN training for up to {epochs} epochs...")
        print(f"    Early Stopping: Patience={patience} epochs, Min Delta={min_delta}%")

        best_accuracy = 0.0
        epochs_no_improve = 0
        best_model_A_state = None
        best_model_B_state = None
        best_epoch = 0

        best_model_A_path = os.path.join(self.results_dir, 'co_teaching_model_A_best.pth')
        best_model_B_path = os.path.join(self.results_dir, 'co_teaching_model_B_best.pth')

        for epoch in range(epochs):
            self.train_epoch(train_loader, epoch, epochs)
            current_accuracy = self.evaluate(test_loader)

            # Step schedulers
            self.scheduler_A.step()
            self.scheduler_B.step()

            print(f"Epoch {epoch+1}/{epochs} | Test Accuracy: {current_accuracy:.2f}% ", end="")

            if current_accuracy > best_accuracy + min_delta:
                best_accuracy = current_accuracy
                epochs_no_improve = 0
                # Save BOTH model states
                best_model_A_state = copy.deepcopy(self.model_A.state_dict())
                best_model_B_state = copy.deepcopy(self.model_B.state_dict())
                best_epoch = epoch + 1
                print(f"(New Best! 🎉 Saving models...)")
                torch.save(best_model_A_state, best_model_A_path)
                torch.save(best_model_B_state, best_model_B_path)
            else:
                epochs_no_improve += 1
                print(f"(No improvement for {epochs_no_improve}/{patience} epochs)")

            if epochs_no_improve >= patience:
                print(f"\n✋ Early stopping triggered after {epoch+1} epochs.")
                break

        if best_model_A_state:
            print(f"\nLoading best models from Epoch {best_epoch} with Accuracy: {best_accuracy:.2f}%")
            self.model_A.load_state_dict(best_model_A_state)
            self.model_B.load_state_dict(best_model_B_state)
            print(f"Best model states saved to: {self.results_dir}")
        else:
             print("\nWarning: No improvement observed during training.")
             best_accuracy = self.history['test_task_acc'][-1] if self.history.get('test_task_acc') else 0.0

        print(f"🏆 Training Finished. Best Co-Teaching Accuracy: {best_accuracy:.2f}%")
        return self.history, best_accuracy
