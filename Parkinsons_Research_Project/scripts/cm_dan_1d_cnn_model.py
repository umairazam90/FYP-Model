
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None

class VoiceCNNEncoder(nn.Module):
    '''1D-CNN Encoder for Voice (Input: 1x128)'''
    def __init__(self, out_dim=64):
        super(VoiceCNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3) # (N, 1, 128) -> (N, 16, 128)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2) # (N, 16, 64)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2) # (N, 32, 64)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2) # (N, 32, 32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1) # (N, 64, 32)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2) # (N, 64, 16)
        
        self.flattened_size = 64 * 16
        self.fc = nn.Linear(self.flattened_size, out_dim)
        
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1) # Add channel dim: (N, L) -> (N, 1, L)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc(x))
        return x

class GaitCNNEncoder(nn.Module):
    '''1D-CNN Encoder for Gait (Input: 1x256)'''
    def __init__(self, out_dim=64):
        super(GaitCNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4) # (N, 1, 256) -> (N, 16, 256)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2) # (N, 16, 128)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3) # (N, 32, 128)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2) # (N, 32, 64)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2) # (N, 64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2) # (N, 64, 32)
        
        self.flattened_size = 64 * 32
        self.fc = nn.Linear(self.flattened_size, out_dim)
        
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1) # Add channel dim: (N, L) -> (N, 1, L)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc(x))
        return x

class CMDAN_1D_CNN(nn.Module):
    '''CM-DAN with 1D-CNN encoders for time-series data.'''
    
    def __init__(self, voice_length, gait_length, hidden_dim=128, latent_dim=64, dropout_rate=0.6):
        super(CMDAN_1D_CNN, self).__init__()
        
        dr_latent = max(0.0, dropout_rate - 0.2)
        dr_classifier = 0.3
        
        self.voice_encoder = VoiceCNNEncoder(out_dim=hidden_dim)
        self.gait_encoder = GaitCNNEncoder(out_dim=hidden_dim)
        
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim), 
            nn.BatchNorm1d(latent_dim), nn.ReLU(), nn.Dropout(dr_latent)
        )
        
        self.domain_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dr_classifier),
            nn.Linear(32, 2), nn.LogSoftmax(dim=1)
        )
        self.task_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dr_classifier),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        print(f"✅ CM-DAN with 1D-CNN Architecture Defined (Dropout: {dropout_rate})")

    def forward(self, voice_data, gait_data, alpha=1.0):
        voice_features = self.voice_encoder(voice_data)
        gait_features = self.gait_encoder(gait_data)
        voice_latent = self.shared_projection(voice_features)
        gait_latent = self.shared_projection(gait_features)
        voice_domain = self.domain_discriminator(GradientReversalLayer.apply(voice_latent, alpha))
        gait_domain = self.domain_discriminator(GradientReversalLayer.apply(gait_latent, alpha))
        voice_task = self.task_classifier(voice_latent)
        gait_task = self.task_classifier(gait_latent)
        
        return {'voice_task': voice_task, 'gait_task': gait_task, 'voice_domain': voice_domain, 'gait_domain': gait_domain}
