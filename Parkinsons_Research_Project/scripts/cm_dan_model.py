
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None

class CrossModalDAN_Regularized(nn.Module):
    def __init__(self, voice_dim, gait_dim, hidden_dim=128, latent_dim=64, dropout_rate=0.6):
        super(CrossModalDAN_Regularized, self).__init__()
        dr_hid = max(0.0, dropout_rate); dr_hid_deep = max(0.0, dropout_rate - 0.1); dr_latent = max(0.0, dropout_rate - 0.2); dr_classifier = 0.3
        self.voice_encoder = nn.Sequential(
            nn.Linear(voice_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dr_hid),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dr_hid_deep)
        )
        self.gait_encoder = nn.Sequential(
            nn.Linear(gait_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dr_hid),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dr_hid_deep)
        )
        self.shared_projection = nn.Sequential(nn.Linear(hidden_dim // 2, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(), nn.Dropout(dr_latent))
        self.domain_discriminator = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dr_classifier), nn.Linear(32, 2), nn.LogSoftmax(dim=1))
        self.task_classifier = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dr_classifier), nn.Linear(32, 1), nn.Sigmoid())
        print(f"✅ Regularized CM-DAN Architecture Defined (Dropout: {dropout_rate})")
    def forward(self, voice_data, gait_data, alpha=1.0):
        voice_features = self.voice_encoder(voice_data); gait_features = self.gait_encoder(gait_data)
        voice_latent = self.shared_projection(voice_features); gait_latent = self.shared_projection(gait_features)
        voice_domain = self.domain_discriminator(GradientReversalLayer.apply(voice_latent, alpha)); gait_domain = self.domain_discriminator(GradientReversalLayer.apply(gait_latent, alpha))
        voice_task = self.task_classifier(voice_latent); gait_task = self.task_classifier(gait_latent)
        return {'voice_task': voice_task, 'gait_task': gait_task, 'voice_domain': voice_domain, 'gait_domain': gait_domain}
