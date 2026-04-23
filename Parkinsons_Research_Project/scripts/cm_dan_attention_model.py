
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None

class AttentionLayer(nn.Module):
    '''
    Learns a weight for each input feature.
    This allows the model to "pay attention" to important features
    and "ignore" noisy or irrelevant ones.
    '''
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        # A simple linear layer to learn the weights
        self.attention_weights = nn.Linear(in_features, in_features)

    def forward(self, x):
        # Calculate weights: simple linear layer + softmax
        # Softmax ensures all weights sum to 1, representing a distribution of attention
        weights = F.softmax(self.attention_weights(x), dim=1)

        # Apply weights to the original features
        # The model learns to multiply noisy features by a small weight (e.g., 0.01)
        # and important features by a large weight (e.g., 0.9)
        attended_x = x * weights
        return attended_x

class CMDAN_Attention(nn.Module):
    '''CM-DAN with an AttentionLayer to weigh input features.'''

    def __init__(self, voice_dim, gait_dim, hidden_dim=128, latent_dim=64, dropout_rate=0.6):
        super(CMDAN_Attention, self).__init__()

        dr_hid = max(0.0, dropout_rate)
        dr_hid_deep = max(0.0, dropout_rate - 0.1)
        dr_latent = max(0.0, dropout_rate - 0.2)
        dr_classifier = 0.3

        # --- New Attention Layers ---
        self.voice_attention = AttentionLayer(voice_dim)
        self.gait_attention = AttentionLayer(gait_dim)

        # Encoders are the same as before
        self.voice_encoder = nn.Sequential(
            nn.Linear(voice_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dr_hid),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dr_hid_deep)
        )
        self.gait_encoder = nn.Sequential(
            nn.Linear(gait_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dr_hid),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dr_hid_deep)
        )

        # Shared and classifier parts are the same
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_dim // 2, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(), nn.Dropout(dr_latent)
        )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dr_classifier),
            nn.Linear(32, 2), nn.LogSoftmax(dim=1)
        )
        self.task_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dr_classifier),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        print(f"✅ CM-DAN with ATTENTION Architecture Defined (Dropout: {dropout_rate})")

    def forward(self, voice_data, gait_data, alpha=1.0):
        # --- Apply Attention FIRST ---
        voice_attended = self.voice_attention(voice_data)
        gait_attended = self.gait_attention(gait_data)

        # Pass attended features to encoders
        voice_features = self.voice_encoder(voice_attended)
        gait_features = self.gait_encoder(gait_attended)

        # Rest of the flow is the same
        voice_latent = self.shared_projection(voice_features)
        gait_latent = self.shared_projection(gait_features)
        voice_domain = self.domain_discriminator(GradientReversalLayer.apply(voice_latent, alpha))
        gait_domain = self.domain_discriminator(GradientReversalLayer.apply(gait_latent, alpha))
        voice_task = self.task_classifier(voice_latent)
        gait_task = self.task_classifier(gait_latent)

        return {'voice_task': voice_task, 'gait_task': gait_task, 'voice_domain': voice_domain, 'gait_domain': gait_domain}
