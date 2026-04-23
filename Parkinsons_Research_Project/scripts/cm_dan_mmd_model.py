
import torch
import torch.nn as nn

class CMDAN_MMD(nn.Module):
    '''
    Cross-Modal Network using MMD loss (no discriminator).
    It uses the same encoders and classifier as the regularized CM-DAN.
    '''
    def __init__(self, voice_dim, gait_dim, hidden_dim=128, latent_dim=64, dropout_rate=0.6):
        super(CMDAN_MMD, self).__init__()
        
        dr_hid = max(0.0, dropout_rate)
        dr_hid_deep = max(0.0, dropout_rate - 0.1)
        dr_latent = max(0.0, dropout_rate - 0.2)
        dr_classifier = 0.3 # Keep classifier dropout moderate

        # --- Encoders (Same as before) ---
        self.voice_encoder = nn.Sequential(
            nn.Linear(voice_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dr_hid),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dr_hid_deep)
        )
        self.gait_encoder = nn.Sequential(
            nn.Linear(gait_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dr_hid),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dr_hid_deep)
        )
        
        # --- Shared Projection (Same as before) ---
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_dim // 2, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(), nn.Dropout(dr_latent)
        )
        
        # --- Task Classifier (Same as before) ---
        self.task_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dr_classifier),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        
        # --- NO DISCRIMINATOR OR GRL ---
        
        print(f"✅ CM-DAN with MMD Loss Architecture Defined (Dropout: {dropout_rate})")

    def forward(self, voice_data, gait_data):
        # Pass features to encoders
        voice_features = self.voice_encoder(voice_data)
        gait_features = self.gait_encoder(gait_data)
        
        # Project to shared latent space
        voice_latent = self.shared_projection(voice_features)
        gait_latent = self.shared_projection(gait_features)
        
        # Get task predictions
        voice_task = self.task_classifier(voice_latent)
        gait_task = self.task_classifier(gait_latent)
        
        # Return latents (for MMD loss) and task predictions (for task loss)
        return {
            'voice_task': voice_task,
            'gait_task': gait_task,
            'voice_latent': voice_latent,
            'gait_latent': gait_latent
        }
