
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# --- Helper classes for the Transformer ---

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# --- GRL (Unchanged) ---
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None

# --- Main FT-Transformer CM-DAN Model ---
class CMDAN_FT_Transformer(nn.Module):
    def __init__(self, num_voice_features, num_gait_features, 
                 embed_dim = 32, # Embedding dimension for each feature
                 depth = 3,      # Number of transformer layers
                 heads = 4,      # Number of attention heads
                 mlp_dim = 64,   # Feedforward hidden dim
                 latent_dim = 64, # Final latent space dim
                 dropout = 0.5): 
        
        super().__init__()
        
        # --- Voice Feature Tokenizer & Transformer ---
        self.voice_embeds = nn.Linear(1, embed_dim) # Embed each single feature value
        self.voice_transformer = Transformer(embed_dim, depth, heads, embed_dim // heads, mlp_dim, dropout)
        # We need a CLS token for voice
        self.voice_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # --- Gait Feature Tokenizer & Transformer ---
        self.gait_embeds = nn.Linear(1, embed_dim) # Embed each single feature value
        self.gait_transformer = Transformer(embed_dim, depth, heads, embed_dim // heads, mlp_dim, dropout)
        # We need a CLS token for gait
        self.gait_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # --- Shared and Classifier Parts ---
        # Project from Transformer output (embed_dim) to final latent space
        self.shared_projection = nn.Sequential(
            nn.Linear(embed_dim, latent_dim), 
            nn.BatchNorm1d(latent_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        
        self.domain_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 2), nn.LogSoftmax(dim=1)
        )
        self.task_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        print(f"✅ CM-DAN with FT-Transformer Architecture Defined")
        print(f"   Feature Embedding Dim: {embed_dim}, Transformer Depth: {depth}, Heads: {heads}")

    def forward(self, voice_data, gait_data, alpha=1.0):
        # --- Voice Branch ---
        # x_voice starts as (b, num_voice_features)
        # 1. Add a dimension: (b, num_voice_features) -> (b, num_voice_features, 1)
        voice_data = voice_data.unsqueeze(-1)
        # 2. Embed each feature: (b, num_voice_features, 1) -> (b, num_voice_features, embed_dim)
        x_voice = self.voice_embeds(voice_data)
        # 3. Prepend CLS token
        b, n, _ = x_voice.shape
        cls_tokens_voice = repeat(self.voice_cls_token, '1 1 d -> b 1 d', b = b)
        x_voice = torch.cat((cls_tokens_voice, x_voice), dim=1) # (b, num_features + 1, embed_dim)
        # 4. Pass through Transformer
        x_voice = self.voice_transformer(x_voice)
        # 5. Get the CLS token output (the representation for the whole modality)
        voice_features = x_voice[:, 0]

        # --- Gait Branch ---
        # x_gait starts as (b, num_gait_features)
        # 1. Add a dimension: (b, num_gait_features) -> (b, num_gait_features, 1)
        gait_data = gait_data.unsqueeze(-1)
        # 2. Embed each feature: (b, num_gait_features, 1) -> (b, num_gait_features, embed_dim)
        x_gait = self.gait_embeds(gait_data)
        # 3. Prepend CLS token
        b, n, _ = x_gait.shape
        cls_tokens_gait = repeat(self.gait_cls_token, '1 1 d -> b 1 d', b = b)
        x_gait = torch.cat((cls_tokens_gait, x_gait), dim=1) # (b, num_features + 1, embed_dim)
        # 4. Pass through Transformer
        x_gait = self.gait_transformer(x_gait)
        # 5. Get the CLS token output
        gait_features = x_gait[:, 0]
        
        # --- Rest of the flow is the same ---
        voice_latent = self.shared_projection(voice_features)
        gait_latent = self.shared_projection(gait_features)
        
        voice_domain = self.domain_discriminator(GradientReversalLayer.apply(voice_latent, alpha))
        gait_domain = self.domain_discriminator(GradientReversalLayer.apply(gait_latent, alpha))
        
        voice_task = self.task_classifier(voice_latent)
        gait_task = self.task_classifier(gait_latent)
        
        return {'voice_task': voice_task, 'gait_task': gait_task, 'voice_domain': voice_domain, 'gait_domain': gait_domain}
