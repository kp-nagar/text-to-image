import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, image_channels=3, hidden_dim=64, num_embeddings=128, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 1)
        )
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, image_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z_e = self.encoder(x)
        flat = z_e.permute(0,2,3,1).reshape(-1, z_e.shape[1])
        d = (flat ** 2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.T + (self.codebook.weight ** 2).sum(1)
        idx = d.argmin(1)
        quant = self.codebook(idx).view(x.size(0), z_e.shape[2], z_e.shape[3], -1).permute(0,3,1,2)
        recon = self.decoder(quant)
        loss = F.mse_loss(quant.detach(), z_e) + F.mse_loss(quant, z_e.detach())
        return recon, loss, idx.view(x.size(0), z_e.shape[2], z_e.shape[3])
