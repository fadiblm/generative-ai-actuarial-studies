import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

LATENT_DIM = 32
COND_DIM   = 2


class SWIMapDataset(Dataset):
    def __init__(self, maps, months):
        self.maps   = torch.from_numpy(maps[:, None, :, :])
        theta       = 2 * np.pi * months / 12
        self.cond   = torch.from_numpy(
            np.stack([np.sin(theta), np.cos(theta)], axis=1).astype(np.float32))
        self.months = torch.from_numpy(months)

    def __len__(self): return len(self.maps)
    def __getitem__(self, i): return self.maps[i], self.cond[i], self.months[i]


class Encoder(nn.Module):
    def __init__(self, H_pad, W_pad, ldim=LATENT_DIM, cdim=COND_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,   32, 3, 2, 1), nn.GroupNorm(8,  32), nn.LeakyReLU(0.2),
            nn.Conv2d(32,  64, 3, 2, 1), nn.GroupNorm(16, 64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(32,128), nn.LeakyReLU(0.2),
        )
        self.hb, self.wb = H_pad // 8, W_pad // 8
        flat = 128 * self.hb * self.wb
        self.fc   = nn.Sequential(nn.Linear(flat + cdim, 512), nn.LeakyReLU(0.2))
        self.mu   = nn.Linear(512, ldim)
        self.logv = nn.Linear(512, ldim)

    def forward(self, x, c):
        h = torch.cat([self.conv(x).flatten(1), c], 1)
        h = self.fc(h)
        return self.mu(h), self.logv(h)


class Decoder(nn.Module):
    def __init__(self, H_pad, W_pad, ldim=LATENT_DIM, cdim=COND_DIM):
        super().__init__()
        self.hb, self.wb = H_pad // 8, W_pad // 8
        flat = 128 * self.hb * self.wb
        self.fc = nn.Sequential(
            nn.Linear(ldim + cdim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, flat),        nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(16, 64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64,  32, 4, 2, 1), nn.GroupNorm(8,  32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32,   1, 4, 2, 1),
        )

    def forward(self, z, c):
        h = self.fc(torch.cat([z, c], 1))
        return self.deconv(h.view(h.size(0), 128, self.hb, self.wb))


class CVAE(nn.Module):
    def __init__(self, H_pad, W_pad, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(H_pad, W_pad, ldim=latent_dim)
        self.decoder = Decoder(H_pad, W_pad, ldim=latent_dim)

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu

    def forward(self, x, c):
        mu, lv = self.encoder(x, c)
        return self.decoder(self.reparameterize(mu, lv), c), mu, lv


# ─────────────────────────────────────────────────────────────────────────────
# Conditional WGAN-GP
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """Generateur conditionnel : z + cond -> carte SWI (H_pad x W_pad)."""

    def __init__(self, H_pad, W_pad, ldim=LATENT_DIM, cdim=COND_DIM):
        super().__init__()
        self.hb, self.wb = H_pad // 8, W_pad // 8
        flat = 128 * self.hb * self.wb
        self.fc = nn.Sequential(
            nn.Linear(ldim + cdim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, flat),        nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(16, 64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32, 4, 2, 1), nn.GroupNorm(8,  32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,   1, 4, 2, 1),
        )

    def forward(self, z, c):
        h = self.fc(torch.cat([z, c], 1))
        return self.deconv(h.view(h.size(0), 128, self.hb, self.wb))


class Critic(nn.Module):
    """Critique WGAN-GP : (carte + cond spatial) -> score reel.
    Pas de normalisation par batch (exigence WGAN-GP).
    La condition est etendue spatialement et concatenee comme canal supplementaire."""

    def __init__(self, H_pad, W_pad, cdim=COND_DIM):
        super().__init__()
        # 1 canal image + cdim canaux condition
        self.conv = nn.Sequential(
            nn.Conv2d(1 + cdim, 32,  4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,       64,  4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,       128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
        )
        self.hb, self.wb = H_pad // 8, W_pad // 8
        self.fc = nn.Linear(128 * self.hb * self.wb, 1)

    def forward(self, x, c):
        # c : (B, cdim) -> (B, cdim, H, W)
        c_map = c[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        return self.fc(self.conv(torch.cat([x, c_map], 1)).flatten(1))


class CGAN(nn.Module):
    """Encapsule generateur + critique pour la sauvegarde et l'inference."""

    def __init__(self, H_pad, W_pad, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator  = Generator(H_pad, W_pad, ldim=latent_dim)
        self.critic     = Critic(H_pad, W_pad)

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        """cond : (B, 2) -> cartes generees (B, 1, H_pad, W_pad)."""
        z = torch.randn(cond.size(0), self.latent_dim, device=cond.device)
        return self.generator(z, cond)
