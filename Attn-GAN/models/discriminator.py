import torch
import torch.nn as nn


# ---------------------------
# PatchGAN discriminator 
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]    # (B, C, H, W)  →  (B, C, H/2, W/2)
            if normalize:
                # layers.append(nn.InstanceNorm2d(out_feat, affine=True))  # ✅ Works better with GP and small batches
                layers = [nn.utils.spectral_norm(nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1))]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),    # (B, 64,  H/2, W/2)
            *block(64, 128),                             # (B, 128, H/4, W/4)
            *block(128, 256),                            # (B, 256, H/8, W/8)
            *block(256, 512),                            # (B, 512, H/16, W/16)
            # nn.Conv2d(512, 1, 3, stride=1, padding=1),
            # NEW: Patch-based discriminator
            # nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)),    # (B, 1, H/16 - 1, W/16 - 1)
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        return torch.clamp(out, min=1e-7, max=1 - 1e-7)  # Prevents BCELoss crash
