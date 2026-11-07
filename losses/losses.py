import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        return self.loss(sr, hr)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg16 = models.vgg16(pretrained=True).features    # extracts only the convolutional layers
        self.feature_extractor = nn.Sequential(*list(vgg16.children())[:9]).eval()    # takes only the first 9 layers i.e. low level features like edges, textures, basic structures
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG parameters

    def forward(self, sr, hr):
        # Convert grayscale (1-channel) images to 3-channel images for VGG16
        sr = sr.repeat(1, 3, 1, 1)  # Repeat across 3 channels (B, 1, H, W)  →  (B, 3, H, W)
        hr = hr.repeat(1, 3, 1, 1)  # # Repeat across 3 channels (B, 1, H, W)  →  (B, 3, H, W)
        return F.mse_loss(self.feature_extractor(sr), self.feature_extractor(hr))    # computing mse on the feature map of generated and real image
