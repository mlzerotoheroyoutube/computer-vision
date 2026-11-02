"""Convolutional Neural Networks: code from the ML Zero to Hero explainer.

Video: https://youtu.be/2gSITaqftKY
"""

# A small CNN for MNIST-sized inputs, with shape tracing
import torch
import torch.nn as nn

torch.manual_seed(0)

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),   # 32×28×28
            nn.MaxPool2d(2),                                                             # 32×14×14
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),  # 64×14×14
            nn.MaxPool2d(2),                                                             # 64×7×7
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),# 128×7×7
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))

model = SmallCNN()
x = torch.randn(8, 1, 28, 28)
h = x
for layer in model.features:
    h = layer(h)
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        print(type(layer).__name__, tuple(h.shape))
print("logits", tuple(model(x).shape), "params", sum(p.numel() for p in model.parameters()))

# One training step
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
y = torch.randint(0, 10, (8,))
loss = nn.functional.cross_entropy(model(x), y)
loss.backward(); opt.step()
print("loss", round(loss.item(), 3))

# --------------------------------------------------------------------------

# Using a pretrained modern CNN (ConvNeXt-Tiny) from torchvision as a feature extractor
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

weights = ConvNeXt_Tiny_Weights.DEFAULT          # downloads ImageNet weights on first use
model = convnext_tiny(weights=weights).eval()
preprocess = weights.transforms()
img = torch.rand(3, 256, 256)                     # stand-in for a real image tensor in [0, 1]
with torch.no_grad():
    feats = model.features(preprocess(img)[None])  # (1, 768, 7, 7)
    logits = model(preprocess(img)[None])          # (1, 1000)
print(feats.shape, logits.shape)
