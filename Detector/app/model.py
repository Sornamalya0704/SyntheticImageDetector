import torch.nn as nn

class SyntheticDetector(nn.Module):
    def __init__(self):
        super(SyntheticDetector, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*16, 256), nn.ReLU(),
            nn.Linear(256, 2)  # binary output
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)