import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 19

class TinyGoCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
    self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
    self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
    self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
    self.fc = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = F.relu(self.conv7(x))
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

class CMKGoCNN(nn.Module):
  def __init__(self, channels=96):
    super().__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(2, channels, kernel_size=7, padding=3)
    self.convs5 = nn.ModuleList([
        nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        for _ in range(4)
    ])
    self.convs3 = nn.ModuleList([
        nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        for _ in range(6)
    ])
    self.fc = nn.Linear(channels * 19 * 19, 361)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    for conv in self.convs5: x = self.relu(conv(x))
    for conv in self.convs3: x = self.relu(conv(x))
    x = x.view(x.size(0), -1)
    return self.fc(x)
