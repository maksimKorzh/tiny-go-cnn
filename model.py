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
