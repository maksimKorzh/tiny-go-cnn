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

class Detlef44(nn.Module):
  # https://github.com/pasky/pachi/releases/tag/pachi_networks
  def __init__(self, apply_xavier_init: bool = True):
    super().__init__()

    # First conv: in_channels=2, out_channels=128, kernel=7, pad=3
    self.conv1 = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=7, padding=3)
    # six 5x5 conv layers (conv2 .. conv7)
    self.conv2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
    self.conv4 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
    self.conv5 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
    self.conv6 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
    self.conv7 = nn.Conv2d(128, 128, kernel_size=5, padding=2)

    # twelve 3x3 conv layers (conv8 .. conv19)
    self.conv8  = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv9  = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv13 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv14 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv15 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv16 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv17 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv18 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv19 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    # final fully-connected (inner product) layer
    self.fc = nn.Linear(128 * 19 * 19, 361)

    # ReLU reuse
    self.relu = nn.ReLU(inplace=True)

    if apply_xavier_init:
      self._init_weights()

  def _init_weights(self):
    # Xavier (aka Glorot) init for convs and fc biases = 0
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0.0)

  def forward(self, x, apply_softmax: bool = False):
    # x shape: (N, 2, 19, 19)
    x = self.relu(self.conv1(x))   # conv1 -> relu
    x = self.relu(self.conv2(x))   # conv2 -> relu
    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))
    x = self.relu(self.conv5(x))
    x = self.relu(self.conv6(x))
    x = self.relu(self.conv7(x))

    x = self.relu(self.conv8(x))
    x = self.relu(self.conv9(x))
    x = self.relu(self.conv10(x))
    x = self.relu(self.conv11(x))
    x = self.relu(self.conv12(x))
    x = self.relu(self.conv13(x))
    x = self.relu(self.conv14(x))
    x = self.relu(self.conv15(x))
    x = self.relu(self.conv16(x))
    x = self.relu(self.conv17(x))
    x = self.relu(self.conv18(x))
    x = self.relu(self.conv19(x))  # conv19 -> relu (this corresponds to relu20 in prototxt)

    # flatten and linear
    N = x.shape[0]
    x = x.view(N, -1)  # (N, 128*19*19)
    logits = self.fc(x)  # (N, 361)

    if apply_softmax:
      probs = F.softmax(logits, dim=1)
      return probs
    return logits
