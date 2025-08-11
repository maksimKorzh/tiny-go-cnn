import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import *

torch.manual_seed(1337)

epochs = 10
batch_size = 128
learning_rate = 0.001
dataset_path = 'games.pt'
checkpoint_path = 'tiny-go-cnn.pth'
device = torch.device('cpu')

model = Detlef44()  # Or TinyGoCNN()
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

start_epoch = 0

# Load dataset
states, moves = torch.load(dataset_path)
states = states.float()
moves = moves.long()
dataset_size = states.size(0)
print(f'Loaded dataset with {dataset_size} samples.')

# Check if checkpoint exists and load
try:
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_epoch = checkpoint['epoch']
  print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
except FileNotFoundError:
  print("No checkpoint found, starting training from scratch.")

model.train()

for epoch in range(start_epoch, epochs):
  permutation = torch.randperm(dataset_size)
  epoch_loss = 0
  for i in range(0, dataset_size, batch_size):
    indices = permutation[i:i+batch_size]
    batch_states = states[indices].to(device)
    batch_moves = moves[indices].to(device)

    optimizer.zero_grad()
    outputs = model(batch_states)
    loss = criterion(outputs, batch_moves)
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    print(f'Iter {i}/{dataset_size}, loss {loss.item():.4f}')

  avg_loss = epoch_loss / (dataset_size / batch_size)
  print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

  torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
  }, checkpoint_path)
  print(f"Checkpoint saved to {checkpoint_path}")
