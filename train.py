import torch
import torch.optim as optim
import numpy as np
from model import *

torch.manual_seed(1337)

epochs = 10
batch_size = 128
learning_rate = 0.001
dataset_path = 'games.pt'
checkpoint_path = 'tiny-go-cnn.pth'
device = torch.device('cpu')

model = TinyGoCNN()
model.to(device)
states, moves = torch.load(dataset_path)
dataset_size = states.size(0)
print(f'Loaded dataset with {dataset_size} samples.')
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
model.train()

for epoch in range(epochs):
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
