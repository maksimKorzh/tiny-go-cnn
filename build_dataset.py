import os
import torch
import numpy as np
from sgfmill import sgf, boards
import glob

BOARD_SIZE = 19
base_dir = 'games'
out_file = 'games.pt'
tmp_dir = 'tmp_batches'
start_year = 2000
end_year = 2017
max_games = 20000
BATCH_SIZE = 5000  # moves per batch

os.makedirs(tmp_dir, exist_ok=True)

def board_to_tensor(board):
  black_plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
  white_plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
  for row in range(BOARD_SIZE):
    for col in range(BOARD_SIZE):
      stone = board.get(row, col)
      if stone == 'b':
        black_plane[row, col] = 1
      elif stone == 'w':
        white_plane[row, col] = 1
  return np.stack([black_plane, white_plane])

def move_to_index(move):
  return move[0] * BOARD_SIZE + move[1]

def process_sgf_file(sgf_path):
  with open(sgf_path, "rb") as f:
    sgf_game = sgf.Sgf_game.from_bytes(f.read())
  board = boards.Board(BOARD_SIZE)
  moves = []
  nodes = list(sgf_game.get_main_sequence())
  for node in nodes[1:]:
    color, move = node.get_move()
    if move is None:
      continue
    state_tensor = board_to_tensor(board)
    move_idx = move_to_index(move)
    moves.append((state_tensor, move_idx))
    board.play(move[0], move[1], color)
  return moves

states_batch = []
moves_batch = []
batch_idx = 0

def save_batch():
  global batch_idx, states_batch, moves_batch
  if not states_batch:
    return
  states_tensor = torch.tensor(np.array(states_batch), dtype=torch.uint8)
  moves_tensor = torch.tensor(np.array(moves_batch), dtype=torch.int16)
  batch_path = os.path.join(tmp_dir, f"batch_{batch_idx}.pt")
  torch.save((states_tensor, moves_tensor), batch_path)
  print(f"Saved batch {batch_idx} with {len(states_batch)} samples")
  states_batch.clear()
  moves_batch.clear()
  batch_idx += 1

years = [str(y) for y in range(start_year, end_year + 1)]
games_processed = 0

for year in years:
  year_path = os.path.join(base_dir, year)
  if not os.path.isdir(year_path):
    print(f"Warning: Year folder not found: {year_path}")
    continue

  for subfolder in sorted(os.listdir(year_path)):
    subfolder_path = os.path.join(year_path, subfolder)
    if not os.path.isdir(subfolder_path):
      continue

    for file in sorted(os.listdir(subfolder_path)):
      if not file.endswith('.sgf'):
        continue
      sgf_path = os.path.join(subfolder_path, file)
      try:
        samples = process_sgf_file(sgf_path)
        for s in samples:
          states_batch.append(s[0])
          moves_batch.append(s[1])
          if len(states_batch) >= BATCH_SIZE:
            save_batch()

        games_processed += 1
        if games_processed % 100 == 0:
          print(f"Processed {games_processed} games...")

        if max_games is not None and games_processed >= max_games:
          break
      except Exception as e:
        print(f"Error processing {sgf_path}: {e}")

    if max_games is not None and games_processed >= max_games:
      break
  if max_games is not None and games_processed >= max_games:
    break

save_batch()

print(f"Total games processed: {games_processed}")
print("Merging all batches into a single file...")

batch_files = sorted(glob.glob(os.path.join(tmp_dir, "*.pt")))

total_samples = 0
for bf in batch_files:
  states, moves = torch.load(bf)
  total_samples += states.shape[0]

print(f"Total samples to merge: {total_samples}")

states_memmap = np.memmap('states_memmap.npy', mode='w+', dtype=np.uint8,
  shape=(total_samples, 2, BOARD_SIZE, BOARD_SIZE))
moves_memmap = np.memmap('moves_memmap.npy', mode='w+', dtype=np.int16,
  shape=(total_samples,))

idx = 0
for bf in batch_files:
  states, moves = torch.load(bf)
  batch_size = states.shape[0]
  states_memmap[idx:idx+batch_size] = states.numpy()
  moves_memmap[idx:idx+batch_size] = moves.numpy()
  idx += batch_size
  print(f"Merged batch {bf} ({batch_size} samples)")

states_memmap.flush()
moves_memmap.flush()

states_final = torch.from_numpy(np.array(states_memmap))
moves_final = torch.from_numpy(np.array(moves_memmap))
torch.save((states_final, moves_final), out_file)
print(f"Final dataset saved to {out_file} with {total_samples} samples")

for f in batch_files:
  os.remove(f)
os.rmdir(tmp_dir)
os.remove('states_memmap.npy')
os.remove('moves_memmap.npy')
print("Temporary files cleaned up.")

