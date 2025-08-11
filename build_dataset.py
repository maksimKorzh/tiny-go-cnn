import os
import torch
import numpy as np
from sgfmill import sgf, boards

BOARD_SIZE = 19
base_dir = 'games'
out_file = 'games.pt'
start_year = 2000
end_year = 2017
max_games = 20000 # 5000 for TinyGoCNN, 20000 for Detlef44

def board_to_tensor(board):
  black_plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
  white_plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
  for row in range(BOARD_SIZE):
    for col in range(BOARD_SIZE):
      stone = board.get(row, col)
      if stone == 'b':
        black_plane[row, col] = 1.0
      elif stone == 'w':
        white_plane[row, col] = 1.0
  return np.stack([black_plane, white_plane])

def move_to_index(move):
  return move[0] * BOARD_SIZE + move[1]

def process_sgf_file(sgf_path):
  with open(sgf_path, "rb") as f: sgf_game = sgf.Sgf_game.from_bytes(f.read())
  board = boards.Board(BOARD_SIZE)
  moves = []
  nodes = list(sgf_game.get_main_sequence())
  for node in nodes[1:]:
    color, move = node.get_move()
    if move is None: continue
    state_tensor = board_to_tensor(board)
    move_idx = move_to_index(move)
    moves.append((state_tensor, move_idx))
    board.play(move[0], move[1], color)
  return moves

all_samples = []
years = [str(y) for y in range(start_year, end_year + 1)]
games_processed = 0
for year in years:
  year_path = os.path.join(base_dir, year)
  if not os.path.isdir(year_path):
    print(f"Warning: Year folder not found: {year_path}")
    continue
  for subfolder in sorted(os.listdir(year_path)):
    subfolder_path = os.path.join(year_path, subfolder)
    if not os.path.isdir(subfolder_path): continue
    for file in sorted(os.listdir(subfolder_path)):
      if not file.endswith('.sgf'): continue
      sgf_path = os.path.join(subfolder_path, file)
      try:
        samples = process_sgf_file(sgf_path)
        all_samples.extend(samples)
        games_processed += 1
        if games_processed % 100 == 0: print(f"Processed {games_processed} games...")
        if max_games is not None and games_processed >= max_games: break
      except: pass
    if max_games is not None and games_processed >= max_games: break
  if max_games is not None and games_processed >= max_games: break

print(f"Total games processed: {games_processed}")
print(f"Total samples collected: {len(all_samples)}")
states = torch.tensor(np.array([s[0] for s in all_samples]))
moves = torch.tensor(np.array([s[1] for s in all_samples]))
torch.save((states, moves), out_file)
print(f"Dataset saved to {out_file}")
