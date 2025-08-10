import sys
import os
import numpy as np
import torch
from sgfmill import boards
from model import *

BOARD_SIZE = 19
LETTERS = 'ABCDEFGHJKLMNOPQRST'  # 19 letters, no I

def encode_board_1d_to_tensor(board_1d):
  board_2d = board_1d.reshape(BOARD_SIZE, BOARD_SIZE)
  black_plane = (board_2d == 1).astype(np.float32)
  white_plane = (board_2d == 2).astype(np.float32)
  state = np.stack([black_plane, white_plane])
  tensor = torch.tensor(state).unsqueeze(0)  # batch dim
  return tensor

def predict_best_move(model, board_1d):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()
  with torch.no_grad():
    input_tensor = encode_board_1d_to_tensor(board_1d).to(device)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    best_move_idx = torch.argmax(probs, dim=1).item()
  return best_move_idx

class GtpEngine:
  def __init__(self, model_path):
    self.board = boards.Board(BOARD_SIZE)
    self.model = TinyGoCNN()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.model.eval()
    self.pass_count = 0

  def board_to_1d(self):
    arr = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int8)
    for r in range(BOARD_SIZE):
      for c in range(BOARD_SIZE):
        stone = self.board.get(r, c)
        idx = r * BOARD_SIZE + c
        if stone == 'b': arr[idx] = 1
        elif stone == 'w': arr[idx] = 2
    return arr

  def genmove(self, color):
    board_1d = self.board_to_1d()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)
    self.model.eval()

    with torch.no_grad():
      input_tensor = encode_board_1d_to_tensor(board_1d).to(device)
      output = self.model(input_tensor)
      probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    move_indices = probs.argsort()[::-1]
    for best_move_idx in move_indices:
      row, col = divmod(best_move_idx, BOARD_SIZE)
      try:
        result = self.board.play(row, col, color)
        if result is not False:
          self.pass_count = 0
          return coords_to_gtp(row, col)
      except ValueError:
        self.pass_count += 1
        print('PASS', file=sys.stderr)
        return 'pass'
        # Alternatively consider another move but this often results in illegal move
        #continue

    self.pass_count += 1
    return 'pass'

  def play(self, color, move):
    if move.lower() == 'pass':
      self.pass_count += 1
      return True
    self.pass_count = 0
    try: row, col = gtp_to_coords(move)
    except Exception: return False
    result = self.board.play(row, col, color)
    return result is not False

  def is_game_over(self):
      return self.pass_count >= 2

def coords_to_gtp(row, col):
  return LETTERS[col] + str(BOARD_SIZE - row)

def gtp_to_coords(move):
  col_letter = move[0].upper()
  if col_letter not in LETTERS:
    print(f"ERROR: {col_letter}", file=sys.stderr)
    raise ValueError(f"Invalid column letter: {col_letter}")
  col = LETTERS.index(col_letter)
  row_num = int(move[1:])
  row = BOARD_SIZE - row_num
  if row < 0 or row >= BOARD_SIZE:
    print(f"ERROR: {row_num}", file=sys.stderr)
    raise ValueError(f"Invalid row number: {row_num}")
  return row, col

def showboard(board):
  print('=')
  print('  ' + ' '.join(LETTERS))
  for r in range(BOARD_SIZE):
    row_str = []
    for c in range(BOARD_SIZE):
      stone = board.get(r, c)
      if stone == 'b': row_str.append('X')
      elif stone == 'w': row_str.append('O')
      else: row_str.append('.')
    print(f"{BOARD_SIZE - r:2d} " + ' '.join(row_str))
  print()

model_path = 'tiny-go-cnn.pth'
if not os.path.exists(model_path):
  print(f"Error: checkpoint '{model_path}' not found.", file=sys.stderr)
  sys.exit(1)

engine = GtpEngine(model_path)
for line in sys.stdin:
  line = line.strip()
  if line == '': continue
  parts = line.split()
  cmd = parts[0].lower()
  if cmd == 'quit': print('='); break
  elif cmd == 'name': print('= tiny go CNN\n')
  elif cmd == 'version': print('= 1.0\n')
  elif cmd == 'protocol_version': print('= 2\n')
  elif cmd == 'boardsize':
    if len(parts) < 2 or int(parts[1]) != BOARD_SIZE: print('? unacceptable size')
    else: print('=\n')
  elif cmd == 'clear_board':
    engine.board = boards.Board(BOARD_SIZE)
    engine.pass_count = 0
    print('=\n')
  elif cmd == 'play':
    if len(parts) < 3: print('? syntax error'); continue
    color = parts[1].lower()
    move = parts[2]
    if engine.play(color, move): print('=\n')
    else: print('? illegal move')
  elif cmd == 'genmove':
    if len(parts) < 2: print('? syntax error'); continue
    color = parts[1].lower()
    move = engine.genmove(color)
    print(f"= {move}\n")
  elif cmd == 'showboard': print('= '); showboard(engine.board); print('\n')
  else: print('=\n')
  sys.stdout.flush()
