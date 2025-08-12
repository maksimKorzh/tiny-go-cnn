###################################
#
#  MCTS wrapper for Go Neural Net
#
###################################

import sys
import copy
import math
import torch
import random
import numpy as np
from model import *
from copy import deepcopy

###################################
#
#              CONFIG
#
###################################

TOP_MOVES = 2  # explore this many
PLAYOUTS = 10
KOMI = 7.5

###################################
#
#              GOBAN
#
###################################

PASS = -1
NONE = -1
EMPTY = 0
BLACK = 1
WHITE = 2
FENCE = 3
ESCAPE = 4
BOARD_SIZE = 19
WIDTH = 21

board = [[]]
side = NONE
ko = [NONE, NONE]
groups = []
best_move = NONE

def init_board():
  global board, side, ko, groups
  board = [[0 for _ in range(WIDTH)] for _ in range(WIDTH)]
  for row in range(WIDTH):
    for col in range(WIDTH):
      if row == 0 or row == WIDTH-1 or col == 0 or col == WIDTH-1:
        board[row][col] = FENCE
  side = BLACK
  ko = [NONE, NONE]
  groups = [[], []]

def print_board():
  for row in range(WIDTH):
    for col in range(WIDTH):
      if col == 0 and row != 0 and row != WIDTH-1:
        rown = WIDTH-row-1
        print((' ' if rown < 10 else ''), rown, end=' ')
      if board[row][col] == FENCE: continue
      if col == ko[0] and row == ko[1]: print('#', end=' ')
      else: print(['.', 'X', 'O', '#'][board[row][col]], end=' ')
    if row < WIDTH-1: print()
  print('   ', 'A B C D E F G H J K L M N O P Q R S T'[:WIDTH*2-4])
  print('\n    Side to move:', ('BLACK' if side == 1 else 'WHITE'), file=sys.stderr)
  print()

def print_groups():
  print('    Black groups:')
  for group in groups[BLACK-1]: print('      ', group)
  print('\n    White groups:')
  for group in groups[WHITE-1]: print('      ', group)
  print()

def count(col, row, color, marks):
  stone = board[row][col]
  if stone == FENCE: return
  if stone and (stone & color) and marks[row][col] == EMPTY:
    marks[row][col] = stone
    count(col+1, row, color, marks)
    count(col-1, row, color, marks)
    count(col, row+1, color, marks)
    count(col, row-1, color, marks)
  elif stone == EMPTY:
    marks[row][col] = ESCAPE

def add_stones(marks, color):
  group = {'stones': [], 'liberties' :[]}
  for row in range(WIDTH):
    for col in range(WIDTH):
      stone = marks[row][col]
      if stone == FENCE or stone == EMPTY: continue
      if stone == ESCAPE: group['liberties'].append((col, row))
      else: group['stones'].append((col, row))
  return group

def make_group(col, row, color):
  marks = [[EMPTY for _ in range(WIDTH)] for _ in range(WIDTH)]
  count(col, row, color, marks)
  return add_stones(marks, color)

def update_groups():
  global groups
  groups = [[], []]
  for row in range(WIDTH):
    for col in range(WIDTH):
      stone = board[row][col]
      if stone == FENCE or stone == EMPTY: continue
      if stone == BLACK:
        group = make_group(col, row, BLACK)
        if group not in groups[BLACK-1]: groups[BLACK-1].append(group)
      if stone == WHITE:
        group = make_group(col, row, WHITE)
        if group not in groups[WHITE-1]: groups[WHITE-1].append(group)

def is_clover(col, row):
  clover_color = -1
  other_color = -1
  for stone in [board[row][col+1], board[row][col-1], board[row+1][col], board[row-1][col]]:
    if stone == FENCE: continue
    if stone == EMPTY: return EMPTY
    if clover_color == -1:
      clover_color = stone
      other_color = (3-clover_color)
    elif stone == other_color: return EMPTY
  return clover_color

def is_suicide(col, row, color):
  suicide = False
  board[row][col] = color
  marks = [[EMPTY for _ in range(WIDTH)] for _ in range(WIDTH)]
  count(col, row, color, marks)
  group = add_stones(marks, color)
  if len(group['liberties']) == 0: suicide = True
  board[row][col] = EMPTY
  return suicide

def is_atari(col, row, color):
  atari = False
  board[row][col] = color
  marks = [[EMPTY for _ in range(WIDTH)] for _ in range(WIDTH)]
  count(col, row, color, marks)
  group = add_stones(marks, color)
  if len(group['liberties']) == 1: atari = True
  board[row][col] = EMPTY
  return atari

def play(col, row, color):
  global ko, side
  ko = [NONE, NONE]
  board[row][col] = color
  update_groups()
  for group in groups[(3-color-1)]:
    if len(group['liberties']) == 0:
      if len(group['stones']) == 1 and is_clover(col, row) == (3-color):
        ko = group['stones'][0]
      for stone in group['stones']:
        board[stone[1]][stone[0]] = EMPTY
  side = (3-color)

def move_to_string(move):
  global WIDTH
  col = chr(move[0]-(1 if move[0]<=8 else 0)+ord('A'))
  row = str(WIDTH-move[1]-1)
  return col+row

def encode_tensor(pos):
  board_2d = pos.reshape(BOARD_SIZE, BOARD_SIZE)
  black_plane = (board_2d == 1).astype(np.float32)
  white_plane = (board_2d == 2).astype(np.float32)
  state = np.stack([black_plane, white_plane])
  tensor = torch.tensor(state).unsqueeze(0)  # batch dim
  return tensor

def encode_position():
  arr = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int8)
  for r in range(1, BOARD_SIZE+1):
    for c in range(1, BOARD_SIZE+1):
      stone = board[r][c]
      idx = (r-1) * BOARD_SIZE + (c-1)
      if stone == BLACK: arr[idx] = 1
      elif stone == WHITE: arr[idx] = 2
  return arr

def policy(rollout):
  color = side
  pos = encode_position()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  with torch.no_grad():
    input_tensor = encode_tensor(pos).to(device)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

  move_indices = probs.argsort()[::-1]
  legal_moves = []
  for i, best_move_idx in enumerate(move_indices):
    row, col = divmod(best_move_idx, BOARD_SIZE)
    if board[row+1][col+1] == EMPTY and (col+1, row+1) != ko and not is_suicide(col+1, row+1, color):
      legal_moves.append(best_move_idx)
  if rollout:
    if legal_moves[0] == move_indices[0]: return legal_moves
    else: return []
  else: return legal_moves

###################################
#
#               MCTS
#
###################################

class Node:
  def __init__(self, parent=None):
    self.parent = parent
    self.children = {}      # move (int) -> Node
    self.visits = 0
    self.value_sum = 0.0

  @property
  def value(self):
    if self.visits == 0: return 0.0
    return self.value_sum / self.visits

def select_puct(node):
  total_visits = sum(child.visits for child in node.children.values())
  best_score = -1e9
  best_move = None
  best_child = None
  PUCT_C = 1.5
  for move, child in node.children.items():
    u = PUCT_C * (math.sqrt(total_visits) / (1 + child.visits)) if total_visits > 0 else PUCT_C
    score = (child.value) + u
    if score > best_score:
      best_score = score
      best_move = move
      best_child = child
  return best_move, best_child

def evaluate():
  black = 0
  white = 0
  for r in board:
    for c in r:
      if c == BLACK: black += 1
      elif c == WHITE: white += 1
  score = black - (white + KOMI)
  result = 1 if score > 0 else -1
  return result if side == BLACK else -result

def policy_rollout():
  global board, groups, side, ko
  max_moves = 3 # kind of lookahead
  passes = 0
  moves_played = 0
  while passes < 2 and moves_played < max_moves:
    pol = policy(True)
    if not pol: move = PASS
    else: move = pol[0]
    if move == PASS:
      passes += 1
      side = (3-side)
      ko = [NONE, NONE]
    else:
      passes = 0
      row, col = divmod(move, BOARD_SIZE)
      play(col + 1, row + 1, side)
    moves_played += 1
  return evaluate()

def mcts_root_search(color, playouts=PLAYOUTS):
  global board, groups, side, ko
  root = Node(parent=None)
  pol = policy(False)
  top_moves = pol[:TOP_MOVES]
  if PASS not in top_moves: top_moves.append(PASS)
  for playout in range(playouts):
    node = root
    path = [node]
    
    # Preserve board and state
    old_board = deepcopy(board)
    old_groups = deepcopy(groups)
    old_side = side
    old_ko = ko

    # Selection
    while node.children:
      move, child = select_puct(node)
      if move == PASS:
        side = 3 - side
        ko = [NONE, NONE]
      else:
        row, col = divmod(move, BOARD_SIZE)
        play(col + 1, row + 1, side)
      node = child
      path.append(node)

    # Expansion
    pol_leaf = policy(False)[:TOP_MOVES]
    if PASS not in pol_leaf: pol_leaf.append(PASS)
    for mv in pol_leaf:
      if mv not in node.children:
        node.children[mv] = Node(parent=node)

    # Simulation
    value = policy_rollout()

    # Backpropagation
    v = value
    for n in reversed(path):
      n.visits += 1
      n.value_sum += v
      v = -v

    # Restore board and state
    board = old_board
    groups = old_groups
    side = old_side
    ko = old_ko

    print(f'\nMCTS stats after {playout+1} simulations:', file=sys.stderr)
    for move, child in root.children.items():
        avg_value = child.value
        print(f'Move: {move}, Visits: {child.visits}, Avg Value: {avg_value:.3f}', file=sys.stderr)
  # Choose best move from root - by highest visit count (common choice)
  if not root.children: return PASS  # no legal moves
  best_move, best_child = max(root.children.items(), key=lambda it: it[1].visits)
  return best_move

def genmove(color):
  best_move = mcts_root_search(color)
  if best_move != PASS:
    row, col = divmod(best_move, BOARD_SIZE)
    play(col+1, row+1, color)
    return 'ABCDEFGHJKLMNOPQRST'[col] + str(BOARD_SIZE - row)
  else: return 'pass'

###################################
#
#               GTP
#
###################################

init_board();
model = CMKGoCNN() # TinyGoCNN()
checkpoint = torch.load("tiny-go-cnn.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])

while True:
  command = input()
  if 'name' in command: print('= Tiny Go CNN\n')
  elif 'protocol_version' in command: print('= 2\n');
  elif 'version' in command: print('=', 'by Code Monkey King\n')
  elif 'list_commands' in command: print('= protocol_version\n')
  elif 'boardsize' in command: WIDTH = int(command.split()[1])+2; print('=\n')
  elif 'clear_board' in command: init_board(); print('=\n')
  elif 'showboard' in command: print('= ', end=''); print_board()
  elif 'play' in command:
    if 'pass'.upper() not in command:
      params = command.split()
      color = BLACK if params[1] == 'B' else WHITE
      col = ord(params[2][0])-ord('A')+(1 if ord(params[2][0]) <= ord('H') else 0)
      row = WIDTH-int(params[2][1:])-1
      play(col, row, color)
      print('=\n')
    else:
      side = (3-side)
      ko = [NONE, NONE]
      print('=\n')
  elif 'genmove' in command:
    parts = command.split()
    if len(parts) < 2: print('? syntax error'); continue
    color = BLACK if parts[1].lower() == 'b' else WHITE
    move = genmove(color)
    print(f"= {move}\n")
  elif 'quit' in command: sys.exit()
  else: print('=\n')
