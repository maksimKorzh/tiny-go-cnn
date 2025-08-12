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

#TOP_MOVES = 18  # explore this many
#MAX_MOVES = 5   # during playout
#PLAYOUTS = 20
#KOMI = 7.5

###################################
#
#              GOBAN
#
###################################

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
        print((' ' if rown < 10 else ''), rown, end=' ', file=sys.stderr)
      if board[row][col] == FENCE: continue
      if col == ko[0] and row == ko[1]: print('#', end=' ', file=sys.stderr)
      else: print(['.', 'X', 'O', '#'][board[row][col]], end=' ', file=sys.stderr)
    if row < WIDTH-1: print(file=sys.stderr)
  print('   ', 'A B C D E F G H J K L M N O P Q R S T'[:WIDTH*2-4], file=sys.stderr)
  print('\n    Side to move:', ('BLACK' if side == 1 else 'WHITE'), file=sys.stderr)
  print(file=sys.stderr)

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

def policy(quick_pass):
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
  if quick_pass:
    if legal_moves[0] == move_indices[0]: return legal_moves
    else: return []
  else: return legal_moves

def evaluate(color):
  black = 0
  white = 0
  for r in board:
    for c in r:
      if c == BLACK: black += 1
      elif c == WHITE: white += 1
  score = black - white
  return score if color == BLACK else -score

def negamax(depth, alpha, beta):
  global board, groups, side, ko, best_move
  if depth == 0:
    score = evaluate()
    return score
  moves = policy(False)[:3]
  #print_board()
  if len(moves):
    for move in genmove(side):
      row, col = divmod(move, BOARD_SIZE)
      old_board = deepcopy(board)
      old_groups = deepcopy(groups)
      old_side = side
      old_ko = ko
      if move != NONE: play(col+1, row+1, side)
      score = 0 #-negamax(depth-1, -beta, -alpha)
      board = old_board
      groups = old_groups
      side = old_side
      ko = old_ko
      if score > alpha:
        if score >= beta: break
        alpha = score
        best_move = move
  best_move = NONE
  return alpha

def root(depth, color):
  global board, groups, side, ko, best_move
  best_score = -10000
  temp_best = NONE
  moves = policy(False)[:5]
  print('root called', moves, file=sys.stderr)
  for move in moves:
    row, col = divmod(move, BOARD_SIZE)
    old_board = deepcopy(board)
    old_groups = deepcopy(groups)
    old_side = side
    old_ko = ko
    if move != NONE: play(col+1, row+1, side)
    score = -negamax(depth-1, -10000, 10000)
    move_string = 'ABCDEFGHJKLMNOPQRST'[col] + str(BOARD_SIZE - row)
    print('>', move_string, move, -score if side == BLACK else score, file=sys.stderr)
    board = old_board
    groups = old_groups
    side = old_side
    ko = old_ko
    if score > best_score:
      best_score = score
      temp_best = move
  best_move = temp_best
  return best_score

def genmove(color):
  best_score = root(5, color)
  if best_move != NONE:
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
  elif 'showboard' in command: print('= ', end=''); print_board(); print('\n')
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
