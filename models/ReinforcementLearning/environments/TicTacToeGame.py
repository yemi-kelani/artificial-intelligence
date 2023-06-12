import torch

# torch config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TicTacToeGame():
  def __init__(self, device, opponent_level="easy"):
    self.device = device
    self.positions = {-1: "X", 1: "O", 0: " "}
    self.opponent_level = opponent_level
    self.board = torch.zeros((3, 3))
    self.player1 = -1   # the agent
    self.player2 = 1    # the opponent
    self.winner = None
    self.minmaxWinner = None

  def get_state(self):
    return torch.clone(self.board).to(self.device)
  
  def take_action(self, action):
    i = action % 3
    j = action // 3
    if self.is_valid_move(i, j):
      self.board[j][i] = self.player1
      reward, done = self.is_game_over()

      if not done:
        self.opponent_move()
        reward, done = self.is_game_over()
    else:
      reward, done = -10, True

    next_state = torch.clone(self.board).to(self.device)
    return next_state, reward, done

  def is_valid_move(self, i, j):
    # check if the cell is empty
    if self.board[j][i] != 0:
      print(f"action {(3*j)+i} is not valid.")
      self.print_board()
      return False

    # check to see if the Xs and Os are balanced
    board_sum = self.board.sum() + self.player1
    balanced = board_sum >= -1 and board_sum <= 1
    if not balanced:
      print(f"board is unbalanced.")
      self.print_board()

    return balanced

  def get_valid_moves(self, board=None):
    board = self.board if board == None else board
    valid_moves, indicies = [], []
    mask = torch.zeros((9,)).to(self.device)
    for i in range(3):
      for j in range(3):
        if board[j][i] == 0:
          valid_moves.append((j, i))
          index = (3 * j) + i
          indicies.append(index)
          mask[index] = 1

    return valid_moves, mask, indicies

  def opponent_move(self):
    # with torch.no_grad():
    valid_moves, _, _ = self.get_valid_moves()

    force_naive_move = True
    num_moves = len(valid_moves)
    if num_moves == 0:
      return
    elif num_moves <= 7:
      force_naive_move = False

    if self.opponent_level == "easy" or force_naive_move:
      # random choice
      move = valid_moves[np.random.choice(len(valid_moves))]
    elif self.opponent_level == "hard":
      # optimal choice
      scores = []
      board = torch.clone(self.board).detach()
      moves, _, _ = self.get_valid_moves(board=board)
      for move in moves:
        board[move[0]][move[1]] = self.player2 
        scores.append(self.minmax(self.player2, self.player1, board))
        board[move[0]][move[1]] = 0

      move = moves[np.argmax(scores)]
    
    self.board[move[0]][move[1]] = self.player2
  
  def minmax(self, role, player, board):
    _, done = self.is_game_over(board)
    if done:
      # NOTE: minmax rewards != Q value rewards
      if self.minmaxWinner != None:
        # return 1 if role wins else return -1
        # -1 *  1 = -1 [role "X" loses to "O"]
        # -1 * -1 =  1 [role "X" beats op "O"]
        #  1 *  1 =  1 [role "O" beats op "X"]
        #  1 * -1 = -1 [role "O" loses to "X"]
        return role * self.minmaxWinner

      # return 0 in case of a tie
      return 0

    scores = []
    moves, _, _ = self.get_valid_moves(board=board)
    next_player = self.player1 if player == self.player2 else self.player2
    for move in moves:
      board[move[0]][move[1]] = player                     # make move
      scores.append(self.minmax(role, next_player, board)) # score move
      board[move[0]][move[1]] = 0                          # revert board

    return max(scores) if player == role else min(scores)

  def is_game_over(self, board=None):
    revertWinner = False if board == None else True
    board = self.board if board == None else board

    # check rows and columns
    for i in range(self.board.size()[0]):
      if board[i][0] == board[i][1] == board[i][2] != 0:
        self.winner = board[i][0]
      elif board[0][i] == board[1][i] == board[2][i] != 0:
        self.winner = board[0][i]
    
    # check diagonals
    if board[0][0] == board[1][1] == board[2][2] != 0:
      self.winner = board[0][0]
    elif board[0][2] == board[1][1] == board[2][0] != 0:
      self.winner = board[0][2]

    # check if there's a tie
    tie = torch.where(board != 0, 1.0, 0.0).sum() == 9

    reward = -1 if tie else 0
    if self.winner != None:
      if self.winner == self.player1:
        reward = 1  # reward for winning
      else:
        reward = -2 # reward for losing

    done = self.winner != None or tie

    # if this function is being used for minmax, 
    # revert the stored winner to None
    if revertWinner:
      self.minmaxWinner = self.winner
      self.winner = None

    return reward, done

  def reset(self):
    self.board = torch.zeros((3, 3))
    self.player1 = -1   # the agent
    self.player2 = 1    # the opponent
    self.winner = None
    self.minmaxWinner = None

  def print_board(self, board=None):
    board = self.board if board == None else board
    print("_______")
    for a in range(board.size()[0]):
      row = r""
      for b in range(board.size()[1]):
        row += "|" + self.positions[board[a][b].item()]
        row += "|" if b == board.size()[1] - 1 else ""
      print(row)
    print("‾‾‾‾‾‾‾")