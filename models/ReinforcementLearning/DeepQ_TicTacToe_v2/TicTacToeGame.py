import torch
import numpy as np
from enum import Enum


class OPPONENT_LEVEL(Enum):
    NAIVE = "naive"
    OPTIMAL = "optimal"
    AGENT = "agent"


class TicTacToeGame():
    __name__      = "TicTacToeGame"
    __author__    = "Yemi Kelani"
    __copyright__ = "Copyright (C) 2023, 2024 Yemi Kelani"
    __license__   = "Apache-2.0 license"
    __version__   = "2.0"
    __github__    = "https://github.com/yemi-kelani/artificial-intelligence"

    def __init__(
            self,
            device: torch.device,
            agent: torch.nn.Module,
            opponent_level: OPPONENT_LEVEL = OPPONENT_LEVEL.NAIVE,
            start_as_X: bool = True):

        self.device = device
        self.agent = agent
        if agent != None:
            self.agent.eval()
        self.set_opponent_level(opponent_level)

        self.winner = None
        self.__lastWinner = None
        self.board = torch.zeros((3, 3))

        self.X = -1
        self.O = 1
        self.SPACE = 0
        self.positions: map[int, str] = {
            self.X: "X",
            self.O: "O",
            self.SPACE: " "
        }

        self.WIN_REWARD = 1
        self.TIE_REWARD = 0
        self.LOSS_REWARD = -1

        self.role = self.X if start_as_X else self.O
        self.agent_role = self.role * -1

        if self.is_X():
            # X always goes first.
            self.make_move()

    def set_opponent_level(self, opponent_level: OPPONENT_LEVEL):
        """
        Parameters:
        opponent_level:
            OPPONENT_LEVEL.NAIVE
            OPPONENT_LEVEL.OPTIMAL
            OPPONENT_LEVEL.AGENT
        """
        valid_level_options = [
            OPPONENT_LEVEL.NAIVE,
            OPPONENT_LEVEL.OPTIMAL,
            OPPONENT_LEVEL.AGENT
        ]
        if opponent_level in valid_level_options:
            self.opponent_level = opponent_level
        else:
            raise Exception(f"""
          (set_opponent_level) Invalid input '{str(opponent_level)}'.
          Valid options are: {valid_level_options}
          """)

    def get_state(self):
        return torch.clone(self.board).to(self.device)

    def convert_action_to_position(self, action: int):
        """
        Parameters:
        action: int - must be in the range of [0,8].
        """
        col = action % 3
        row = action // 3
        return row, col

    def convert_position_to_action(self, row: int, col: int):
        """
        Parameters:  
        row: int - board row.
        col: int - board column.
        """
        return ((3 * row) + col)

    def take_action(self, action: int):
        """
        Parameters:
        action: int - must be in the range of [0,8].
        """
        row, col = self.convert_action_to_position(action)
        if not self.is_valid_move(row, col, self.agent_role):
            raise Exception(f"""
                (take_action:TicTacToeGame.py) Invalid move: {action}.
                """)

        self.board[row][col] = self.agent_role
        reward, done = self.is_game_over()
        
        # if not done:
        #     reward, done = self.move()
        
        next_state = self.get_state()
        return next_state, reward, done
    
    def after_action(self):
        return self.move()

    def is_valid_move(self, row: int, col: int, player: int):
        """
        Parameters:  
        row: int - board row.
        col: int - board column.
        """
        if row >= self.board.size()[0] or row < 0 \
                or col >= self.board.size()[1] or col < 0:
            return False

        if self.board[row][col] != 0:
            # if the cell is not empty
            # the move is not valid
            return False

        # check to see if the # of Xs and Os are balanced
        with torch.no_grad():
            board_sum = self.board.sum() + player
            balanced = abs(board_sum) <= 1
            if not balanced:
                # for debugging
                print(f"⚠ Board is unbalanced ⚠"
                      f"player: {self.positions[player]}",
                      f"row: {row}, column: {col}")
                self.print_board()

        return balanced

    def get_valid_moves(self, board=None):
        board = self.board if board == None else board
        valid_moves, indicies = [], []
        mask = torch.zeros((9,)).to(self.device)
        for j in range(board.size()[0]):
            for i in range(board.size()[1]):
                if board[j][i] == 0:
                    valid_moves.append((j, i))
                    index = self.convert_position_to_action(j, i)
                    indicies.append(index)
                    mask[index] = 1

        return valid_moves, mask, indicies

    def move(self):
        remainder = torch.where(self.board != 0, 1.0, 0.0).sum() % 2
        if self.is_X() and remainder == 1 or not self.is_X() and remainder == 0:
            raise Exception(f"""
                            (move:TicTacToeGame.py) 
                            Attempting to make a move during the opponent's turn. 
                            """)

        self.make_move()
        reward, done = self.is_game_over()
        return reward, done

    def make_move(self):
        valid_moves, mask, _ = self.get_valid_moves()

        force_naive_move = False
        num_moves = len(valid_moves)
        if num_moves == 0:
            return
        elif num_moves > 7 and self.opponent_level == OPPONENT_LEVEL.OPTIMAL:
            force_naive_move = True

        if self.opponent_level == OPPONENT_LEVEL.NAIVE or force_naive_move:
            # random choice
            move = valid_moves[np.random.choice(len(valid_moves))]
        elif self.opponent_level == OPPONENT_LEVEL.AGENT:
            # deep neural network
            self.agent.eval()
            with torch.no_grad():
                q_values = self.agent.forward(self.get_state())
                q_masked = torch.where(mask != 0, q_values, -1000)
                action = torch.argmax(q_masked)
                row, col = self.convert_action_to_position(action)
                move = [row, col]
        elif self.opponent_level == OPPONENT_LEVEL.OPTIMAL:
            # optimal choice
            scores = []
            board = self.get_state().detach()
            for move in valid_moves:
                board[move[0]][move[1]] = self.role
                scores.append(self.minmax(self.role, self.agent_role, board))
                board[move[0]][move[1]] = 0

            move = valid_moves[torch.argmax(torch.tensor(scores))]

        self.board[move[0]][move[1]] = self.role

    def minmax(self, role, player, board):
        _, done = self.is_game_over(board)
        if done:
            # NOTE: minmax rewards != Q value rewards
            if self.__lastWinner != None:
                # return 1 if role wins else return -1
                # -1 *  1 = -1 [role "X" loses to "O"]
                # -1 * -1 =  1 [role "X" beats op "O"]
                #  1 *  1 =  1 [role "O" beats op "X"]
                #  1 * -1 = -1 [role "O" loses to "X"]
                return role * self.__lastWinner

            # return 0 in case of a tie
            return 0

        scores = []
        moves, _, _ = self.get_valid_moves(board=board)
        next_player = player * -1
        for move in moves:
            board[move[0]][move[1]] = player
            scores.append(self.minmax(role, next_player, board))
            board[move[0]][move[1]] = 0

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
        tie = self.winner == None \
            and torch.where(board != 0, 1.0, 0.0).sum() == 9

        # compute reward
        reward = self.TIE_REWARD if tie else 0
        if self.winner != None:
            if self.winner == self.agent_role:
                reward = self.WIN_REWARD  # reward for winning
            else:
                reward = self.LOSS_REWARD  # reward for losing

        done = self.winner != None or tie

        # if this function is being used on
        # another board (i.e. in minmax),
        # revert the stored winner to None
        if revertWinner:
            self.__lastWinner = self.winner
            self.winner = None

        return reward, done

    def is_X(self):
        """
        Returns a boolean specifying whether the 
        opponent (this instance) is playing as X.
        """
        return self.role == self.X

    def flip_role(self):
        self.agent_role = self.role
        self.role *= -1

    def reset_board(self):
        self.board.detach()
        del self.board
        self.board = torch.zeros((3, 3))

    def reset(self, flip_roles=True):
        self.winner = None
        self.__lastWinner = None
        self.reset_board()

        if flip_roles:
            self.flip_role()

        if self.is_X():
            # X always goes first.
            self.make_move()

    def print_state(self):
        self.print_board()

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
