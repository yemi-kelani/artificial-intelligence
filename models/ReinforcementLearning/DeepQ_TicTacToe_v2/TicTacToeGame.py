import torch
import numpy as np
from enum import Enum
from .logger import get_logger


class OPPONENT_LEVEL(Enum):
    NAIVE = "naive"
    OPTIMAL = "optimal"
    AGENT = "agent"


class TicTacToeGame():
    __name__      = "TicTacToeGame"
    __author__    = "Yemi Kelani"
    __copyright__ = "Copyright (C) 2023, 2025 Yemi Kelani"
    __license__   = "Apache-2.0 license"
    __version__   = "2.0"
    __github__    = "https://github.com/yemi-kelani/artificial-intelligence"

    def __init__(
        self,
        device: torch.device,
        agent: torch.nn.Module,
        opponent_level: OPPONENT_LEVEL = OPPONENT_LEVEL.NAIVE,
        start_as_X: bool = True
    ):

        self.device = device
        self.agent = agent
        # Don't change agent mode in constructor - let training loop manage this
        self.set_opponent_level(opponent_level)
        self.LOG_DETAILS = False  # Add logging control
        self.logger = get_logger(self.__class__.__name__)

        self.__lastWinner = None
        self.board = torch.zeros((3, 3))

        self.X = -1
        self.O = 1
        self.SPACE = 0
        self.positions: dict[int, str] = {
            self.X: "X",
            self.O: "O",
            self.SPACE: " "
        }

        # Base rewards (clean signal for Q-learning)
        self.WIN_REWARD = 1.0
        self.TIE_REWARD = 0.3
        self.LOSS_REWARD = -1.0

        self.role = self.X if start_as_X else self.O
        self.agent_role = self.O if start_as_X else self.X

        if self.is_X():
            # X always goes first.
            self.move()

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
        if action not in range(0,9):
            raise Exception(f"""
                (convert_action_to_position:TicTacToeGame.py) Invalid action: {action}.
                Agent Role: {self.agent_role} ({self.positions[self.agent_role]}),
                State: {self.get_state()}.
                """)
        row = action // 3
        col = action % 3
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
        Once the agent's action is taken, this class expects
        a call to the "move" method before another agent 
        action is taken.
        """
        row, col = self.convert_action_to_position(action)
        if not self.is_valid_move(row, col, self.agent_role):
            raise Exception(f"""
                (take_action:TicTacToeGame.py) Invalid move... action: {action}, row: {row}, col: {col}.
                Agent Role: {self.agent_role} ({self.positions[self.agent_role]}),
                State: {self.get_state()}.
                """)

        self.board[row][col] = self.agent_role
        reward, done, _ = self.is_game_over()

        next_state = self.get_state()
        return next_state, reward, done

    def is_valid_move(self, row: int, col: int, player: int):
        """
        Parameters:  
        row: int - board row.
        col: int - board column.
        """
        if row >= self.board.size()[0] or row < 0 \
                or col >= self.board.size()[1] or col < 0:
            return False

        # if the cell is not empty
        # the move is not valid
        if self.board[row][col] != 0:
            return False

        # check to see if the # of Xs and Os are balanced
        with torch.no_grad():
            board_sum = self.board.sum() + player
            balanced = abs(board_sum) <= 1
            if not balanced:
                # for debugging
                print(f"⚠ Board is unbalanced ⚠",
                      f"player: {self.positions[player]}",
                      f"row: {row}, column: {col}",
                      "Are the players moving in the correct order?")
                self.print_board()

        return balanced

    def get_valid_moves(self, board=None):
        board = self.board if board is None else board
        valid_moves, indices = [], []
        mask = torch.zeros((9,)).to(self.device)
        for j in range(board.size()[0]):
            for i in range(board.size()[1]):
                if board[j][i] == 0:
                    valid_moves.append((j, i))
                    index = self.convert_position_to_action(j, i)
                    indices.append(index)
                    mask[index] = 1

        return valid_moves, mask, indices

    def move(self):
        remainder = torch.where(self.board != 0, 1.0, 0.0).sum() % 2
        if self.is_X() and remainder == 1 or not self.is_X() and remainder == 0:
            error = "(move:TicTacToeGame.py) Attempting to make a move during the opponent's turn."
            raise Exception(error)

        force_naive_move = False

        valid_moves, mask, _ = self.get_valid_moves()
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
                state = self.get_state().float().reshape(-1) 
                q_values = self.agent.forward(state)
                q_masked = torch.where(mask != 0, q_values, -1e9)
                action = torch.argmax(q_masked)
                row, col = self.convert_action_to_position(action)
                move = [row, col]
        elif self.opponent_level == OPPONENT_LEVEL.OPTIMAL:
            # min max algorithm
            scores = []
            board = self.get_state().detach()
            for move in valid_moves:
                board[move[0]][move[1]] = self.role
                scores.append(self.minmax(self.role, self.agent_role, board))
                board[move[0]][move[1]] = 0

            move = valid_moves[torch.argmax(torch.tensor(scores))]

        self.board[move[0]][move[1]] = self.role
        
        reward, done, _ = self.is_game_over()
        next_state = self.get_state()
        return next_state, reward, done


    def minmax(self, role, player, board):
        _, done, _ = self.is_game_over(board)
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
        board = self.board if board is None else board
        winner = None

        # check rows and columns
        for i in range(board.size()[0]):
            if board[i][0] == board[i][1] == board[i][2] != 0:
                winner = board[i][0].item()
                break
            elif board[0][i] == board[1][i] == board[2][i] != 0:
                winner = board[0][i].item()
                break

        # check diagonals (only if no winner found yet)
        if winner is None:
            if board[0][0] == board[1][1] == board[2][2] != 0:
                winner = board[0][0].item()
            elif board[0][2] == board[1][1] == board[2][0] != 0:
                winner = board[0][2].item()

        # check if there's a tie
        tie = winner is None \
            and torch.where(board != 0, 1, 0).sum().item() == 9

        # compute reward, convention is return
        # rewards for the agent, not the environment
        reward = 0.0
        if tie:
            reward = self.TIE_REWARD
        elif winner is not None:
            if winner == self.agent_role:
                reward = self.WIN_REWARD  # reward for winning
            else:
                reward = self.LOSS_REWARD  # reward for losing
        # Note: No intermediate rewards - cleaner learning signal

        done = winner is not None or tie
        self.__lastWinner = winner
        return reward, done, winner

    def is_X(self):
        """
        Returns a boolean specifying whether the 
        opponent/environment (this instance) is 
        playing as X.
        """
        return self.role == self.X

    def flip_roles(self):
        # Swap roles: opponent and agent switch sides
        self.role *= -1
        self.agent_role *= -1

    def reset_board(self):
        self.board.detach()
        del self.board
        self.board = torch.zeros((3, 3))

    def reset(self, flip_roles=True):
        self.__lastWinner = None
        self.reset_board()

        if flip_roles:
            self.flip_roles()

        if self.is_X():
            # X always goes first.
            self.move()

    def print_state(self):
        self.print_board()

    def print_board(self, board=None):
        board = self.board if board is None else board
        print("_______")
        for a in range(board.size()[0]):
            row = ""
            for b in range(board.size()[1]):
                row += "|" + self.positions[board[a][b].item()]
                row += "|" if b == board.size()[1] - 1 else ""
            print(row)
        print("‾‾‾‾‾‾‾")
