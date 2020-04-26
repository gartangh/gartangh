import numpy as np

from game_logic.board import Board
from rewards.reward import Reward
from utils.color import Color


class DifferenceWithBoardPrev(Reward):
	"""
	number of player's disks - number of player's disks on the previous board
	"""
	def reward(self, board: Board, color: Color) -> float:
		num_player_disks: int = len(np.where(board.board == color.value)[0])
		num_player_disks_prev: int = len(np.where(board.prev_board == color.value)[0])
		reward: float = 1.0 * (num_player_disks - num_player_disks_prev)

		return reward
