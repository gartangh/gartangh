import numpy as np

from game_logic.board import Board
from rewards.reward import Reward
from utils.color import Color


class DifferenceBetweenAgents(Reward):
	"""
	number of players disks - number of opponent's disks
	"""
	def reward(self, board: Board, color: Color) -> float:
		num_player_disks: int = len(np.where(board.board == color.value)[0])
		num_opponent_disks: int = len(np.where(board.board == 1 - color.value)[0])
		reward: float = 1.0 * (num_player_disks - num_opponent_disks)

		return reward
