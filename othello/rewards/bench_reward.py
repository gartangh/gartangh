import numpy as np

from game_logic.board import Board
from rewards.reward import Reward
from utils.color import Color


class BenchReward(Reward):
	def __init__(self, board_size: int) -> None:
		assert board_size == 8, f'Invalid board size: board_size should be 8, but got {board_size}'

		self._weights: np.array = np.array([
			[80, -26, 24, -1, -5, 28, -18, 79],
			[-23, -39, -18, -9, -6, -8, -39, -1],
			[46, -16, 4, 1, -3, 6, -20, 52],
			[-13, -5, 2, -1, 4, 3, -12, -2],
			[-5, -6, 1, -2, -3, 0, -9, -5],
			[48, -13, 12, 5, 0, 5, -24, 41],
			[-27, -53, -11, -1, -11, -16, -58, -15],
			[87, -25, 27, -1, 5, 36, -3, 100]
		])

	def evaluate_board(self, board: np.array, color: Color) -> float:
		player_disks: np.array = np.where(board == color.value, 1, 0)
		opponent_disks: np.array = np.where(board == 1 - color.value, -1, 0)
		disks: np.array = np.add(player_disks, opponent_disks)
		weighted_disks: np.array = np.multiply(disks, self._weights)
		reward: float = float(np.sum(weighted_disks))

		return reward

	def reward(self, board: Board, color: Color) -> float:
		reward: float = self.evaluate_board(board.board, color)

		return reward
