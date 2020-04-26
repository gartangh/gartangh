import numpy as np

from game_logic.board import Board
from rewards.reward import Reward
from utils.color import Color
from utils.risk_regions import risk_regions


class RiskRegionsReward(Reward):
	def __init__(self, board_size: int) -> None:
		self._weights: np.array = risk_regions(board_size)

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
