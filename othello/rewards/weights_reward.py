import numpy as np

from game_logic.board import Board
from rewards.reward import Reward
from utils.color import Color


class WeightsReward(Reward):
	def __init__(self, weights: np.array) -> None:
		self.weights: np.array = weights

	def __str__(self) -> str:
		return f'Weights{super().__str__()}'

	def evaluate_board(self, board: np.array, color: Color) -> float:
		player_disks: np.array = np.where(board == color.value, 1, 0)
		opponent_disks: np.array = np.where(board == 1 - color.value, -1, 0)
		disks: np.array = np.add(player_disks, opponent_disks)
		weighted_disks: np.array = np.multiply(disks, self.weights)
		score: float = float(np.sum(weighted_disks))

		return score

	def reward(self, board: Board, color: Color) -> float:
		score: float = self.evaluate_board(board.board, color)
		prev_score: float = self.evaluate_board(board.prev_board, color)
		reward: float = score - prev_score

		return reward
