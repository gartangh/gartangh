import numpy as np

from game_logic.board import Board
from utils.immediate_rewards.immediate_reward import ImmediateReward
from utils.risk_regions import risk_regions


class MinimaxHeuristic(ImmediateReward):
	def __init__(self, board_size: int = 8):
		self._weights: np.array = risk_regions(board_size)

	def _evaluate_board(self, board: np.array, color_value: int):
		player_disks: np.array = np.where(board == color_value, 1, 0)
		opponent_disks: np.array = np.where(board == 1 - color_value, -1, 0)
		disks: np.array = np.add(player_disks, opponent_disks)
		weighted_disks: np.array = np.multiply(disks, self._weights)
		result: int = int(np.sum(weighted_disks))  # will be an int, but numpy says it will be np.array

		return result

	def immediate_reward(self, board: Board, color_value: int) -> float:
		board_score: int = self._evaluate_board(board.board, color_value)
		prev_board_score: int = self._evaluate_board(board.prev_board, color_value)
		immediate_reward: float = 1.0 * (board_score - prev_board_score)

		return immediate_reward
