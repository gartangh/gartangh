import numpy as np

from game_logic.board import Board
from utils.color import Color
from utils.immediate_rewards.immediate_reward import ImmediateReward


class BenchHeuristic(ImmediateReward):
	# initialize static variables
	_weights: np.array = np.array([
		[80, -26, 24, -1, -5, 28, -18, 79],
		[-23, -39, -18, -9, -6, -8, -39, -1],
		[46, -16, 4, 1, -3, 6, -20, 52],
		[-13, -5, 2, -1, 4, 3, -12, -2],
		[-5, -6, 1, -2, -3, 0, -9, -5],
		[48, -13, 12, 5, 0, 5, -24, 41],
		[-27, -53, -11, -1, -11, -16, -58, -15],
		[87, -25, 27, -1, 5, 36, -3, 100]
	])

	def __init__(self, board_size: int = 8):
		# check arguments
		assert board_size == 8, f'Invalid board size: board_size should be 8, but got {board_size}'

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
		return 1.0 * (board_score - prev_board_score)

	def final_reward(self, board: Board, color: Color) -> float:
		# TODO tune this reward
		# if board.num_black_disks > board.num_white_disks:
		# 	if color == Color.BLACK:
		# 		return sum(sum(self._weights))
		# 	else:
		# 		return -sum(sum(self._weights))
		# elif board.num_black_disks < board.num_white_disks:
		# 	if color == Color.BLACK:
		# 		return -sum(sum(self._weights))
		# 	else:
		# 		return sum(sum(self._weights))
		return 0
