from game_logic.board import Board
from rewards.reward import Reward
from utils.color import Color


class DifferenceReward(Reward):
	def __str__(self) -> str:
		return f'Difference{super().__str__()}'

	def reward(self, board: Board, color: Color) -> float:
		if color is Color.BLACK:
			score: float = board.num_black_disks / (board.num_black_disks + board.num_white_disks)
			prev_score: float = board.prev_num_black_disks / (board.prev_num_black_disks + board.prev_num_white_disks)
		elif color is Color.WHITE:
			score: float = board.num_white_disks / (board.num_black_disks + board.num_white_disks)
			prev_score: float = board.prev_num_white_disks / (board.prev_num_black_disks + board.prev_num_white_disks)
		else:
			raise Exception(f'Invalid color: expected color to be BLACK or WHITE, but got {color.name}')

		reward: float = 100.0 * (score - prev_score)

		return reward
