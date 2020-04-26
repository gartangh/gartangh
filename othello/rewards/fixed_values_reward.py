from game_logic.board import Board
from rewards.reward import Reward
from utils.color import Color


class FixedValuesReward(Reward):
	def __init__(self, win: float, draw: float, lose: float):
		assert win >= draw >= lose, f'Invalid order: win must be greater or equal to draw and draw must be greater or equal to lose'

		self.win: float = win
		self.draw: float = draw
		self.lose: float = lose

	def reward(self, board: Board, color: Color) -> float:
		if color is Color.BLACK:
			if board.num_black_disks > board.num_white_disks:
				return self.win
			elif board.num_black_disks < board.num_white_disks:
				return self.lose
			else:
				return self.draw
		else:
			if board.num_black_disks < board.num_black_disks:
				return self.win
			elif board.num_black_disks > board.num_white_disks:
				return self.draw
			else:
				return self.lose
