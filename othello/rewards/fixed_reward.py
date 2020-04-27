from game_logic.board import Board
from rewards.reward import Reward
from utils.color import Color


class FixedReward(Reward):
	def __init__(self, win: float, draw: float, loss: float):
		assert win >= draw >= loss, f'Invalid order: win must be greater or equal to draw and draw must be greater or equal to lose'

		self.win: float = win
		self.loss: float = loss
		self.draw: float = draw

	def __str__(self) -> str:
		return f'Fixed{super().__str__()}'

	def reward(self, board: Board, color: Color) -> float:
		if color is Color.BLACK:
			if board.num_black_disks > board.num_white_disks:
				return self.win
			elif board.num_black_disks < board.num_white_disks:
				return self.loss
			else:
				return self.draw
		elif color is Color.WHITE:
			if board.num_black_disks < board.num_black_disks:
				return self.win
			elif board.num_black_disks > board.num_white_disks:
				return self.draw
			else:
				return self.loss
		else:
			raise Exception(f'Invalid color: expected color to be BLACK or WHITE, but got {color.name}')
