from abc import abstractmethod

from game_logic.board import Board
from utils.color import Color
from utils.types import Actions, Action


class Agent:
	def __init__(self, color: Color) -> None:
		self.color: Color = color

		self.num_games_won: int = 0

	def __str__(self) -> str:
		return f'Agent (color={self.color.name}'

	def update_score(self, board: Board) -> None:
		if self.color is Color.BLACK and board.num_black_disks > board.num_white_disks:
			self.num_games_won += 1  # BLACK won
		elif self.color is Color.WHITE and board.num_white_disks > board.num_black_disks:
			self.num_games_won += 1  # WHITE won

	def reset(self) -> None:
		self.num_games_won: int = 0

	@abstractmethod
	def next_action(self, board: Board, legal_actions: Actions) -> Action:
		raise NotImplementedError
