from abc import abstractmethod

from game_logic.board import Board
from policies.policy import Policy
from utils.color import Color
from utils.types import Actions, Action


class UntrainablePolicy(Policy):
	def __str__(self) -> str:
		return f'Untrainable{super().__str__()}'

	@abstractmethod
	def get_action(self, board: Board, legal_actions: Actions, color: Color) -> Action:
		raise NotImplementedError
