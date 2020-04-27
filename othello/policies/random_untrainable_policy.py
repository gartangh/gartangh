import random

from game_logic.board import Board
from policies.policy import Policy
from policies.untrainable_policy import UntrainablePolicy
from utils.color import Color
from utils.types import Actions, Location, Directions, Action


class RandomUntrainablePolicy(UntrainablePolicy):
	def __str__(self) -> str:
		return f'Random{super().__str__()}'

	def get_action(self, board: Board, legal_actions: Actions, color: Color) -> Action:
		location: Location = random.choice(list(legal_actions))
		directions: Directions = legal_actions[location]
		action: Action = (location, directions)

		return action
