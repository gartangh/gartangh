import random

from policies.policy import Policy
from utils.types import Actions, Location, Directions, Action


class RandomPolicy(Policy):
	def __str__(self) -> str:
		return f'Random{super().__str__()}'

	def get_action(self, legal_actions: Actions) -> Action:
		location: Location = random.choice(list(legal_actions))
		directions: Directions = legal_actions[location]
		action: Action = (location, directions)

		return action
