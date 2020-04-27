import random
from typing import Union

import numpy as np

from game_logic.board import Board
from policies.untrainable_policy import UntrainablePolicy
from utils.color import Color
from utils.types import Actions, Location, Directions, Action, Locations


class WeightsUntrainablePolicy(UntrainablePolicy):
	def __init__(self, weights: np.array) -> None:
		self.weights: np.array = weights

	def __str__(self) -> str:
		return f'Weights{super().__str__()}'

	def get_action(self, board: Board, legal_actions: Actions, color: Color) -> Action:
		best_locations: Locations = []
		best_score: Union[int, None] = None
		for location in legal_actions:
			score: int = self.weights[location]
			if best_score is None or score > best_score:
				best_score: int = score
				best_locations: Locations = [location]
			elif score == best_score:
				best_locations.append(location)

		location: Location = random.choice(list(legal_actions))
		legal_directions: Directions = legal_actions[location]
		action: Action = (location, legal_directions)

		return action
