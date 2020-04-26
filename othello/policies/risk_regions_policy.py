import random
from typing import Union

import numpy as np

from policies.policy import Policy
from utils.risk_regions import risk_regions
from utils.types import Actions, Location, Directions, Action, Locations


class RiskRegionsPolicy(Policy):
	def __init__(self, board_size: int) -> None:
		self._weights: np.array = risk_regions(board_size)

	def __str__(self) -> str:
		return f'RiskRegions{super().__str__()}'

	def get_action(self, legal_actions: Actions) -> Action:
		best_locations: Locations = []
		best_score: Union[int, None] = None
		for location in legal_actions:
			score: int = self._weights[location]
			if best_score is None or score > best_score:
				best_score: int = score
				best_locations: Locations = [location]
			elif score == best_score:
				best_locations.append(location)

		location: Location = random.choice(list(legal_actions))
		legal_directions: Directions = legal_actions[location]
		action: Action = (location, legal_directions)

		return action
