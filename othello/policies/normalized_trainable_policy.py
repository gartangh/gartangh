from typing import List

import numpy as np
from numpy.random import choice

from policies.trainable_policy import TrainablePolicy
from utils.types import Actions, Action, Location, Directions


class NormalizedTrainablePolicy(TrainablePolicy):
	def __init__(self, board_size: int, k: int) -> None:
		assert 1 < k, f'Invalid k: k must be greater than 1, but got {k}'

		self.board_size: int = board_size
		self.k: int = k

	def __str__(self) -> str:
		return f'Normalized{super().__str__()}'

	def get_action(self, legal_actions: Actions, q_values: np.array) -> Action:
		indices: List[int] = [row * self.board_size + col for (row, col) in list(legal_actions)]
		q_values: np.array = q_values[0, indices]
		q_values /= sum(q_values)
		index: int = choice(np.arange(len(q_values)), p=q_values)
		location: Location = list(legal_actions)[index]
		directions: Directions = legal_actions[location]
		action: Action = (location, directions)

		return action
