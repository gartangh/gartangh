from typing import List

import numpy as np
from numpy.random import choice

from policies.trainable_policy import TrainablePolicy
from utils.types import Actions, Action, Directions, Location


class TopKNormalizedTrainablePolicy(TrainablePolicy):
	def __init__(self, board_size: int, k: int) -> None:
		assert 1 < k, f'Invalid k: k must be greater than 1, but got {k}'

		self.board_size: int = board_size
		self.k: int = k

	def __str__(self) -> str:
		return f'TopKNormalized{super().__str__()}'

	def get_action(self, legal_actions: Actions, q_values: np.array) -> Action:
		indices: List[int] = [row * self.board_size + col for (row, col) in list(legal_actions)]
		q_values: np.array = q_values[0, indices]
		k: int = self.k if self.k <= len(legal_actions) else len(legal_actions)
		indices: List[int] = np.argpartition(q_values, -k)[-k:]
		q_values: np.array = q_values[indices]
		q_sum = np.sum(q_values)
		if q_sum <= 1e-40:
			q_values /= q_sum
		else:
			q_values: np.array([1/k]*k)
		locations: np.array = np.array(list(legal_actions))[indices]
		index: int = choice(np.arange(len(q_values)), p=q_values)
		location: Location = locations[index]
		directions: Directions = legal_actions[(location[0], location[1])]
		action: Action = (location, directions)

		return action
