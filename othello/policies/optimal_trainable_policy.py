from typing import List

import numpy as np

from policies.trainable_policy import TrainablePolicy
from utils.types import Actions, Action, Location, Directions


class OptimalTrainablePolicy(TrainablePolicy):
	def __init__(self, board_size: int) -> None:
		self.board_size: int = board_size

	def __str__(self) -> str:
		return f'Optimal{super().__str__()}'

	def get_action(self, legal_actions: Actions, q_values: np.array) -> Action:
		indices: List[int] = [row * self.board_size + col for (row, col) in list(legal_actions)]
		q_values: np.array = q_values[0, indices]
		index: int = q_values.argmax()
		location: Location = list(legal_actions)[index]
		directions: Directions = legal_actions[location]
		action: Action = (location, directions)

		return action
