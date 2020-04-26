from abc import abstractmethod

import numpy as np

from policies.policy import Policy
from utils.types import Actions, Action


class TrainablePolicy(Policy):
	def __str__(self) -> str:
		return f'Trainable{super().__str__()}'

	@abstractmethod
	def get_action(self, legal_actions: Actions, q_values: np.array) -> Action:
		raise NotImplementedError
