from abc import abstractmethod
from typing import Union

import numpy as np

from policies.trainable_policy import TrainablePolicy
from utils.types import Action, Actions


class AnnealingTrainablePolicy(TrainablePolicy):
	def __init__(self, inner_policy: TrainablePolicy) -> None:
		self.inner_policy: TrainablePolicy = inner_policy
		self.num_episodes: Union[int, None] = None

	def __str__(self) -> str:
		return f'Annealing{super().__str__()}'

	@abstractmethod
	def update(self, episode: int) -> None:
		raise NotImplementedError

	def get_action(self, legal_actions: Actions, q_values: np.array) -> Action:
		action: Action = self.inner_policy.get_action(legal_actions, q_values)

		return action
