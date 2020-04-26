import random

import numpy as np

from policies.random_policy import RandomPolicy
from policies.trainable_policy import TrainablePolicy
from utils.types import Action, Actions


class EpsilonGreedyTrainablePolicy(TrainablePolicy):
	def __init__(self, inner_policy: TrainablePolicy, epsilon: float) -> None:
		self.inner_policy: TrainablePolicy = inner_policy
		self.epsilon: float = epsilon

		self.random_policy: RandomPolicy = RandomPolicy()

	def __str__(self) -> str:
		return f'EpsilonGreedy{super().__str__()}'

	def get_action(self, legal_actions: Actions, q_values: np.array) -> Action:
		if random.random() < self.epsilon:
			action: Action = self.random_policy.get_action(legal_actions)
		else:
			action: Action = self.inner_policy.get_action(legal_actions, q_values)

		return action
