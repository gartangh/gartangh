import random

import numpy as np

from policies.trainable_policy import TrainablePolicy
from utils.types import Action, Actions, Location, Directions


class EpsilonGreedyTrainablePolicy(TrainablePolicy):
	def __init__(self, inner_policy: TrainablePolicy, epsilon: float) -> None:
		self.inner_policy: TrainablePolicy = inner_policy
		self.epsilon: float = epsilon

	def __str__(self) -> str:
		return f'EpsilonGreedy{super().__str__()}(inner_policy={self.inner_policy})'

	def get_action(self, legal_actions: Actions, q_values: np.array) -> Action:
		if random.random() < self.epsilon:
			location: Location = random.choice(list(legal_actions))
			directions: Directions = legal_actions[location]
			action: Action = (location, directions)
		else:
			action: Action = self.inner_policy.get_action(legal_actions, q_values)

		return action
