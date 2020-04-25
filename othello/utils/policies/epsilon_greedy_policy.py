import random
from typing import Dict, List, Tuple

import numpy as np
from tensorflow.keras import Sequential

from utils.policies.policy import Policy


class EpsilonGreedyPolicy(Policy):
	def __init__(self, epsilon: float, board_size: int, policy_sampling):
		self.epsilon: float = epsilon
		self.board_size: int = board_size
		self.policy_sampling = policy_sampling

	def __str__(self):
		return f'EpsilonGreedy{super().__str__()}'

	def get_action(self, board: np.ndarray, action_value_network: Sequential,
	               legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> tuple:
		explore: bool = random.random() < self.epsilon
		if explore:
			return random.choice(list(legal_actions.keys()))
		else:
			q_values: np.array = action_value_network.predict(np.expand_dims(board, axis=0))
			indices = [row * self.board_size + col for (row, col) in legal_actions.keys()]
			if self.policy_sampling:
				# turn the Q values of the legal actions into a distribution to choose the next action
				legal_q = [q_values[0, i] for i in indices]
				legal_q /= sum(legal_q)  # normalize
				return list(legal_actions)[np.random.choice(np.arange(0, len(legal_q)), p=legal_q)]
			else:
				# choose the next action legal as the one with the highest Q value
				q_values = q_values[0, indices]
				return list(legal_actions)[q_values.argmax()]
