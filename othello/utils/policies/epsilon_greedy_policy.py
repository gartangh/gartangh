import random
from typing import Dict, List, Tuple

import numpy as np
from tensorflow.keras import Sequential

from utils.policies.policy import Policy


class EpsilonGreedyPolicy(Policy):
	def __init__(self, epsilon: float, board_size: int):
		self.epsilon: float = epsilon
		self.board_size: int = board_size

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
			q_values = q_values[0, indices]  # q_values has shape (1,x)
			return list(legal_actions)[q_values.argmax()]
