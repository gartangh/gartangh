import numpy as np
from tensorflow.keras import Sequential

from utils.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from utils.policies.policy import Policy


class AnnealingEpsilonGreedyPolicy(Policy):
	def __init__(self, start_eps, end_eps, n_steps, board_size, allow_exploration):
		self.start_eps = start_eps
		self.end_eps = end_eps
		self.n_steps = n_steps
		self.inner_policy = EpsilonGreedyPolicy(self.start_eps, board_size)
		self.decisions_made = 0
		self.episode = 1
		self.allow_exploration = allow_exploration
		self.curr_eps = start_eps

	def __str__(self):
		return f'AnnealingEpsilonGreedy{super().__str__()}'

	def update_policy(self, episode: int) -> None:
		self.episode = episode

	@property
	def current_eps_value(self) -> float:
		if not self.allow_exploration:
			value = self.get_linear_value()
		# Exploration allowed
		elif self.episode == 101 or self.episode == 201 or self.episode == 301 or self.episode == 401:
			value = (self.start_eps + self.curr_eps) / 2
		else:
			value = self.get_linear_value()
		self.curr_eps = value
		return value

	def get_current_eps_policy(self) -> EpsilonGreedyPolicy:
		self.inner_policy.epsilon = self.curr_eps
		return self.inner_policy

	def get_action(self, board: np.ndarray, action_value_network: Sequential, legal_actions: dict) -> tuple:
		self.decisions_made += 1
		return self.get_current_eps_policy().get_action(board, action_value_network, legal_actions)

	def get_linear_value(self) -> float:
		# Linear annealed: f(x) = ax + b.
		a = -float(self.start_eps - self.end_eps) / float(self.n_steps)
		b = float(self.start_eps)
		value = max(self.end_eps, a * float(self.decisions_made) + b)
		return value
