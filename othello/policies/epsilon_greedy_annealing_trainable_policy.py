from policies.annealing_trainable_policy import AnnealingTrainablePolicy
from policies.epsilon_greedy_trainable_policy import EpsilonGreedyTrainablePolicy
from policies.trainable_policy import TrainablePolicy


class EpsilonGreedyAnnealingTrainablePolicy(AnnealingTrainablePolicy):
	def __init__(self, inner_policy: TrainablePolicy, start_epsilon: float, stop_epsilon: float) -> None:
		super().__init__(EpsilonGreedyTrainablePolicy(inner_policy, start_epsilon))

		self.start_epsilon: float = start_epsilon
		self.stop_epsilon: float = stop_epsilon

	def __str__(self) -> str:
		return f'EpsilonGreedy{super().__str__()}'

	def update(self, episode: int) -> None:
		progress: float = 1.0 * episode / self.num_episodes
		self.inner_policy.epsilon = self.start_epsilon - progress * (self.start_epsilon - self.stop_epsilon)
