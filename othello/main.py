from typing import List

from agents.agent import Agent
from agents.cnn_trainable_agent import CNNTrainableAgent
from agents.human_agent import HumanAgent
from agents.trainable_agent import TrainableAgent
from agents.untrainable_agent import UntrainableAgent
from policies.epsilon_greedy_annealing_trainable_policy import EpsilonGreedyAnnealingTrainablePolicy
from policies.random_untrainable_policy import RandomUntrainablePolicy
from policies.top_k_normalized_trainable_policy import TopKNormalizedTrainablePolicy
from policies.weights_untrainable_policy import WeightsUntrainablePolicy
from rewards.fixed_reward import FixedReward
from rewards.no_reward import NoReward
from utils.color import Color
from utils.config import Config
from utils.global_config import GlobalConfig
from utils.risk_regions import risk_regions, bench

if __name__ == '__main__':
	# board size
	board_size: int = 8

	# trainable black agent
	black: TrainableAgent = CNNTrainableAgent(
		color=Color.BLACK,
		train_policy=EpsilonGreedyAnnealingTrainablePolicy(
			inner_policy=TopKNormalizedTrainablePolicy(board_size=board_size, k=3),
			start_epsilon=1.0,
			stop_epsilon=0.0,
		),
		immediate_reward=NoReward(),
		final_reward=FixedReward(win=1, draw=0.5, loss=0),
		board_size=board_size,
	)

	# white agent
	white: Agent = CNNTrainableAgent(
		color=Color.WHITE,
		train_policy=EpsilonGreedyAnnealingTrainablePolicy(
			inner_policy=TopKNormalizedTrainablePolicy(board_size=board_size, k=3),
			start_epsilon=1.0,
			stop_epsilon=0.0,
		),
		immediate_reward=NoReward(),
		final_reward=FixedReward(win=1, draw=0.5, loss=0),
		board_size=board_size,
	)

	# share same networks:
	white.dnn = black.dnn

	# train strategy
	train_configs: List[Config] = [
		# self play
		Config(
			white=white,
			train_white=True,
			num_episodes=1000,
			verbose=False,
			verbose_live=False,
		),
	]

	# eval configs
	eval_configs: List[Config] = [
		Config(
			white=UntrainableAgent(color=Color.WHITE, policy=RandomUntrainablePolicy()),
			num_episodes=100,
			verbose=True,
			verbose_live=False,
		),
		Config(
			white=UntrainableAgent(color=Color.WHITE, policy=WeightsUntrainablePolicy(risk_regions(board_size))),
			num_episodes=100,
			verbose=True,
			verbose_live=False,
		),
		Config(
			white=UntrainableAgent(color=Color.WHITE, policy=WeightsUntrainablePolicy(bench(board_size))),
			num_episodes=100,
			verbose=True,
			verbose_live=False,
		),
	]

	# test configs
	test_configs: List[Config] = [
		Config(
			white=UntrainableAgent(color=Color.WHITE, policy=RandomUntrainablePolicy()),
			num_episodes=1000,
			verbose=True,
			verbose_live=False,
		),
		Config(
			white=UntrainableAgent(color=Color.WHITE, policy=WeightsUntrainablePolicy(risk_regions(board_size))),
			num_episodes=1000,
			verbose=True,
			verbose_live=False,
		),
		Config(
			white=UntrainableAgent(color=Color.WHITE, policy=WeightsUntrainablePolicy(bench(board_size))),
			num_episodes=1000,
			verbose=True,
			verbose_live=False,
		),
	]

	# human configs
	human_configs: List[Config] = [
		Config(
			white=HumanAgent(color=Color.WHITE),
			num_episodes=3,
			verbose=True,
			verbose_live=False,
		),
	]

	# run all configs
	GlobalConfig(board_size, black, train_configs, eval_configs, test_configs, human_configs).start()
