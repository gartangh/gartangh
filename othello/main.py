from typing import List

from colorama import init
from tqdm import tqdm

from agents.agent import Agent
from agents.cnn_trainable_agent import CNNTrainableAgent
from agents.dense_trainable_agent import DenseTrainableAgent
from agents.human_agent import HumanAgent
from agents.trainable_agent import TrainableAgent
from agents.untrainable_agent import UntrainableAgent
from game_logic.game import Game
from gui.controller import Controller
from policies.annealing_trainable_policy import AnnealingTrainablePolicy
from policies.epsilon_greedy_annealing_trainable_policy import EpsilonGreedyAnnealingTrainablePolicy
from policies.random_untrainable_policy import RandomUntrainablePolicy
from policies.top_k_normalized_trainable_policy import TopKNormalizedTrainablePolicy
from policies.weights_untrainable_policy import WeightsUntrainablePolicy
from rewards.difference_reward import DifferenceReward
from rewards.no_reward import NoReward
from rewards.fixed_reward import FixedReward
from utils.color import Color
from utils.config import Config
from utils.plot import Plot
from utils.risk_regions import risk_regions, bench


def main() -> None:
	# initialize colors
	init()

	# agents
	black = config.black
	white = config.white
	print(f'\nAgents:\n\t{black}\n\t{white}\n')

	# initialize training policy
	if isinstance(black, TrainableAgent) and isinstance(black.train_policy, AnnealingTrainablePolicy):
		black.train_policy.num_episodes = config.num_episodes
	if isinstance(white, TrainableAgent) and isinstance(white.train_policy, AnnealingTrainablePolicy):
		white.train_policy.num_episodes = config.num_episodes

	# initialize live plot
	if config.plot is not None:
		config.plot.set_plot_live(config.plot_win_ratio_live)

	for episode in tqdm(range(1, config.num_episodes + 1)):
		# create new game
		game: Game = Game(board_size, config, episode)
		if isinstance(white, HumanAgent):
			# create GUI controller
			controller: Controller = Controller(game)
			# play game
			controller.start()
		else:
			if isinstance(black, TrainableAgent) and isinstance(black.train_policy, AnnealingTrainablePolicy):
				# update epsilon annealing policy
				black.train_policy.update(episode)
			if isinstance(white, TrainableAgent) and isinstance(black.train_policy, AnnealingTrainablePolicy):
				# update epsilon annealing policy
				white.train_policy.update(episode)

			# play game
			game.play()

		# plot win ratio
		if config.plot is not None:
			config.plot.update(game.board.num_black_disks, game.board.num_white_disks, episode,
			                   config.plot_every_n_episodes)

	ties: int = config.num_episodes - black.num_games_won - white.num_games_won
	print(f'({black.num_games_won:>4}|{white.num_games_won:>4}|{ties:>4}) / {config.num_episodes:>4}')
	print(f'win ratio: {(black.num_games_won - white.num_games_won) / config.num_episodes}\n')

	# save models
	if isinstance(black, TrainableAgent) and black.train_mode:
		black.final_save()
	if isinstance(white, TrainableAgent) and white.train_mode:
		white.final_save()

	# reset agents
	black.reset()
	white.reset()


if __name__ == '__main__':
	# board size
	board_size: int = 8

	# trainable black agent
	black: Agent = CNNTrainableAgent(
		color=Color.BLACK,
		train_policy=EpsilonGreedyAnnealingTrainablePolicy(
			inner_policy=TopKNormalizedTrainablePolicy(board_size=board_size, k=3),
			start_epsilon=0.1,
			stop_epsilon=0,
		),
		immediate_reward=NoReward(),
		final_reward=FixedReward(win=1, draw=0.5, loss=0),
		board_size=board_size,
	)

	# init plot
	if isinstance(black, TrainableAgent):
		plot: Plot = Plot(black)
	else:
		plot: None = None

	# train strategy
	train_configs: List[Config] = [
		# random
		Config(
			black=black,
			train_black=True,
			white=UntrainableAgent(color=Color.WHITE, policy=RandomUntrainablePolicy()),
			train_white=False,
			num_episodes=25_000,
			plot=plot,
			plot_win_ratio_live=False,
			verbose=False,
			verbose_live=False,
			random_start=True,
		),
		# risk regions
		Config(
			black=black,
			train_black=True,
			white=UntrainableAgent(color=Color.WHITE, policy=WeightsUntrainablePolicy(risk_regions(board_size))),
			train_white=False,
			num_episodes=25_000,
			plot=plot,
			plot_win_ratio_live=False,
			verbose=False,
			verbose_live=False,
			random_start=True,
		),
		# bench
		Config(
			black=black,
			train_black=True,
			white=UntrainableAgent(color=Color.WHITE, policy=WeightsUntrainablePolicy(bench(board_size))),
			train_white=False,
			num_episodes=25_000,
			plot=plot,
			plot_win_ratio_live=False,
			verbose=False,
			verbose_live=False,
			random_start=True,
		),
	]
	for config in train_configs:
		main()

	# after training is complete, save the plot
	plot.save_plot()
	plot.set_plot_live(False)

	# test strategy
	test_configs: List[Config] = [
		Config(
			black=black,
			train_black=False,
			white=UntrainableAgent(color=Color.WHITE, policy=RandomUntrainablePolicy()),
			train_white=False,
			num_episodes=1_000,
			plot=None,
			plot_win_ratio_live=False,
			verbose=True,
			verbose_live=False,
			random_start=False,
		),
		Config(
			black=black,
			train_black=False,
			white=UntrainableAgent(color=Color.WHITE, policy=WeightsUntrainablePolicy(risk_regions(board_size))),
			train_white=False,
			num_episodes=1_000,
			plot=None,
			plot_win_ratio_live=False,
			verbose=True,
			verbose_live=False,
			random_start=False,
		),
		Config(
			black=black,
			train_black=False,
			white=UntrainableAgent(color=Color.WHITE, policy=WeightsUntrainablePolicy(bench(board_size))),
			train_white=False,
			num_episodes=1_000,
			plot=None,
			plot_win_ratio_live=False,
			verbose=True,
			verbose_live=False,
			random_start=False,
		),

	]
	for config in test_configs:
		main()

	# play against a human
	# config: Config = Config(
	# 	black=black,
	# 	train_black=False,
	# 	white=HumanAgent(color=Color.WHITE),
	# 	train_white=False,
	# 	num_episodes=3,
	# 	plot=None,
	# 	plot_win_ratio_live=False,
	# 	verbose=True,
	# 	verbose_live=False,
	# 	random_start=False,
	# )
	# main()
