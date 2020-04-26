from typing import List

from colorama import init
from termcolor import colored
from tqdm import tqdm

from agents.cnn_trainable_agent import CNNTrainableAgent
from agents.human_agent import HumanAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from agents.risk_regions_agent import RiskRegionsAgent
from agents.trainable_agent import TrainableAgent
from game_logic.game import Game
from gui.controller import Controller
from policies.annealing_trainable_policy import AnnealingTrainablePolicy
from policies.epsilon_greedy_annealing_trainable_policy import EpsilonGreedyAnnealingTrainablePolicy
from policies.top_k_random_trainable_policy import TopKRandomTrainablePolicy
from rewards.minimax_heuristic import MinimaxHeuristic
from rewards.risk_regions_reward import RiskRegionsReward
from rewards.fixed_values_reward import FixedValuesReward
from utils.color import Color
from utils.config import Config
from utils.global_config import GlobalConfig
from utils.plot import Plot


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
		game: Game = Game(global_config, config, episode)
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

	print_scores(config.num_episodes, black.num_games_won, white.num_games_won)

	# save models
	if isinstance(black, TrainableAgent) and black.train_mode:
		black.final_save()
	if isinstance(white, TrainableAgent) and white.train_mode:
		white.final_save()

	# reset agents
	black.reset()
	white.reset()


def print_scores(num_episodes: int, black_won: int, white_won: int) -> None:
	ties: int = num_episodes - black_won - white_won
	if black_won > white_won:
		print(colored(f'BLACK {black_won:>5}/{num_episodes:>5} ({black_won:>5}|{white_won:>5}|{ties:>5})\n', 'red'))
	elif black_won < white_won:
		print(colored(f'WHITE {white_won:>5}/{num_episodes:>5} ({black_won:>5}|{white_won:>5}|{ties:>5})\n', 'green'))
	else:
		print(colored(f'DRAW  {black_won:>5}/{num_episodes:>5} ({black_won:>5}|{white_won:>5}|{ties:>5})\n', 'cyan'))


if __name__ == '__main__':
	# one-time global configuration
	global_config: GlobalConfig = GlobalConfig(board_size=8, gui_size=400)

	# trainable black agent
	black: TrainableAgent = CNNTrainableAgent(
		color=Color.BLACK,
		train_policy=EpsilonGreedyAnnealingTrainablePolicy(
			inner_policy=TopKRandomTrainablePolicy(board_size=global_config.board_size, k=3),
			start_epsilon=1.0,
			stop_epsilon=0.0,
		),
		immediate_reward=RiskRegionsReward(board_size=global_config.board_size),
		final_reward=FixedValuesReward(win=1000, draw=100, lose=-1000),
		board_size=global_config.board_size,
	)

	# init plot
	plot: Plot = Plot(black)

	# train strategy
	train_configs: List[Config] = [
		# RandomAgent
		Config(
			black=black,
			train_black=True,
			white=RandomAgent(color=Color.WHITE),
			train_white=False,
			num_episodes=200,
			plot=plot,
			plot_win_ratio_live=True,
			verbose=False,
			verbose_live=False,
			random_start=True,
		),
		# RiskRegionsAgent
		Config(
			black=black,
			train_black=True,
			white=RiskRegionsAgent(color=Color.WHITE, board_size=global_config.board_size),
			train_white=False,
			num_episodes=100,
			plot=plot,
			plot_win_ratio_live=True,
			verbose=False,
			verbose_live=False,
			random_start=True,
		),
		# MiniMaxAgent
		Config(
			black=black,
			train_black=True,
			white=MinimaxAgent(color=Color.WHITE,
			                   immediate_reward=MinimaxHeuristic(board_size=global_config.board_size),
			                   depth=2,
			                   ),
			train_white=False,
			num_episodes=100,
			plot=plot,
			plot_win_ratio_live=True,
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
			white=RandomAgent(color=Color.WHITE),
			train_white=False,
			num_episodes=100,
			plot=None,
			plot_win_ratio_live=False,
			verbose=True,
			verbose_live=False,
			random_start=False,
		)
	]
	for config in test_configs:
		main()

	# play against a human
	config: Config = Config(
		black=black,
		train_black=False,
		white=HumanAgent(color=Color.WHITE),
		train_white=False,
		num_episodes=3,
		plot=None,
		plot_win_ratio_live=False,
		verbose=True,
		verbose_live=False,
		random_start=False,
	)
	main()
