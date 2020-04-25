from typing import List

import matplotlib.pyplot as plt
from colorama import init
from termcolor import colored
from tqdm import tqdm

from game_logic.agents.cnn_dqn_trainable_agent import CNNDQNTrainableAgent
from game_logic.agents.dqn_trainable_agent import DQNTrainableAgent
from game_logic.agents.human_agent import HumanAgent
from game_logic.agents.minimax_agent import MinimaxAgent
from game_logic.agents.random_agent import RandomAgent
from game_logic.agents.risk_regions_agent import RiskRegionsAgent
from game_logic.agents.trainable_agent import TrainableAgent
from game_logic.game import Game
from gui.controller import Controller
from utils.color import Color
from utils.config import Config
from utils.global_config import GlobalConfig
from utils.immediate_rewards.minimax_heuristic import MinimaxHeuristic


def main() -> None:
	# initialize colors
	init()

	# agents
	black = config.black
	white = config.white
	print(f'\nAgents:\n\t{black}\n\t{white}\n')

	win_rates = [0.0]
	if isinstance(black, DQNTrainableAgent):
		epsilons = [black.training_policy.current_eps_value]
	last_matches = []

	# initialize live plot
	if config.plot_win_ratio_live:
		plt.ion()  # non-blocking plot
		plt.title('Win ratio of black (red), epsilon (green)')
		plt.xlabel('number of games played')
		plt.ylabel('win ratio and epsilon')
		plt.draw()
		plt.pause(0.001)

	for episode in tqdm(range(1, config.num_episodes + 1)):
		# create new game
		game: Game = Game(global_config, config, episode)
		if isinstance(white, HumanAgent):
			# create GUI controller
			controller: Controller = Controller(game)
			controller.start()
		elif isinstance(black, DQNTrainableAgent):
			# update epsilon annealing policy
			black.training_policy.update_policy(episode)
			# play game
			game.play()
		else:
			# play game
			game.play()

		# plot win ratio
		if config.plot_win_ratio:
			if game.board.num_black_disks > game.board.num_white_disks:
				last_matches.append(1)
			elif game.board.num_black_disks < game.board.num_white_disks:
				last_matches.append(-1)
			else:
				last_matches.append(0)

			if episode % config.plot_every_n_episodes == config.plot_every_n_episodes - 1 and len(last_matches) > 0:
				win_rates.append(sum(last_matches) / len(last_matches))
				if isinstance(black, DQNTrainableAgent):
					epsilons.append(black.training_policy.current_eps_value)
				if config.plot_win_ratio_live:
					plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], win_rates,
					         color='red')
					if isinstance(black, TrainableAgent):
						plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], epsilons,
						         color='green')
					plt.draw()
					plt.pause(0.001)

				last_matches = []

	print_scores(config.num_episodes, black.num_games_won, white.num_games_won)

	# plot win ratio
	if config.plot_win_ratio and not config.plot_win_ratio_live:
		# show plot
		plt.title('Win ratio of black (red), epsilon (green)')
		plt.xlabel('number of games played')
		plt.ylabel('win ratio and epsilon')
		plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], win_rates, color='red')
		plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], epsilons, color='green')
		plt.show()

	# save models
	if isinstance(black, TrainableAgent) and black.train_mode:
		black.final_save()
	if isinstance(white, TrainableAgent) and white.train_mode:
		white.final_save()

	# reset agents
	black.num_games_won = 0
	white.num_games_won = 0


def print_scores(num_episodes: int, black_won: int, white_won: int):
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
	black: TrainableAgent = CNNDQNTrainableAgent(
		Color.BLACK,
		immediate_reward=MinimaxHeuristic(global_config.board_size),
		board_size=global_config.board_size,
		start_epsilon=0.99,
		end_epsilon=0.01,
		epsilon_steps=20_000,
		policy_sampling = False,
	)

	# train strategy
	train_configs: List[Config] = [
		# RandomAgent
		Config(
			black=black,
			train_black=True,
			white=RandomAgent(Color.WHITE),
			train_white=False,
			num_episodes=200,
			plot_win_ratio=True,
			plot_win_ratio_live=True,
			verbose=False,
			verbose_live=False,
			random_start=True,
		),
		# RiskRegionsAgent
		Config(
			black=black,
			train_black=True,
			white=RiskRegionsAgent(Color.WHITE, board_size=global_config.board_size),
			train_white=False,
			num_episodes=200,
			plot_win_ratio=True,
			plot_win_ratio_live=True,
			verbose=False,
			verbose_live=False,
			random_start=True,
		),
		# MiniMaxAgent
		Config(
			black=black,
			train_black=True,
			white=MinimaxAgent(
				Color.WHITE,
				immediate_reward=MinimaxHeuristic(board_size=global_config.board_size),
				depth=2,
			),
			train_white=False,
			num_episodes=20,
			plot_win_ratio=True,
			plot_win_ratio_live=True,
			verbose=False,
			verbose_live=False,
			random_start=True,
		),
	]
	for config in train_configs:
		main()

	# test strategy
	test_configs: List[Config] = [
		Config(
			black=black,
			train_black=False,
			white=RandomAgent(Color.WHITE),
			train_white=False,
			num_episodes=100,
			plot_win_ratio=False,
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
		white=HumanAgent(Color.WHITE),
		train_white=False,
		num_episodes=3,
		plot_win_ratio=False,
		plot_win_ratio_live=False,
		verbose=True,
		verbose_live=False,
		random_start=False,
	)
	main()
