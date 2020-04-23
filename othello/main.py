import matplotlib.pyplot as plt
from colorama import init
from termcolor import colored
from tqdm import tqdm

from game_logic.agents.cnn_dqn_trainable_agent import CNNDQNTrainableAgent
from game_logic.agents.dqn_trainable_agent import DQNTrainableAgent
from game_logic.agents.human_agent import HumanAgent
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
		plt.show()

	for episode in tqdm(range(1, config.num_episodes + 1)):
		# create new game
		game: Game = Game(global_config, config, episode)
		if isinstance(white, HumanAgent):
			# create GUI controller
			controller: Controller = Controller(game)
			controller.start()
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

	# print end score
	ties: int = config.num_episodes - black.num_games_won - white.num_games_won
	if black.num_games_won > white.num_games_won:
		print(colored(
			f'\nBLACK {black.num_games_won:>5}/{config.num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})\n',
			'red'))
	elif black.num_games_won < white.num_games_won:
		print(colored(
			f'\nWHITE {white.num_games_won:>5}/{config.num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})\n',
			'green'))
	else:
		print(colored(
			f'\nDRAW  {black.num_games_won:>5}/{config.num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})\n',
			'cyan'))

	# plot win ratio
	if config.plot_win_ratio_live:
		# keep showing live plot
		plt.ioff()
		plt.show()
	elif config.plot_win_ratio:
		# show plot
		plt.ion()  # non-blocking plot
		plt.title('Win ratio of black (red), epsilon (green)')
		plt.xlabel('number of games played')
		plt.ylabel('win ratio and epsilon')
		plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], win_rates, color='red')
		plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], epsilons, color='green')
		plt.show()
		plt.draw()

	# save models
	if isinstance(black, TrainableAgent) and black.train_mode:
		black.final_save()
	if isinstance(white, TrainableAgent) and white.train_mode:
		white.final_save()


def log(logline: str, path: str = 'log.txt'):
	with open(path, 'a') as f:
		f.write(f'{logline}\n')


def hardcore_training(black, white, board_size, total_iterations: int = 100_000, interval_log: int = 5000):
	black_dqn = black
	white_dqn = white

	total_runs = total_iterations // interval_log
	for i in range(total_runs):
		black = black_dqn
		white = white_dqn
		black.num_games_won = 0
		white.num_games_won = 0

		black.set_train_mode(True)
		white.set_train_mode(True)
		num_episodes = interval_log
		black = black_dqn
		white = white_dqn
		# TODO: use configs
		# main(num_episodes, black, white, board_size, False, False, False)

		black.final_save()
		white.final_save()

		black.set_train_mode(False)
		white.set_train_mode(False)

		black_dqn = black
		white_dqn = white

		tournament_mode = True

		if isinstance(black, TrainableAgent):
			# test against random white
			print('test ' + str(i) + ', BLACK DQN VS WHITE RANDOM')
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			white = RandomAgent(color=Color.WHITE)
			# TODO: use configs
			# main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log('test ' + str(i) + '\tBLACK DQN VS WHITE RANDOM\t' + str(black.num_games_won) + '\t' + str(
				white.num_games_won))

		if isinstance(white, TrainableAgent):
			# test against random black
			black = black_dqn
			white = white_dqn
			print('test ' + str(i) + ', BLACK RANDOM VS WHITE DQN')
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			black = RandomAgent(color=Color.BLACK)
			# TODO: use configs
			# main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log('test ' + str(i) + '\tBLACK RANDOM VS WHITE DQN\t' + str(black.num_games_won) + '\t' + str(
				white.num_games_won))

		if isinstance(black, TrainableAgent):
			# test against risk region white
			black = black_dqn
			white = white_dqn
			print('test ' + str(i) + ', BLACK DQN VS WHITE RISK_REGION')
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			white = RiskRegionsAgent(color=Color.WHITE)
			# TODO: use configs
			# main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log('test ' + str(i) + '\tBLACK DQN VS WHITE RISK_REGION\t' + str(black.num_games_won) + '\t' + str(
				white.num_games_won))

		if isinstance(white, TrainableAgent):
			# test against risk region white
			black = black_dqn
			white = white_dqn
			print('test ' + str(i) + ', BLACK RISK_REGION VS WHITE DQN')
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			black = RiskRegionsAgent(color=Color.BLACK)
			# TODO: use configs
			# main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log('test ' + str(i) + '\tBLACK RISK_REGION VS WHITE DQN\t' + str(black.num_games_won) + '\t' + str(
				white.num_games_won))


if __name__ == '__main__':
	# one-time global configuration
	global_config: GlobalConfig = GlobalConfig(board_size=8, gui_size=400)  # global config

	# TRAIN
	config: Config = Config(
		black=CNNDQNTrainableAgent(
			Color.BLACK,
			immediate_reward=MinimaxHeuristic(global_config.board_size),
			board_size=global_config.board_size
		),
		train_black=True,
		white=RandomAgent(Color.WHITE),
		train_white=False,
		num_episodes=1,
		plot_win_ratio=False,
		plot_win_ratio_live=False,
		verbose=True,
		verbose_live=True,
		random_start=True,
	)
	main()

	# EVALUATE
	config: Config = Config(
		black=config.black,
		train_black=False,
		white=config.white,
		train_white=False,
		num_episodes=100,
		plot_win_ratio=False,
		plot_win_ratio_live=False,
		verbose=True,
		verbose_live=False,
		random_start=False,
	)
	main()

	# HUMAN
	config: Config = Config(
		black=config.black,
		train_black=False,
		white=HumanAgent(Color.WHITE),
		train_white=False,
		num_episodes=2,
		plot_win_ratio=False,
		plot_win_ratio_live=False,
		verbose=True,
		verbose_live=False,
		random_start=False,
	)
	main()
