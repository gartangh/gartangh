from game_logic.agents.agent import Agent
from game_logic.agents.risk_regions_agent import RiskRegionsAgent
from utils.color import Color
from colorama import init
from termcolor import colored

from game_logic.game import Game
from game_logic.agents.random_agent import RandomAgent
from game_logic.agents.minimax_agent import MinimaxAgent

from utils.immediate_rewards.minimax_heuristic import MinimaxHeuristic


def main():
	# initialize colors
	init()

	# check global variables
	assert 1 <= num_episodes <= 10000, f'Invalid number of episodes: num_episodes should be between 1 and 10000, but got {num_episodes}'
	assert black.num_games_won == 0, f'Invalid black agent'
	assert white.num_games_won == 0, f'Invalid white agent'

	print(f'\nAgents:\n\t{black}\n\t{white}\n')

	for episode in range(1, num_episodes + 1):
		# create new game
		game: Game = Game(episode, black, white, board_size, verbose)
		# play game
		game.play()

	print()

	ties: int = num_episodes - black.num_games_won - white.num_games_won
	if black.num_games_won > white.num_games_won:
		print(colored(
			f'BLACK {black.num_games_won:>5}/{num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})',
			'red'))
	elif black.num_games_won < white.num_games_won:
		print(colored(
			f'WHITE {white.num_games_won:>5}/{num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})',
			'green'))
	elif black.num_games_won == white.num_games_won:
		print(colored(
			f'DRAW  {black.num_games_won:>5}/{num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})',
			'cyan'))
	else:
		raise Exception('The scores were miscalculated')

	print()


if __name__ == "__main__":
	# initialize global variables
	board_size: int = 8  # the size of the board e.g. 8x8
	black: Agent = MinimaxAgent(color=Color.BLACK, immediate_reward=MinimaxHeuristic(board_size))  # the black agent
	white: Agent = RiskRegionsAgent(color=Color.WHITE)  # the white agent
	num_episodes: int = 100  # the number of episodes e.g. 100
	verbose: bool = False  # wetter or not to print intermediate steps

	# call main
	main()
