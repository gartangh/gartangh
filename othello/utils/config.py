from math import ceil

from game_logic.agents.agent import Agent
from game_logic.agents.trainable_agent import TrainableAgent
from utils.color import Color


class Config:
	def __init__(self, black: Agent, train_black: bool, white: Agent, train_white: bool, num_episodes: int,
	             plot_win_ratio: bool = True, plot_win_ratio_live: bool = False, verbose: bool = False,
	             verbose_live: bool = False, random_start: bool = False) -> None:
		# check parameters
		assert black.color is Color.BLACK, f'Invalid black agent: black agent\'s color is not black'
		if not isinstance(black, TrainableAgent):
			assert not train_black, f'Invalid black agent: black agent is not trainable'
		assert white.color is Color.WHITE, f'Invalid white agent: white agent\'s color is not white'
		if not isinstance(white, TrainableAgent):
			assert not train_white, f'Invalid white agent: white agent is not trainable'
		assert 0 <= num_episodes <= 10000000, f'Invalid number of episodes: num_episodes should be between 0 and 10000, but got {num_episodes}'
		if not isinstance(black, TrainableAgent):
			assert not plot_win_ratio, f'Cannot plot win ratio if black agent is not trainable'
		if not plot_win_ratio:
			assert not plot_win_ratio_live, f'Cannot plot win ratio live if plot win ratio is not set'
		if not verbose:
			assert not verbose_live, f'Cannot be verbose live if verbose is not set'

		self.black: Agent = black
		self.train_black: bool = train_black
		self.white: Agent = white
		self.train_white: bool = train_white
		self.num_episodes: int = num_episodes
		self.plot_win_ratio: bool = plot_win_ratio
		self.plot_win_ratio_live: bool = plot_win_ratio_live
		self.verbose: bool = verbose
		self.verbose_live: bool = verbose_live
		self.random_start: bool = random_start

		self.plot_every_n_episodes: int = ceil(self.num_episodes / 100)

		self.black.num_games_won = 0
		if isinstance(black, TrainableAgent):
			black.set_train_mode(train_black)
		self.white.num_games_won = 0
		if isinstance(white, TrainableAgent):
			black.set_train_mode(train_white)
