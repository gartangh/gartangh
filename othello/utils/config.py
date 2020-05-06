from math import ceil
from typing import Union

from agents.agent import Agent
from agents.trainable_agent import TrainableAgent
from utils.color import Color


class Config:
	def __init__(self, white: Agent, num_episodes: int, train_white: bool = False, verbose: bool = False, verbose_live: bool = False) -> None:
		# check parameters
		assert white.color is Color.WHITE, f'Invalid white agent: white agent\'s color is not white'
		if not isinstance(white, TrainableAgent):
			assert not train_white, f'Invalid white agent: white agent is not trainable'
		assert 0 <= num_episodes < 100_000, f'Invalid number of episodes: num_episodes should be between 0 and 100000, but got {num_episodes}'
		if not verbose:
			assert not verbose_live, f'Cannot be verbose live if verbose is not set'

		self.white: Agent = white
		self.train_white: bool = train_white
		self.num_episodes: int = num_episodes
		self.verbose: bool = verbose
		self.verbose_live: bool = verbose_live

		self.plot_every_n_episodes: int = ceil(self.num_episodes / 50)
