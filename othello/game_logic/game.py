import copy

import numpy as np
from termcolor import colored

from game_logic.agents.agent import Agent
from game_logic.agents.trainable_agent import TrainableAgent
from game_logic.board import Board
from utils.color import Color
from utils.config import Config
from utils.global_config import GlobalConfig


class Game:
	def __init__(self, global_config: GlobalConfig, config: Config, episode: int):
		self.global_config: GlobalConfig = global_config
		self.config: Config = config
		self.episode: int = episode

		self.board: Board = Board(global_config.board_size) if not config.tournament_mode else Board(
			global_config.board_size, change_board_after_n_plays=4)  # create board
		self.ply = 0  # no plies so far
		self.agent: Agent = self.config.black  # black always begins
		self.prev_pass: bool = False  # opponent did not pass in previous ply
		self.done: bool = False  # not done yet

	def play(self) -> None:
		if self.config.verbose_live:
			print(f'Episode {self.episode}:')
			print(f'\tPly {self.ply}: INIT')
			print(self.board)

		# play until done
		while not self.done:
			# update
			self.ply += 1

			if self.config.verbose_live:
				disk_icon: str = u'\u25CF' if self.agent.color == Color.BLACK else u'\u25CB'
				print(f'\tPly {self.ply}: {self.agent.color.name} {disk_icon}')

			# get legal actions
			legal_actions: dict = self.board.get_legal_actions(self.agent.color.value)

			if not legal_actions:
				# pass if no legal actions
				if self.config.verbose_live:
					print(f'\tNo legal actions')
					print(self.board)
					print(f'\tNext action: PASS')
				if self.prev_pass:
					self.done = True  # no agent has legal actions, deadlock
				self.prev_pass = True  # this agent has no legal actions, pass
			else:
				# get next action from legal actions and take it
				location, legal_directions = self.agent.get_next_action(self.board, legal_actions)
				if self.config.verbose_live:
					print(f'\tLegal actions: {list(legal_actions.keys())}')
					board_copy: Board = copy.deepcopy(self.board)
					for legal_location in legal_actions:
						board_copy.board[legal_location] = Color.LEGAL.value
					print(board_copy)
					print(f'\tNext action: {location}')
				self.prev_pass = False  # this agent has legal actions, no pass

				prev_board = np.copy(self.board.board) if isinstance(self.agent,
				                                                     TrainableAgent) and self.agent.train_mode else None
				self.done = self.board.take_action(location, legal_directions, self.agent.color.value)
				if self.config.verbose_live:
					print(self.board)

				# get immediate reward if agent makes use of it
				if self.agent.immediate_reward:
					immediate_reward: float = self.agent.immediate_reward.immediate_reward(self.board,
					                                                                       self.agent.color.value)
					if self.config.verbose_live:
						print(f'Immediate reward: {immediate_reward}')
					# remember the board, the taken action and the resulting reward
					if isinstance(self.agent, TrainableAgent):
						self.agent.replay_buffer.add(prev_board, location, immediate_reward, False)

			if self.config.verbose_live:
				print(self.board)

			if not self.done:
				# change turns
				self.agent = self.config.black if self.agent == self.config.white else self.config.white
			else:
				# the game is done
				# update scores of both agents
				self.config.black.update_final_score(self.board)
				self.config.white.update_final_score(self.board)

				# train the agents on the made moves
				for agent in [self.config.black, self.config.white]:
					if isinstance(agent, TrainableAgent) and agent.train_mode:
						# use a final reward for winning/losing
						final_reward = agent.immediate_reward.final_reward(self.board, agent.color)
						# change reward in last buffer entry
						agent.replay_buffer.add_final_reward(final_reward)
						# learn from the game
						agent.train()
						# clear the buffer
						agent.replay_buffer.clear()

				# print end result
				if self.config.verbose_live:
					if self.board.num_black_disks > self.board.num_white_disks:
						print(colored(
							f'{self.episode:>5}: BLACK ({self.board.num_black_disks:>3}|{self.board.num_white_disks:>3}|{self.board.num_free_spots:>3})',
							'red'))
					elif self.board.num_black_disks < self.board.num_white_disks:
						print(colored(
							f'{self.episode:>5}: WHITE ({self.board.num_black_disks:>3}|{self.board.num_white_disks:>3}|{self.board.num_free_spots:>3})',
							'green'))
					else:
						print(colored(
							f'{self.episode:>5}: DRAW  ({self.board.num_black_disks:>3}|{self.board.num_white_disks:>3}|{self.board.num_free_spots:>3})',
							'cyan'))
