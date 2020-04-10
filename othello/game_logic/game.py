import copy

from termcolor import colored

from game_logic.agents.agent import Agent
from game_logic.agents.trainable_agent import TrainableAgent
from game_logic.board import Board
from utils.color import Color


class Game:
	def __init__(self, episode: int, black: Agent, white: Agent, board_size: int, verbose: bool, tournament_mode: bool = False):
		# check arguments
		assert black.color is Color.BLACK, f'Invalid black agent'
		assert white.color is Color.WHITE, f'Invalid white agent'

		self.episode: int = episode
		self.black: Agent = black
		self.white: Agent = white
		self.board: Board = Board(board_size) if not tournament_mode else Board(board_size, change_board_after_n_plays=1) # create a new board
		self.verbose: bool = verbose

		# state of the game
		self.ply = 0  # no plies so far
		self.agent: Agent = black  # black begins
		self.prev_pass: bool = False  # opponent did not pass in previous ply
		self.done: bool = False  # not done yet

	def play(self) -> None:
		if self.verbose:
			print(f'Episode: {self.episode}')

		if self.verbose:
			print(f'Ply: {self.ply} (INIT)')
			print(self.board)

		# play until done
		while not self.done:
			# update
			self.ply += 1
			if self.verbose:
				print(f'Ply: {self.ply} ({self.agent.color.name})')

			# get legal actions
			legal_actions: dict = self.board.get_legal_actions(self.agent.color.value)

			if not legal_actions:
				# pass if no legal actions
				if self.verbose:
					print(f'No legal actions')
					print(f'Next action: PASS')
				if self.prev_pass:
					self.done = True  # no agent has legal actions, deadlock
				self.prev_pass = True  # this agent has no legal actions, pass
			else:
				# get next action from legal actions and take it
				location, legal_directions = self.agent.get_next_action(self.board, legal_actions)
				if self.verbose:
					print(f'Legal actions: {list(legal_actions.keys())}')
					print(f'Next action: {location}')
				self.prev_pass = False  # this agent has legal actions, no pass

				prev_board = copy.deepcopy(self.board) if isinstance(self.agent,
				                                                     TrainableAgent) and self.agent.train_mode else None
				self.done = self.board.take_action(location, legal_directions, self.agent.color.value)

				# get immediate reward if agent makes use of it
				if self.agent.immediate_reward:
					immediate_reward: float = self.agent.immediate_reward.immediate_reward(self.board,
					                                                                       self.agent.color.value)
					if self.verbose:
						print(f'Immediate reward: {immediate_reward}')
					if isinstance(self.agent, TrainableAgent) and self.agent.train_mode:
						# if the agent is ready, let it learn from the replay buffer
						self.agent.train(prev_board, location, immediate_reward, self.board, self.done, render=False)

			if self.verbose:
				print(self.board)

			if not self.done:
				# the game is not done yet
				# change turns
				self.agent = self.black if self.agent == self.white else self.white
			else:
				# the game is done
				# update scores of both agents
				self.black.update_final_score(self.board)
				self.white.update_final_score(self.board)

				# print end result
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
