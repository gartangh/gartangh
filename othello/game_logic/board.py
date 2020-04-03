import numpy as np
import copy

from utils.color import Color


class Board:
	# initialize static variables
	directions: list = [[+1, +0],  # down
	                    [+1, +1],  # down right
	                    [+0, +1],  # right
	                    [-1, +1],  # up right
	                    [-1, +0],  # up
	                    [-1, -1],  # up left
	                    [+0, -1],  # left
	                    [+1, -1]]  # down left

	# public methods
	# constructor
	def __init__(self, board_size: int = 8):
		# check arguments
		assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
		assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'
		self.board_size: int = board_size

		board: np.array = -np.ones([board_size, board_size], dtype=int)
		board[board_size // 2 - 1, board_size // 2 - 1] = 1  # white
		board[board_size // 2, board_size // 2 - 1] = 0  # black
		board[board_size // 2 - 1, board_size // 2] = 0  # black
		board[board_size // 2, board_size // 2] = 1  # white
		self.board: np.array = board
		self.prev_board: np.array = copy.deepcopy(board)
		self.num_black_disks: int = 2
		self.num_white_disks: int = 2
		self.num_free_spots: int = board_size ** 2 - 4

	def __str__(self):
		string = '\t|'
		for j in range(self.board_size):
			string += f'{j}\t'
		string += '\n_\t|'
		for j in range(self.board_size):
			string += '_\t'
		string += '\n'
		for i, row in enumerate(self.board):
			string += f'{i}\t|'
			for val in row:
				if val == -1:
					string += ' \t'
				elif val == 0:
					string += 'B\t'
				elif val == 1:
					string += 'W\t'
				else:
					raise Exception(f'Incorrect value on board: expected 1, -1 or 0, but got {val}')
			string += '\n'
		return string

	def get_legal_actions(self, color_value: int) -> dict:
		legal_actions: dict = {}
		for i in range(self.board_size):
			for j in range(self.board_size):
				legal_directions: list = self._get_legal_directions((i, j), color_value)
				if len(legal_directions) > 0:
					legal_actions[(i, j)]: list = legal_directions

		# pass if no legal action
		if len(list(legal_actions.keys())) == 0:
			legal_actions['pass']: None = None

		return legal_actions

	def take_action(self, location: tuple, legal_directions: list, color_value: int) -> bool:
		if location == 'pass':
			return False

		# check if location does point to an empty spot
		assert self.board.item(location) == -1, f'Invalid location: location ({location}) does not point to an empty spot on the board)'

		# save state before action
		self.prev_board: np.array = copy.deepcopy(self.board)

		# put down own disk in the provided location
		i: int = location[0]
		j: int = location[1]
		self.board[i, j]: int = color_value

		# turn around opponent's disks
		for direction in legal_directions:
			i: int = location[0] + direction[0]
			j: int = location[1] + direction[1]
			while 0 <= i < self.board_size and 0 <= j < self.board_size:
				disk = self.board[i, j]
				if disk == color_value or disk == Color.EMPTY.value:
					break  # encountered empty spot or own disk
				if disk == 1 - color_value:
					self.board[i, j] = color_value  # encountered opponent's disk
				i += direction[0]
				j += direction[1]

		# update scores
		self._update_score()

		# calculate immediate reward, not in take_action()
		# TODO: put this in a Game
		# immediate_reward: float = player.get_immediate_reward(self)

		# check if othello is finished
		done = self._check_game_finished()

		# TODO: put this outside of take_action()
		# TODO: also call update for opponent
		# if done:
		# update final score
		# player.update_final_score(self)

		return done

	# private methods
	def _get_legal_directions(self, location: tuple, color_value: int) -> list:
		legal_directions: list = []

		# check if location points to an empty spot
		if self.board.item(location) != -1:
			return legal_directions

		# search in all directions
		for direction in Board.directions:
			found_opponent: bool = False  # check wetter there is an opponent's disk
			i: int = location[0] + direction[0]
			j: int = location[1] + direction[1]
			while 0 <= i < self.board_size and 0 <= j < self.board_size:
				# while not out of the board, keep going
				disk = self.board[i, j]
				if disk == -1:
					break  # found empty spot

				if disk == 1 - color_value:
					found_opponent: bool = True  # found opponent's disk

				if disk == color_value and found_opponent:
					legal_directions.append(direction)  # found own disk after finding opponent's disk
					break

				i += direction[0]
				j += direction[1]

		return legal_directions

	def _update_score(self):
		# get scores
		num_black_disks: int = len(np.where(self.board == Color.BLACK.value)[0])
		num_white_disks: int = len(np.where(self.board == Color.WHITE.value)[0])
		num_free_spots: int = len(np.where(self.board == Color.EMPTY.value)[0])
		num_disks: int = num_black_disks + num_white_disks

		# check scores
		assert 0 <= num_black_disks <= self.board_size ** 2, f'Invalid number of black disks: num_black_disks should be between 0 and {self.board_size ** 2}, but got {num_black_disks}'
		assert 0 <= num_white_disks <= self.board_size ** 2, f'Invalid number of white disks: num_white_disks should be between 0 and {self.board_size ** 2}, but got {num_white_disks}'
		assert 0 <= num_free_spots <= self.board_size ** 2 - 4, f'Invalid number of free spots: num_free_spots should be between 0 and {self.board_size ** 2 - 4}, but got {num_free_spots}'
		assert num_disks + num_free_spots == self.board_size ** 2, f'Invalid number of disks and free spots: sum of disks and num_free_spots should be {self.board_size ** 2}, but got {num_disks + num_free_spots}'

		self.num_black_disks: int = num_black_disks
		self.num_white_disks: int = num_white_disks
		self.num_free_spots: int = num_free_spots

	def _check_game_finished(self):
		# return finished or not
		if self.num_black_disks == 0 or self.num_white_disks == 0 or self.num_free_spots == 0:
			return True
		else:
			return False