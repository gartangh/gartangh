import random
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.random import choice

from utils.color import Color


class Board:
	# initialize static variables
	_directions: List[Tuple[int, int]] = [
		(+1, +0),  # down
		(+1, +1),  # down right
		(+0, +1),  # right
		(-1, +1),  # up right
		(-1, +0),  # up
		(-1, -1),  # up left
		(+0, -1),  # left
		(+1, -1),  # down left
	]

	def __init__(self, board_size: int = 8, random_start: bool = False):
		# check arguments
		assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
		assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'

		self.board_size: int = board_size

		# create board
		board: np.array = -np.ones([board_size, board_size], dtype=int)
		board[board_size // 2 - 1, board_size // 2 - 1] = 1  # white
		board[board_size // 2, board_size // 2 - 1] = 0  # black
		board[board_size // 2 - 1, board_size // 2] = 0  # black
		board[board_size // 2, board_size // 2] = 1  # white
		self.board: np.array = board

		if random_start:
			# 0, 1, or 2 plays (0, 2, or 4 plies)
			num_plays: int = choice(3, 1, p=[0.2, 0.4, 0.4])[0]

		if not random_start or num_plays == 0:
			self.num_black_disks: int = 2
			self.num_white_disks: int = 2

			self.prev_board: Union[np.array, None] = None
		else:
			# adding random start at 2 or 4 steps in future (B - W or B - W - B - W)
			for play in range(num_plays):
				legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]] = self._get_legal_actions(self.board,
				                                                                                      self.board_size,
				                                                                                      Color.BLACK.value)
				location: Tuple[int, int] = random.choice(list(legal_actions.keys()))
				directions: List[Tuple[int, int]] = legal_actions[location]
				self.take_action(location, directions, Color.BLACK.value)

				if play == num_plays - 1:
					self.prev_board: np.array = np.copy(board)

				legal_actions = self._get_legal_actions(self.board, self.board_size, Color.WHITE.value)
				location: Tuple[int, int] = random.choice(list(legal_actions.keys()))
				directions: List[Tuple[int, int]] = legal_actions[location]
				self.take_action(location, directions, Color.WHITE.value)

			self.num_black_disks: int = 2 + num_plays
			self.num_white_disks: int = 2 + num_plays

		self.num_free_spots: int = board_size ** 2 - self.num_black_disks - self.num_white_disks

	def __str__(self):
		string: str = '\t\t\u2502'
		for j in range(self.board_size):
			string += f'{j}\t'
		string += '\n\t\u2500\t\u2502'
		for j in range(self.board_size):
			string += '\u2500\t'
		string += '\n'
		for i, row in enumerate(self.board):
			string += f'\t{i}\t\u2502'
			for val in row:
				if val == Color.LEGAL.value:
					string += u'\u274C\t'
				elif val == Color.EMPTY.value:
					string += ' \t'
				elif val == Color.BLACK.value:
					string += u'\u25CF\t'
				elif val == Color.WHITE.value:
					string += u'\u25CB\t'
				else:
					raise Exception(f'Incorrect value on board: expected 1, -1 or 0, but got {val}')
			string += '\n'
		return string

	def get_deepcopy(self):
		new_board = Board(self.board_size, random_start=False)
		new_board.num_black_disks = self.num_black_disks
		new_board.num_white_disks = self.num_white_disks
		new_board.num_free_spots = self.num_free_spots

		new_board.board = np.copy(self.board)
		new_board.prev_board = np.copy(self.prev_board)

		return new_board

	def get_legal_actions(self, color_value: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
		return self._get_legal_actions(self.board, self.board_size, color_value)

	def take_action(self, location: Tuple[int, int], legal_directions: List[Tuple[int, int]], color_value: int) -> bool:
		# check if location does point to an empty spot
		assert self.board[location[0], location[
			1]] == Color.EMPTY.value, f'Invalid location: location ({location}) does not point to an empty spot on the board)'

		# save state before action
		self.prev_board: np.array = np.copy(self.board)

		# put down own disk in the provided location
		self.board[location[0], location[1]]: int = color_value

		# turn around opponent's disks
		for direction in legal_directions:
			i: int = location[0] + direction[0]
			j: int = location[1] + direction[1]
			while 0 <= i < self.board_size and 0 <= j < self.board_size:
				if self.board[i, j] == Color.EMPTY.value:
					# encountered empty spot
					break

				if self.board[i, j] == color_value:
					# encountered own disk
					break

				if self.board[i, j] == 1 - color_value:
					# encountered opponent's disk
					self.board[i, j] = color_value

				i += direction[0]
				j += direction[1]

		# update scores
		self._update_score()

		# check if othello is finished
		done: bool = self._is_game_finished()

		return done

	def _update_score(self) -> None:
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

	def _is_game_finished(self) -> bool:
		# return whether or not game is finished
		if self.num_free_spots == 0 or self.num_black_disks == 0 or self.num_white_disks == 0:
			return True

		return False

	@staticmethod
	def _get_legal_actions(board: np.array, board_size: int, color_value: int) -> Dict[
		Tuple[int, int], List[Tuple[int, int]]]:
		legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
		for i in range(board_size):
			for j in range(board_size):
				if board[i, j] == Color.EMPTY.value:
					# an empty spot
					legal_directions: List[Tuple[int, int]] = Board._get_legal_directions(board, board_size, (i, j),
					                                                                      color_value)
					if len(legal_directions) > 0:
						legal_actions[(i, j)]: List[Tuple[int, int]] = legal_directions

		return legal_actions

	@staticmethod
	def _get_legal_directions(board: np.array, board_size: int, location: tuple, color_value: int) -> List[
		Tuple[int, int]]:
		legal_directions: List[Tuple[int, int]] = []

		# check if location points to an empty spot
		if board[location[0], location[1]] == Color.EMPTY.value:
			# search in all directions
			for direction in Board._directions:
				found_opponent: bool = False
				i: int = location[0] + direction[0]
				j: int = location[1] + direction[1]
				# while not out of the board, keep going in that direction
				while 0 <= i < board_size and 0 <= j < board_size:
					if board[i, j] == Color.EMPTY.value:
						# found empty spot
						break

					if board[i, j] == color_value and not found_opponent:
						# found a player's disk before finding opponent's disk
						break

					if board[i, j] == color_value and found_opponent:
						# found own disk after finding opponent's disk
						legal_directions.append(direction)
						break

					if board[i, j] == 1 - color_value:
						# found opponent's disk
						found_opponent: bool = True

					i += direction[0]
					j += direction[1]

		return legal_directions
