import numpy as np

from utils.color import Color


def flatten(board: np.array) -> np.array:
	"""e.g. (8,8) -> (64)"""
	return board.flatten()


def split(board: np.array, color: Color) -> np.array:
	"""e.g. (8,8) -> (2,8,8)"""
	own: np.array = np.where(board == color.value, 1, 0)
	opponent: np.array = np.where(board == 1 - color.value, 1, 0)
	return np.stack([own, opponent])


def flatten_split(board: np.array) -> np.array:
	"""e.g. (8,8) -> (2,64)"""
	return split(flatten(board))


def split_flatten(board: np.array) -> np.array:
	"""e.g. (8,8) -> (128)"""
	return flatten(split(board))


def flatten_negative(board: np.array, color: Color) -> np.array:
	"""e.g. (8,8) -> (64) with -1, 0, 1"""
	own: np.array = np.where(board == color.value, 1, 0)
	opponent: np.array = np.where(board == 1 - color.value, -1, 0)
	board: np.array = np.add(own, opponent)
	return flatten(board)
