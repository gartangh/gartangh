import numpy as np

from utils.color import Color


def flatten(board: np.array) -> np.array:
	"""e.g. (8,8) -> (64)"""
	return board.flatten()


def split(board: np.array) -> np.array:
	"""e.g. (8,8) -> (2,8,8)"""
	blacks = np.where(board == Color.BLACK.value, 1, 0)
	whites = np.where(board == Color.WHITE.value, 1, 0)
	return np.stack([blacks, whites])


def flatten_split(board: np.array) -> np.array:
	"""e.g. (8,8) -> (2,64)"""
	return split(flatten(board))


def split_flatten(board: np.array) -> np.array:
	"""e.g. (8,8) -> (128)"""
	return flatten(split(board))
