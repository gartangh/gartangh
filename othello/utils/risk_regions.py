import numpy as np

_edge: np.array = np.array([5, 2, 1])
_corner: np.array = np.array(
	[[100, -25, 10],
	 [-25, -25, 2],
	 [10, 2, 5]])


def risk_regions(board_size: int = 8):
	# check arguments
	assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
	assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'

	# create weights
	weights: np.array = _corner
	half_board_size: int = board_size // 2
	if half_board_size <= 3:
		weights: np.array = _corner[:half_board_size, :half_board_size]
	else:
		weights: np.array = np.vstack([weights, _edge])
		weights: np.array = np.column_stack([weights, np.append(_edge, 1)])
		weights: np.array = np.pad(weights, (0, half_board_size - 4), 'edge')
		weights[half_board_size - 1, half_board_size - 1] = 2

	weights: np.array = np.pad(weights, (0, half_board_size), 'symmetric')

	return weights