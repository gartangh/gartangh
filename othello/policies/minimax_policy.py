from typing import Tuple, Union

from game_logic.board import Board
from policies.policy import Policy
from rewards.reward import Reward
from utils.color import Color
from utils.types import Actions, Location, Directions, Action


class MinimaxPolicy(Policy):
	def __init__(self, immediate_reward: Reward, depth: int) -> None:
		assert 2 <= depth <= 3, f'Invalid depth: depth should be between 2 and 3, but got {depth}'

		self.immediate_reward: Reward = immediate_reward
		self.depth: int = depth

	def __str__(self) -> str:
		return f'Minimax{super().__str__()}'

	@staticmethod
	def _finished(board: Board) -> Tuple[bool, Color]:
		black_legal_actions: Actions = board.get_legal_actions(Color.BLACK)
		white_legal_actions: Actions = board.get_legal_actions(Color.WHITE)

		ended: bool = False
		won: Union[Color, None] = None
		if not white_legal_actions and not black_legal_actions:
			ended: bool = True  # someone won
			if board.num_black_disks > board.num_white_disks:
				won: Color = Color.BLACK
			elif board.num_white_disks > board.num_black_disks:
				won: Color = Color.WHITE
			else:
				won: Color = Color.EMPTY

		return ended, won

	def minimax(self, board: Board, legal_actions: Actions, color: Color, level: int = 0,
	            prev_best_points: float = None) -> Tuple[float, Location]:
		cur_best_score: None = None
		cur_best_location: None = None
		opponent_color: Color = Color.WHITE if color is Color.BLACK else Color.BLACK

		for location in legal_actions:
			new_board: Board = board.get_deepcopy()
			new_board.take_action(location, legal_actions[location], color)
			if level < self.depth:
				new_legal_actions: Actions = new_board.get_legal_actions(
					opponent_color)
				if not new_legal_actions:  # opponent passes -> player plays again
					new_legal_actions: Actions = new_board.get_legal_actions(
						color)
					points, _ = self.minimax(new_board, new_legal_actions, color, level + 1, cur_best_score)
				else:  # opponent plays next ply
					points, _ = self.minimax(new_board, new_legal_actions, opponent_color, level + 1, cur_best_score)
			else:
				points: float = self.immediate_reward.reward(new_board, color)

			# when points is not assigned -> due to nobody can play anymore
			if points is None:
				ended, won = self._finished(new_board)
				if ended:
					if won == color.value:
						points: float = 1000.0
					elif won == 1 - color.value:
						points: float = -1000.0
					else:
						points: float = 0.0

			if color.value == color.value:  # max_step
				if cur_best_score is None or cur_best_score < points:
					cur_best_score: float = points
					cur_best_location: Location = location
				if prev_best_points is not None and cur_best_score > prev_best_points:
					break
			elif opponent_color == color.value:  # min step
				if cur_best_score is None or cur_best_score > points:
					cur_best_score: float = points
					cur_best_location: Location = location
				if prev_best_points is not None and cur_best_score < prev_best_points:
					break

		return cur_best_score, cur_best_location

	def get_action(self, board: Board, legal_actions: Actions, color: Color) -> Action:
		_, location = self.minimax(board, legal_actions, color)
		directions: Directions = legal_actions[location]
		action: Action = (location, directions)

		return action
