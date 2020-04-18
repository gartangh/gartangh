from typing import List, Tuple, Dict

from game_logic.agents.agent import Agent
from game_logic.board import Board
from utils.color import Color
from utils.immediate_rewards.immediate_reward import ImmediateReward


class MinimaxAgent(Agent):
	def __init__(self, color: Color, immediate_reward: ImmediateReward, depth: int = 2):
		super().__init__(color, immediate_reward)
		self.max_depth: int = depth

	def __str__(self):
		return f'Minimax{super().__str__()}'

	@staticmethod
	def _finished(board: Board) -> tuple:
		black_legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]] = board.get_legal_actions(Color.BLACK.value)
		white_legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]] = board.get_legal_actions(Color.WHITE.value)

		ended: bool = False
		won: None = None
		if not white_legal_actions and not black_legal_actions:
			ended: bool = True  # someone won
			if board.num_black_disks > board.num_white_disks:
				won: int = Color.BLACK.value
			elif board.num_white_disks > board.num_black_disks:
				won: int = Color.WHITE.value
			else:
				won: int = Color.EMPTY.value

		return ended, won

	def minimax(self, board: Board, color_value: int, legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]],
	            level: int = 0,
	            prev_best_points: float = None) -> Tuple[float, Tuple[int, int]]:
		cur_best_score: None = None
		cur_best_location: None = None
		opponent_color_value: int = 1 - color_value

		for location in legal_actions:
			new_board: Board = board.get_deepcopy()
			new_board.take_action(location, legal_actions[location], color_value)
			if level < self.max_depth:
				new_legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]] = new_board.get_legal_actions(
					opponent_color_value)
				if not new_legal_actions:  # opponent passes -> player plays again
					new_legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]] = new_board.get_legal_actions(
						color_value)
					points, _ = self.minimax(new_board, color_value, new_legal_actions, level + 1, cur_best_score)
				else:  # opponent plays next ply
					points, _ = self.minimax(new_board, opponent_color_value, new_legal_actions, level + 1,
					                         cur_best_score)
			else:
				points: float = self.immediate_reward.immediate_reward(new_board, color_value)

			# when points is not assigned -> due to nobody can play anymore
			if points is None:
				ended, won = self._finished(new_board)
				if ended:
					if won == self.color.value:
						points: float = 1000.0
					elif won == 1 - self.color.value:
						points: float = -1000.0
					else:
						points: float = 0.0

			if color_value == self.color.value:  # max_step
				if cur_best_score is None or cur_best_score < points:
					cur_best_score: float = points
					cur_best_location: Tuple[int, int] = location
				if prev_best_points is not None and cur_best_score > prev_best_points:
					break
			elif opponent_color_value == self.color.value:  # min step
				if cur_best_score is None or cur_best_score > points:
					cur_best_score: float = points
					cur_best_location: Tuple[int, int] = location
				if prev_best_points is not None and cur_best_score < prev_best_points:
					break

		return cur_best_score, cur_best_location

	def get_next_action(self, board: Board, legal_actions: dict) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
		_, location = self.minimax(board, self.color.value, legal_actions)
		legal_directions: List[Tuple[int, int]] = legal_actions[location]

		return location, legal_directions
