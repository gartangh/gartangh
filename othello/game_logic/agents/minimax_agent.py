import copy

from game_logic.agents.agent import Agent
from game_logic.board import Board
from utils.color import Color


class MinimaxAgent(Agent):
	def __init__(self, color: Color, immediate_reward, max_depth: int = 3):
		super().__init__(color, immediate_reward)
		self.name: str = "Minimax"
		self.max_depth: int = max_depth

	def __str__(self):
		return f'{self.name}{super().__str__()}'

	def minimax(self, board: Board, color_value: int, legal_actions: dict, level: int = 0,
	            prev_best_points: float = None) -> tuple:
		cur_best_score = None
		cur_best_location = None
		opponent_color_value: int = 1 - color_value

		for location in legal_actions:
			new_board: Board = copy.deepcopy(board)
			new_board.take_action(location, legal_actions[location], color_value)
			if level < self.max_depth:
				new_legal_actions: dict = new_board.get_legal_actions(opponent_color_value)
				if not new_legal_actions:  # opponent passes -> player plays again
					new_legal_actions: dict = new_board.get_legal_actions(color_value)
					points, _ = self.minimax(new_board, color_value, new_legal_actions, level + 1, cur_best_score)
				else:  # opponent plays next ply
					points, _ = self.minimax(new_board, opponent_color_value, new_legal_actions, level + 1,
					                         cur_best_score)
			else:
				points = self.immediate_reward.immediate_reward(new_board, color_value)

			if color_value == self.color.value:  # max_step
				if cur_best_score is None or cur_best_score < points:
					cur_best_score = points
					cur_best_location = location
				if prev_best_points is not None and cur_best_score > prev_best_points:
					break
			elif opponent_color_value == self.color.value:  # min step
				if cur_best_score is None or cur_best_score > points:
					cur_best_score = points
					cur_best_location = location
				if prev_best_points is not None and cur_best_score < prev_best_points:
					break

		return cur_best_score, cur_best_location

	def get_next_action(self, board: Board, legal_actions: dict) -> tuple:
		_, location = self.minimax(board, self.color.value, legal_actions)
		legal_directions: list = legal_actions[location]

		return location, legal_directions
