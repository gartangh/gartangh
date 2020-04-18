from game_logic.board import Board
from utils.color import Color


class ImmediateReward:
	def immediate_reward(self, board: Board, color_value: int) -> float:
		raise NotImplementedError

	def final_reward(self, board: Board, color: Color) -> float:
		raise NotImplementedError
