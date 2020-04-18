from game_logic.board import Board
from utils.color import Color
from utils.immediate_rewards.immediate_reward import ImmediateReward


class Agent:
	def __init__(self, color: Color, immediate_reward: ImmediateReward = None):
		self.color: Color = color
		self.immediate_reward: ImmediateReward = immediate_reward
		self.num_games_won: int = 0

	def __str__(self):
		return f'Agent: {self.color.name}'

	def get_next_action(self, board: Board, legal_actions: dict) -> tuple:
		raise NotImplementedError

	def update_final_score(self, board: Board) -> None:
		if self.color is Color.BLACK and board.num_black_disks > board.num_white_disks:
			self.num_games_won += 1  # BLACK won
		elif self.color is Color.WHITE and board.num_white_disks > board.num_black_disks:
			self.num_games_won += 1  # WHITE won
