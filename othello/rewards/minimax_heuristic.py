from game_logic.board import Board
from rewards.reward import Reward
from rewards.risk_regions_reward import RiskRegionsReward
from utils.color import Color


class MinimaxHeuristic(Reward):
	def __init__(self, board_size: int) -> None:
		self.inner_reward = RiskRegionsReward(board_size)

	def reward(self, board: Board, color: Color) -> float:
		score: float = self.inner_reward.evaluate_board(board.board, color)
		prev_score: float = self.inner_reward.evaluate_board(board.prev_board, color)
		reward: float = score - prev_score

		return reward
