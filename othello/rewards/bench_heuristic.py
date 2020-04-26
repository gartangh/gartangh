from game_logic.board import Board
from rewards.bench_reward import BenchReward
from rewards.reward import Reward
from utils.color import Color


class BenchHeuristic(Reward):
	def __init__(self, board_size: int = 8) -> None:
		self.inner_reward = BenchReward(board_size)

	def reward(self, board: Board, color: Color) -> float:
		score: float = self.inner_reward.evaluate_board(board.board, color)
		prev_score: float = self.inner_reward.evaluate_board(board.prev_board, color)
		reward: float = score - prev_score

		return reward
