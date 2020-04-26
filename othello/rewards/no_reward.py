from game_logic.board import Board
from rewards.reward import Reward
from utils.color import Color


class NoReward(Reward):
	def reward(self, board: Board, color: Color) -> float:
		reward: float = 0.0

		return reward
