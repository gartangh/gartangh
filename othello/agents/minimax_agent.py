from typing import List, Tuple, Dict

from agents.agent import Agent
from game_logic.board import Board
from policies.minimax_policy import MinimaxPolicy
from rewards.reward import Reward
from utils.color import Color
from utils.types import Action, Actions


class MinimaxAgent(Agent):
	def __init__(self, color: Color, immediate_reward: Reward, depth: int = 2) -> None:
		super().__init__(color)

		self.policy = MinimaxPolicy(immediate_reward, depth)
		self.max_depth: int = depth

	def __str__(self) -> str:
		return f'Minimax{super().__str__()}'

	def get_next_action(self, board: Board, legal_actions: Actions) -> Action:
		action: Action = self.policy.get_action(board, legal_actions, self.color)

		return action
