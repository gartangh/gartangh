from agents.agent import Agent
from game_logic.board import Board
from policies.untrainable_policy import UntrainablePolicy
from utils.color import Color
from utils.types import Actions, Action


class UntrainableAgent(Agent):
	def __init__(self, color: Color, policy: UntrainablePolicy) -> None:
		super().__init__(color)

		self.policy: UntrainablePolicy = policy

	def __str__(self) -> str:
		return f'Untrainable{super().__str__()}, policy={self.policy})'

	def next_action(self, board: Board, legal_actions: Actions) -> Action:
		action: Action = self.policy.get_action(board, legal_actions, self.color)

		return action
