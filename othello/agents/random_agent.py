from agents.agent import Agent
from game_logic.board import Board
from policies.random_policy import RandomPolicy
from utils.color import Color
from utils.types import Actions, Action


class RandomAgent(Agent):
	def __init__(self, color: Color) -> None:
		super().__init__(color)

		self.policy: RandomPolicy = RandomPolicy()

	def __str__(self) -> str:
		return f'Random{super().__str__()}'

	def get_next_action(self, board: Board, legal_actions: Actions) -> Action:
		action: Action = self.policy.get_action(legal_actions)

		return action
