from agents.agent import Agent
from game_logic.board import Board
from policies.risk_regions_policy import RiskRegionsPolicy
from utils.color import Color
from utils.types import Actions, Action


class RiskRegionsAgent(Agent):
	def __init__(self, color: Color, board_size: int) -> None:
		super().__init__(color)

		self.policy: RiskRegionsPolicy = RiskRegionsPolicy(board_size)

	def __str__(self) -> str:
		return f'RiskRegions{super().__str__()}'

	def get_next_action(self, board: Board, legal_actions: Actions) -> Action:
		action: Action = self.policy.get_action(legal_actions)

		return action
