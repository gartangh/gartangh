import random

import numpy as np
from typing import List, Tuple

from game_logic.agents.agent import Agent
from game_logic.board import Board
from utils.color import Color
from utils.risk_regions import risk_regions


class RiskRegionsAgent(Agent):
	def __init__(self, color: Color, board_size: int = 8):
		super().__init__(color)

		self._weights: np.array = risk_regions(board_size)
		self.name: str = 'RiskRegions'

	def __str__(self):
		return f'{self.name}{super().__str__()}'

	def get_next_action(self, board: Board, legal_actions: dict) -> tuple:
		best_locations: List[Tuple[int, int]] = []
		best_score: None = None

		for location in legal_actions:
			score: int = self._weights[location]
			if best_score is None or score > best_score:
				best_score: int = score
				best_locations: List[Tuple[int, int]] = [location]
			elif score == best_score:
				best_locations.append(location)

		location: Tuple[int, int] = random.choice(list(legal_actions.keys()))
		legal_directions: List[Tuple[int, int]] = legal_actions[location]

		return location, legal_directions
