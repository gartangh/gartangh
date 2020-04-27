import tkinter as tk

import numpy as np

from agents.agent import Agent
from game_logic.board import Board
from utils.color import Color
from utils.types import Directions, Location, Action


class HumanAgent(Agent):
	def __init__(self, color: Color) -> None:
		assert color is Color.WHITE, f'Invalid color for HumanAgent: color must be WHITE, but got {color.name}'

		super().__init__(color)

		self.first_move = True

	def __str__(self) -> str:
		return f'Human{super().__str__()})'

	def __update_board(self, board: Board, legal_directions: dict) -> None:
		disks: np.array = board.board
		board_size: int = board.board_size
		if self.first_move:
			for widget in self.frame.winfo_children():
				widget.destroy()
			self.first_move = False
		for i in range(board_size):
			for j in range(board_size):
				val = disks[i, j]
				if val == 0:
					label = tk.Label(self.frame, bg='black')
					label.place(relx=j / board_size, rely=i / board_size, relwidth=1 / board_size,
					            relheight=1 / board_size)
				elif val == 1:
					label = tk.Label(self.frame, bg='white')
					label.place(relx=j / board_size, rely=i / board_size, relwidth=1 / board_size,
					            relheight=1 / board_size)
				elif (i, j) in list(legal_directions.keys()):
					btn = tk.Button(self.frame, command=lambda: self.__click(j, i))
					btn.place(relx=j / board_size, rely=i / board_size, relwidth=1 / board_size,
					          relheight=1 / board_size)
				else:
					continue
		self.root.update_idletasks()
		self.root.update()

	def next_action(self, board: Board, legal_directions: dict) -> Action:
		self.__update_board(board, legal_directions)
		while (not self.clicked):
			continue
		self.clicked = False
		location: Location = (self.selected_col, self.selected_row)
		legal_directions: Directions = legal_directions[location]
		return location, legal_directions
