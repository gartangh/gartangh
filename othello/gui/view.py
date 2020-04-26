import tkinter

from gui.model import Score
from utils.color import Color


class View:
	def __init__(self) -> None:
		self.controller = None
		self.game_width = 400
		self.game_height = 400
		self._rows = 8
		self._cols = 8

	def configure_view(self) -> None:
		self._root_window = tkinter.Tk()
		self._root_window.title('Othello')
		self._root_window.configure(background='green')

		self._board = tkinter.Canvas(master=self._root_window, width=self.game_width,
		                             height=self.game_height, background='green')
		self._board.bind('<Configure>', self._resize_board)
		self._board.bind('<Button-1>', self.controller.on_board_clicked)

		self._black_score = Score(Color.BLACK, self.controller.game.board, self._root_window)
		self._white_score = Score(Color.WHITE, self.controller.game.board, self._root_window)

		self._black_score.get_score_label().grid(row=0, column=0,
		                                         sticky=tkinter.S)
		self._white_score.get_score_label().grid(row=0, column=1,
		                                         sticky=tkinter.S)
		self._board.grid(row=1, column=0, columnspan=2, padx=50, pady=10,
		                 sticky=tkinter.N + tkinter.E + tkinter.S + tkinter.W)

		self._root_window.rowconfigure(0, weight=1)
		self._root_window.rowconfigure(1, weight=1)
		self._root_window.columnconfigure(0, weight=1)
		self._root_window.columnconfigure(1, weight=1)

	def startup_gui(self) -> None:
		self._root_window.mainloop()

	def update_move(self) -> None:
		self._redraw_board()
		self._black_score.update_score()
		self._white_score.update_score()

	def _redraw_board(self) -> None:
		self._board.delete(tkinter.ALL)
		self._redraw_lines()
		self._redraw_cells()

	def _resize_board(self, event: tkinter.Event) -> None:
		self._redraw_board()

	def _redraw_lines(self) -> None:
		row_multiplier = float(self._board.winfo_height()) / self._rows
		col_multiplier = float(self._board.winfo_width()) / self._cols

		# Draw the horizontal lines first
		for row in range(1, self._rows):
			self._board.create_line(0, row * row_multiplier, self.get_board_width(), row * row_multiplier)

		# Draw the column lines next
		for col in range(1, self._cols):
			self._board.create_line(col * col_multiplier, 0, col * col_multiplier, self.get_board_height())

	def _redraw_cells(self) -> None:
		for row in range(self._rows):
			for col in range(self._cols):
				if self.controller.game.board.board[row][col] != Color.EMPTY.value:
					self._draw_cell(row, col)
				elif (row, col) in list(self.controller.legal_actions):
					self._draw_legal_location(row, col)

	def _draw_cell(self, row: int, col: int) -> None:
		self._board.create_oval(col * self.get_cell_width(),
		                        row * self.get_cell_height(),
		                        (col + 1) * self.get_cell_width(),
		                        (row + 1) * self.get_cell_height(),
		                        fill=Color.BLACK.name.lower() if self._is_black(row, col) else Color.WHITE.name.lower())

	def _draw_legal_location(self, row: int, col: int) -> None:
		self._board.create_oval(col * self.get_cell_width(),
		                        row * self.get_cell_height(),
		                        (col + 1) * self.get_cell_width(),
		                        (row + 1) * self.get_cell_height(),
		                        fill='grey',
		                        activefill='white')

	def get_cell_width(self) -> float:
		return self.get_board_width() / self.get_columns()

	def get_cell_height(self) -> float:
		return self.get_board_height() / self.get_rows()

	def get_board_width(self) -> float:
		return float(self._board.winfo_width())

	def get_board_height(self) -> float:
		return float(self._board.winfo_height())

	def get_rows(self) -> int:
		return self._rows

	def get_columns(self) -> int:
		return self._cols

	def get_board(self) -> tkinter.Canvas:
		return self._board

	def close(self) -> None:
		self._root_window.destroy()

	def _is_black(self, row, col) -> bool:
		return self.controller.game.board.board[row][col] == Color.BLACK.value
