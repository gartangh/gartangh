class GlobalConfig:
    def __init__(self, board_size: int = 8, gui_size: int = 400) -> None:
        # check parameters
        assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
        assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'
        assert 100 <= gui_size <= 1000, f'Invalid gui size: gui_size should be between 100 and 1000, but got {gui_size}'

        self.board_size: int = board_size
        self.gui_size: int = gui_size