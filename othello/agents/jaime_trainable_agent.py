import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from agents.trainable_agent import TrainableAgent


class JaimeTrainableAgent(TrainableAgent):
	def __str__(self) -> str:
		return f'Jaime{super().__str__()}'

	def create_model(self, verbose: bool = False, lr: float = 0.00025) -> Sequential:
		# input: 2 nodes per board location:
		#              - 1 node that is 0 if location does not contain black, else 1
		#              - 1 node that is 0 if location does not contain white, else 1
		# output: 1 node per board location, with probabilities to take action on that location
		model: Sequential = Sequential()
		model.add(Dense(self.board_size ** 2, input_shape=(self.board_size ** 2,), activation='relu'))
		model.add(Dense(self.board_size ** 2, activation='softmax'))
		model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))

		return model

	def board_to_nn_input(self, board: np.array) -> np.array:
		# Transform board (numpy ndarray) to a numpy 1D array
		# that can serve as input to the neural network.
		# There is 1 node per board location: 0 = empty, 1 = own, -1 = opponent
		own_color = self.color.value
		oponent_color = 1 - own_color
		board_array = np.zeros((1, self.board_size ** 2))
		for row in range(self.board_size):
			for col in range(self.board_size):
				if board[row][col] == own_color:
					board_array[0][row * self.board_size + col] = 1
				elif board[row][col] == oponent_color:
					board_array[0][row * self.board_size + col] = -1

		return board_array
