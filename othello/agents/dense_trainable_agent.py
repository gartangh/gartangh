import numpy as np
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from agents.trainable_agent import TrainableAgent
from utils.reshapes import split_flatten, flatten_negative


class DenseTrainableAgent(TrainableAgent):
	def __str__(self) -> str:
		return f'Dense{super().__str__()})'

	def create_model(self, verbose: bool = False, lr: float = 0.001) -> Sequential:
		# input: 2 nodes per board location:
		#              - 1 node that is 0 if location does not contain black, else 1
		#              - 1 node that is 0 if location does not contain white, else 1
		# output: 1 node per board location, with probabilities to take action on that location
		model: Sequential = Sequential([
			Input(shape=(self.board_size ** 2)),
			Dense(self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(self.board_size ** 2, activation='softmax'),
		])
		model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))

		return model

	def board_to_nn_input(self, board: np.array) -> np.array:
		return flatten_negative(board, self.color)
