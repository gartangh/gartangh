import numpy as np
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from game_logic.agents.dqn_trainable_agent import DQNTrainableAgent
from utils.reshapes import split_flatten


class DenseDQNTrainableAgent(DQNTrainableAgent):
	def __str__(self):
		return f'Dense{super().__str__()}'

	def create_model(self, verbose: bool = False, lr: float = 0.00025) -> Sequential:
		# input: 2 nodes per board location:
		#              - 1 node that is 0 if location does not contain black, else 1
		#              - 1 node that is 0 if location does not contain white, else 1
		# output: 1 node per board location, with probabilities to take action on that location
		model: Sequential = Sequential([
			Input(shape=(2 * self.board_size ** 2)),
			Dense(2 * self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(2 * self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(2 * self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(2 * self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(2 * self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(2 * self.board_size ** 2, activation='relu', kernel_initializer='he_uniform'),
			Dense(self.board_size ** 2, activation='softmax'),
		])
		model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))

		return model

	def board_to_nn_input(self, board: np.ndarray) -> np.array:
		return split_flatten(board)
