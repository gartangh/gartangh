import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import tensorflow as tf

if 0:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.optimizer.set_jit(True)  # XLA enabled

from game_logic.agents.dqn_trainable_agent import DQNTrainableAgent
from utils.reshapes import split


class CNNDQNTrainableAgent(DQNTrainableAgent):
	def __str__(self):
		return f'CNN{super().__str__()}'

	def create_model(self, verbose: bool = False, lr: float = 0.00025) -> Sequential:
		model: Sequential = Sequential([
			Input(shape=(2, self.board_size, self.board_size)),
			Conv2D(16, (3, 3), padding='same', data_format='channels_first', activation='relu',
			       kernel_initializer='he_uniform'),
			Conv2D(32, (3, 3), padding='same', data_format='channels_first', activation='relu',
			       kernel_initializer='he_uniform'),
			Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu',
			       kernel_initializer='he_uniform'),
			Conv2D(128, (3, 3), padding='same', data_format='channels_first', activation='relu',
			       kernel_initializer='he_uniform'),
			Conv2D(256, (3, 3), padding='same', data_format='channels_first', activation='relu',
			       kernel_initializer='he_uniform'),
			GlobalAveragePooling2D(data_format='channels_first'),
			Flatten(data_format='channels_first'),
			Dense(256, activation='softmax'),
			Dense(self.board_size ** 2, activation='softmax'),
		])
		model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr))

		if verbose:
			model.summary()

		return model

	def board_to_nn_input(self, board: np.ndarray):
		return split(board)
