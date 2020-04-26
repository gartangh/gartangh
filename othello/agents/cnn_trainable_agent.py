import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from agents.trainable_agent import TrainableAgent
from utils.reshapes import split

try:
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	tf.config.optimizer.set_jit(True)  # XLA enabled
except:
	pass


class CNNTrainableAgent(TrainableAgent):
	def __str__(self) -> str:
		return f'CNN{super().__str__()}'

	def create_model(self, verbose: bool = False, lr: float = 0.00025) -> Sequential:
		model: Sequential = Sequential([
			Input(shape=(2, self.board_size, self.board_size)),
			Conv2D(16, (3, 3), padding='same', data_format='channels_first', activation='relu',
			       kernel_initializer='he_uniform'),
			Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu',
			       kernel_initializer='he_uniform'),
			Conv2D(256, (3, 3), padding='same', data_format='channels_first', activation='relu',
			       kernel_initializer='he_uniform'),
			GlobalAveragePooling2D(data_format='channels_first'),
			Flatten(data_format='channels_first'),
			Dense(self.board_size ** 2, activation='softmax'),
		])
		model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr))

		if verbose:
			model.summary()

		return model

	def board_to_nn_input(self, board: np.array) -> np.array:
		return split(board)
