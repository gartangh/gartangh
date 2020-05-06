import datetime
import os
import pickle
import shutil
from abc import abstractmethod
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential

from agents.agent import Agent
from game_logic.board import Board
from policies.optimal_trainable_policy import OptimalTrainablePolicy
from policies.trainable_policy import TrainablePolicy
from rewards.reward import Reward
from utils.color import Color
from utils.replay_buffer import ReplayBuffer
from utils.types import Action, Actions


class TrainableAgent(Agent):
	def __init__(self, color: Color, train_policy: TrainablePolicy,
	             immediate_reward: Reward, final_reward: Reward, board_size: int, discount_factor: float = 0.95,
	             load_old_weights: bool = False) -> None:
		super().__init__(color)

		self.train_policy: TrainablePolicy = train_policy
		self.test_policy: OptimalTrainablePolicy = OptimalTrainablePolicy(board_size)
		self.immediate_reward: Reward = immediate_reward
		self.final_reward: Reward = final_reward
		self.board_size = board_size
		self.discount_factor: float = discount_factor

		self.replay_buffer: ReplayBuffer = ReplayBuffer((board_size ** 2 - 4) // 2)
		self.train_mode: Union[bool, None] = None

		# old and new network to compare training loss
		self.dnn: Sequential = self.create_model()

		if load_old_weights:
			self.load_weights()

		# save the weights of deep neural network periodically, i.e. when
		# self.n_training_cycles % self.persist_weights_every_n_times_trained == 0:
		self.persist_weights_every_n_times_trained: int = int(1e3)

		# Bookkeeping values
		self.n_training_cycles: int = 0

	def __str__(self) -> str:
		if self.train_mode:
			return f'Trainable{super().__str__()}, policy={self.train_policy}, immediate_reward={self.immediate_reward}, final_reward={self.final_reward}'
		else:
			return f'Trainable{super().__str__()}, policy={self.test_policy}'

	def train(self, persist_weights: bool = False) -> None:
		assert self.train_mode, 'Cannot train while not in train mode'

		self.n_training_cycles += 1

		states = np.array([self.board_to_nn_input(move[0]) for move in self.replay_buffer.buffer])
		# the goal is to update these old_q_values
		old_q_values = self.dnn.predict(states)

		for i in range(len(self.replay_buffer.buffer) - 1):
			# get move i: (s, a, r, t)
			prev_state, prev_action, prev_reward, _, locations = self.replay_buffer.buffer[i]
			# get move i+1: (s', a', r', t')
			curr_state, _, _, _, _ = self.replay_buffer.buffer[i + 1]

			# calculate the new estimate of Q(s,a)
			# via formula Q(s,a) = r + gamma * max Q(s', a')
			# but only consider the legal actions a'
			if len(locations) == 0:
				new_q_state_s = prev_reward
			else:
				indices = [row * self.board_size + col for (row, col) in locations]
				q_values = old_q_values[i + 1, indices]
				new_q_state_s = prev_reward + self.discount_factor * q_values.max()
			old_q_values[i, prev_action[0] * self.board_size + prev_action[1]] = new_q_state_s

		# use final reward
		last_state, last_action, last_reward, last_terminal, _ = self.replay_buffer.buffer[-1]
		old_q_values[-1, last_action[0] * self.board_size + last_action[1]] = last_reward

		# train the NN on the now updated q_values
		self.dnn.train_on_batch(states, old_q_values)

		if persist_weights and self.n_training_cycles % self.persist_weights_every_n_times_trained == 0:
			self._persist_weights()

	def next_action(self, board: Board, legal_actions: Actions) -> Action:
		q_values = self.dnn.predict(np.expand_dims(self.board_to_nn_input(board.board), axis=0))
		if self.train_mode:
			action: Action = self.train_policy.get_action(legal_actions, q_values)
		else:
			action: Action = self.test_policy.get_action(legal_actions, q_values)

		return action

	def _persist_weights(self, prefix: str = '') -> None:
		# clean and remake the folders
		weights_path: str = 'network_weights/' + self.color.name
		if os.path.exists(weights_path):
			shutil.rmtree(weights_path)  # clear the directory
		os.makedirs(weights_path)
		buffer_path: str = 'replay_buffers/' + self.color.name
		if os.path.exists(buffer_path):
			shutil.rmtree(buffer_path)  # clear the directory
		os.makedirs(buffer_path)
		hyperpar_path: str = 'hyper_values/' + self.color.name
		if os.path.exists(hyperpar_path):
			shutil.rmtree(hyperpar_path)  # clear the directory
		os.makedirs(hyperpar_path)

		# save the values
		col_time = prefix + self.color.name + datetime.datetime.now().strftime('%y%m%d%H%M%S')
		path: str = '{}/weights_agent_{}.h5'.format(weights_path, col_time)
		self.dnn.save(path, overwrite=True)

		path: str = os.path.join(buffer_path, 'replay_buffer_agent_{}.pkl'.format(col_time))
		self.replay_buffer.persist(path)

		path: str = os.path.join(hyperpar_path, 'vals_{}.pkl'.format(col_time))
		values = {'n_training_cycles': self.n_training_cycles}
		pickle.dump(values, open(path, 'wb'))

	def final_save(self) -> None:
		self._persist_weights('FINAL_')

	def load_weights(self, file_name=None) -> None:
		color = ('BLACK' if self.color is Color.BLACK else 'WHITE')
		weights_path = 'network_weights/' + color
		buffer_path = 'replay_buffers/' + color
		if file_name is None:
			all_paths = os.listdir(weights_path)
			all_paths = sorted(all_paths, reverse=True)
			path_network = all_paths[0] if len(all_paths) > 0 else None
			print(path_network)
			if path_network is None:
				return
			path_network = os.path.join(weights_path, path_network)
			name = os.path.basename(path_network)
			name = name.replace('.h5', '.pkl')
			path_replay = name.replace('weights_agent_', 'replay_buffer_agent_')
			path_replay = os.path.join(buffer_path, path_replay)
		else:
			path_network = os.path.join(weights_path, 'weights_agent_' + file_name + '.h5')
			path_replay = os.path.join(buffer_path, 'replay_buffer_agent_' + file_name + '.pkl')

		self.dnn.load_weights(path_network)  # loading in the action value network weights
		self.dnn = tf.keras.models.load_model(path_network)

		self.replay_buffer.load(path_replay)

		print('WEIGHTS HAVE BEEN LOADED -> CONTINUING TRAINING')

	@abstractmethod
	def create_model(self, verbose: bool = False, lr: float = 0.00025) -> Sequential:
		raise NotImplementedError

	@abstractmethod
	def board_to_nn_input(self, board: np.array) -> np.array:
		raise NotImplementedError
