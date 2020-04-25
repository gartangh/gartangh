import datetime
import os
import pickle
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential

from game_logic.agents.trainable_agent import TrainableAgent
from game_logic.board import Board
from utils.color import Color
from utils.immediate_rewards.immediate_reward import ImmediateReward
from utils.policies.annealing_epsilon_greedy_policy import AnnealingEpsilonGreedyPolicy
from utils.policies.epsilon_greedy_policy import EpsilonGreedyPolicy


class DQNTrainableAgent(TrainableAgent):
	def __init__(self, color: Color, immediate_reward: ImmediateReward = None, board_size: int = 8,
	             load_old_weights: bool = False, start_epsilon: float = 0.99, end_epsilon: float = 0.01,
	             epsilon_steps: int = 75_000, allow_exploration: bool = False, policy_sampling: bool = False) -> None:
		super().__init__(color, immediate_reward, board_size, load_old_weights, policy_sampling)
		self.start_epsilon: float = start_epsilon
		self.epsilon: float = end_epsilon
		self.epsilon_steps: int = epsilon_steps
		self.allow_exploration: bool = allow_exploration
		self.discount_factor: float = 1.0

		# start with epsilon 0.99 and slowly decrease it
		self.play_policy: EpsilonGreedyPolicy = EpsilonGreedyPolicy(self.epsilon, board_size, policy_sampling)
		self.training_policy: AnnealingEpsilonGreedyPolicy = AnnealingEpsilonGreedyPolicy(self.start_epsilon,
		                                                                                  self.epsilon,
		                                                                                  self.epsilon_steps,
		                                                                                  board_size,
		                                                                                  self.allow_exploration,
		                                                                                  policy_sampling)

		# old and new network to compare training loss
		self.action_value_network: Sequential = self.create_model()

		# save the weights of action_value_network periodically, i.e. when
		# self.n_training_cycles % self.persist_weights_every_n_times_trained == 0:
		self.persist_weights_every_n_times_trained: int = int(1e3)

		# Bookkeeping values
		self.n_training_cycles: int = 0

	def __str__(self) -> str:
		return f'DQN{super().__str__()}'

	def train(self, persist_weights: bool = False) -> None:
		assert self.train_mode is True, 'Cannot train while not in train mode'

		self.n_training_cycles += 1

		states = np.array([self.board_to_nn_input(move[0]) for move in self.replay_buffer.buffer])
		# the goal is to update these old_q_values
		old_q_values = self.action_value_network.predict(states)

		for i in range(len(self.replay_buffer.buffer) - 1):
			# get move i: (s, a, r, t)
			prev_state, prev_action, prev_reward, prev_terminal = self.replay_buffer.buffer[i]
			# get move i+1: (s', a', r', t')
			curr_state, curr_action, curr_reward, curr_terminal = self.replay_buffer.buffer[i + 1]

			# calculate the new estimate of Q(s,a)
			# via formula Q(s,a) = r + gamma * max Q(s', a')
			# but only consider the legal actions a'
			legal_next_actions = Board._get_legal_actions(curr_state, self.board_size, self.color.value)
			if len(legal_next_actions) == 0:
				new_q_state_s = prev_reward
			else:
				indices = [row * self.board_size + col for (row, col) in legal_next_actions.keys()]
				q_values = old_q_values[i + 1, indices]
				new_q_state_s = prev_reward + self.discount_factor * q_values.max()
			old_q_values[i, prev_action[0] * self.board_size + prev_action[1]] = new_q_state_s

		# use final reward
		last_state, last_action, last_reward, last_terminal = self.replay_buffer.buffer[-1]
		old_q_values[-1, last_action[0] * self.board_size + last_action[1]] = last_reward

		# train the NN on the now updated q_values
		self.action_value_network.train_on_batch(states, old_q_values)

		if persist_weights and self.n_training_cycles % self.persist_weights_every_n_times_trained == 0:
			self._persist_weights()

	def get_next_action(self, board: Board, legal_actions: dict) -> tuple:
		if self.train_mode:
			action = self.training_policy.get_action(self.board_to_nn_input(board.board), self.action_value_network,
			                                         legal_actions)
		else:
			action = self.play_policy.get_action(self.board_to_nn_input(board.board), self.action_value_network,
			                                     legal_actions)

		return action, legal_actions[action]

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
		self.action_value_network.save(path, overwrite=True)

		path: str = os.path.join(buffer_path, 'replay_buffer_agent_{}.pkl'.format(col_time))
		self.replay_buffer.persist(path)

		path: str = os.path.join(hyperpar_path, 'vals_{}.pkl'.format(col_time))
		values = {'decisions_made': self.training_policy.decisions_made, 'n_training_cycles': self.n_training_cycles}
		pickle.dump(values, open(path, 'wb'))

	def final_save(self) -> None:
		self._persist_weights('FINAL_')

	def load_weights(self, file_name=None) -> None:
		color = ('BLACK' if self.color == Color.BLACK else 'WHITE')
		weights_path = 'network_weights/' + color
		buffer_path = 'replay_buffers/' + color
		hyperpar_path = 'hyper_values/' + color
		if file_name is None:
			# path_network = tf.train.latest_checkpoint(self.weight_persist_path)
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

			path_vals = name.replace('weights_agent_', 'vals_')
			path_vals = os.path.join(hyperpar_path, path_vals)
		else:
			path_network = os.path.join(weights_path, 'weights_agent_' + file_name + '.h5')
			path_replay = os.path.join(buffer_path, 'replay_buffer_agent_' + file_name + '.pkl')
			path_vals = os.path.join(hyperpar_path, 'vals_' + file_name + '.pkl')

		self.action_value_network.load_weights(path_network)  # loading in the action value network weights
		self.action_value_network = tf.keras.models.load_model(path_network)

		self.replay_buffer.load(path_replay)

		values = pickle.load(open(path_vals, 'rb'))

		self.training_policy.decisions_made = values['decisions_made']
		self.n_training_cycles = values['n_training_cycles']

		print('WEIGHTS HAVE BEEN LOADED -> CONTINUING TRAINING')

	def create_model(self, verbose: bool = False, lr: float = 0.00025) -> Sequential:
		raise NotImplementedError

	def board_to_nn_input(self, board: np.ndarray) -> np.array:
		raise NotImplementedError
