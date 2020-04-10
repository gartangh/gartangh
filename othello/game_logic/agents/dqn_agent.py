import numpy as np
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from game_logic.agents.trainable_agent import TrainableAgent
from game_logic.board import Board
from utils.color import Color
from utils.immediate_rewards.immediate_reward import ImmediateReward
from utils.policies.annealing_epsilon_greedy_policy import AnnealingEpsilonGreedyPolicy
from utils.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from utils.policies.random_policy import RandomPolicy

import datetime


class DQNAgent(TrainableAgent):
	def __init__(self, color: Color, immediate_reward: ImmediateReward = None, board_size: int = 8):
		super().__init__(color, immediate_reward, board_size)
		self.end_eps: float = 0  # epsilon
		self.discount_factor: float = 0.99

		# start with epsilon 0.99 and slowly decrease it over 75 000 steps
		self.play_policy: EpsilonGreedyPolicy = EpsilonGreedyPolicy(self.end_eps, board_size)
		self.training_policy: AnnealingEpsilonGreedyPolicy = AnnealingEpsilonGreedyPolicy(0.99, self.end_eps, 75_000,
		                                                                                  board_size)
		self.buffer_filling_policy: RandomPolicy = RandomPolicy()

		# after n_steps_start_learning steps, use policy instead of buffer_filling_policy
		# and start training the neural net
		self.n_steps_start_learning: int = 32

		# old and new network to compare training loss
		self.action_value_network: Sequential = self.create_model()
		self.target_network: Sequential = self.create_model()

		# number of steps per mini batch
		self.mini_batch_size: int = 32

		# learn every learning_frequency steps (after n_steps_start_learning have passed)
		self.learning_frequency: int = 32

		# save the weights of action_value_network periodically, i.e. when
		# self.n_training_cycles % self.persist_weights_every_n_times_trained == 0:
		self.persist_weights_every_n_times_trained: int = int(1e3)
		self.weight_persist_path: str = 'network_weights/'
		if not os.path.exists(self.weight_persist_path):
			os.makedirs(self.weight_persist_path)

		# rate at which target_network is updated to be the same as action_value_network
		self.target_network_update_freq: float = 0.01 * self.replay_buffer.size

		# Bookkeeping values
		self.n_training_cycles: int = 0
		self.n_steps: int = 0
		self.n_episodes: int = 0

	def __str__(self):
		return f'DQN{super().__str__()}'

	def create_model(self, verbose: bool = False, lr: float = 0.00025) -> Sequential:
		# input: 2 nodes per board location:
		#              - 1 node that is 0 if location does not contain black, else 1
		#              - 1 node that is 0 if location does not contain white, else 1
		# output: 1 node per board location, with probabilities to take action on that location
		model: Sequential = Sequential()
		model.add(Dense(2 * self.board_size ** 2, input_shape=(2 * self.board_size ** 2,), activation='relu'))
		model.add(Dense(self.board_size ** 2, activation='softmax'))
		model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr))

		return model

	def train(self, board: Board, action: tuple, reward: float, next_board: Board, terminal: bool,
	          render: bool = False):
		assert (self.train_mode is True)
		self.episode_rewards.append(reward)
		self.replay_buffer.add(board.board, action, reward, next_board.board, terminal)

		if self._can_start_learning():
			training_error = self.q_learn_mini_batch()
			self.training_errors.append(training_error)
			self._persist_weights_if_necessary()

		self.n_steps += 1

		if self.n_steps % self.target_network_update_freq == 1:
			self.update_target_network()

		return

	def get_next_action(self, board: Board, legal_actions: dict) -> tuple:
		if self.n_steps < self.n_steps_start_learning:
			action = self.buffer_filling_policy.get_action(board.board, legal_actions)
		elif self.train_mode is True:
			action = self.training_policy.get_action(self.board_to_nn_input(board.board), self.action_value_network,
			                                         legal_actions)
		else:
			action = self.play_policy.get_action(self.board_to_nn_input(board.board), self.action_value_network,
			                                     legal_actions)

		return action, legal_actions[action]

	def q_learn_mini_batch(self) -> list:
		self.n_training_cycles += 1

		# Sample a mini batch from our buffer
		mini_batch: list = self.replay_buffer.sample(self.mini_batch_size)

		# Extract states and subsequent states from mini batch
		states: np.array = np.array([self.board_to_nn_input(sample[0]) for sample in mini_batch])
		next_states: np.array = np.array([self.board_to_nn_input(sample[3]) for sample in mini_batch])

		# reshape from (x,1,y) to (x,y)
		states: np.array = states.reshape((states.shape[0], states.shape[-1]))
		next_states: np.array = next_states.reshape((states.shape[0], states.shape[-1]))

		# We predict the Q values for all current states in the batch using the online network to use as targets.
		# The Q values for the state-action pairs that are being trained on in the mini batch will be overridden with
		# the real target r + gamma * argmax[Q(s', a); target network]
		targets = self.action_value_network.predict(states)

		# Predict all next Q values for the subsequent states in the batch using the target network
		target_q_values_next_states = self.target_network.predict(next_states)

		for sample_nr, transition in enumerate(mini_batch):
			# action: tuple, reward: float, next_board: np.ndarray, terminal: bool
			__, action, reward, next_board, terminal = transition
			q_values: np.array = target_q_values_next_states[sample_nr].flatten()

			# associate each q value with its location on the board
			q_values: list = [(q_values[row * self.board_size + col], (row, col)) for row in range(self.board_size) for
			                  col in range(self.board_size)]

			# calculate the best q value achievable in next_state and the corresponding action
			best_next_action: str = 'pass'
			q_value_next_state: int = 0
			legal_actions = Board._get_legal_actions(next_board, self.board_size, self.color.value)

			# get best legal action by sorting according to q value and taking the last legal entry
			q_values: list = sorted(q_values, key=lambda q: q[0])
			n_actions: int = len(q_values) - 1
			while n_actions >= 0:
				if q_values[n_actions][1] in legal_actions:
					best_next_action = q_values[n_actions][1]
					q_value_next_state: int = q_values[n_actions][0]
					break
				n_actions -= 1

			if best_next_action is not 'pass':
				# Insert real target value using r + gamma * argmax[Q(s', a); target network] if not terminal
				if terminal:
					# This is important as it gives you a stable, consistent reward which will not fluctuate (i.e. depend on the target network)
					targets[sample_nr, action[0] * self.board_size + action[1]] = reward
				else:
					targets[sample_nr, action[0] * self.board_size + action[
						1]] = reward + self.discount_factor * q_value_next_state

		training_loss = self.action_value_network.train_on_batch(states, targets)

		return training_loss

	def update_target_network(self) -> None:
		target_weights = self.action_value_network.get_weights()
		self.target_network.set_weights(target_weights)

	def _can_start_learning(self) -> bool:
		return self.n_steps > self.n_steps_start_learning and \
		       self.n_steps % self.learning_frequency == 0 and \
		       self.replay_buffer.n_obs > self.mini_batch_size

	def _persist_weights_if_necessary(self) -> None:
		if self.n_training_cycles % self.persist_weights_every_n_times_trained == 0:
			name = "BLACK" if self.color == Color.BLACK else "WHITE"
			name = name + datetime.datetime.now().strftime("%y%m%d%H%M%S")
			path: str = '{}/weights_agent_{}.h5f'.format(self.weight_persist_path, name)
			self.action_value_network.save_weights(path, overwrite=True)

			path: str = 'replay_buffers'
			if not os.path.exists(path):
				os.makedirs(path)
			file_path: str = os.path.join(path, 'replay_buffer_agent_{}.pkl'.format(name))
			self.replay_buffer.persist(file_path)

	def final_save(self) -> None:
		name = "FINAL_"
		name += "BLACK" if self.color == Color.BLACK else "WHITE"
		name += datetime.datetime.now().strftime("%y%m%d%H%M%S")
		path: str = '{}/weights_agent_{}.h5f'.format(self.weight_persist_path, name)
		self.action_value_network.save_weights(path, overwrite=True)

		path: str = 'replay_buffers'
		if not os.path.exists(path):
			os.makedirs(path)
		file_path: str = os.path.join(path, 'replay_buffer_agent_{}.pkl'.format(name))
		self.replay_buffer.persist(file_path)

