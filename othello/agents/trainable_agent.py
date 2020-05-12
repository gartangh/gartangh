from abc import abstractmethod
from typing import Union

import numpy as np
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
	def __init__(self, color: Color, model_name: str, train_policy: TrainablePolicy, immediate_reward: Reward,
	             final_reward: Reward, board_size: int, discount_factor: float = 1.0) -> None:
		super().__init__(color)

		self.weights_path: str = f'weights\\{model_name}_{self.color.name}'
		self.train_policy: TrainablePolicy = train_policy
		self.test_policy: OptimalTrainablePolicy = OptimalTrainablePolicy(board_size)
		self.immediate_reward: Reward = immediate_reward
		self.final_reward: Reward = final_reward
		self.board_size = board_size
		self.discount_factor: float = discount_factor

		self.replay_buffer: ReplayBuffer = ReplayBuffer((board_size ** 2 - 4) // 2)
		self.train_mode: Union[bool, None] = None

		try:
			# create new model
			self.dnn: Sequential = self.create_model()
			# load existing weights
			self.load_weights()
		except:
			# create new model
			self.dnn: Sequential = self.create_model()
			# save initial weights
			self.save_weights()

	def __str__(self) -> str:
		if self.train_mode:
			return f'Trainable{super().__str__()}, policy={self.train_policy}, immediate_reward={self.immediate_reward}, final_reward={self.final_reward}'
		else:
			return f'Trainable{super().__str__()}, policy={self.test_policy}'

	def train(self) -> None:
		assert self.train_mode, 'Cannot train while not in train mode'

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

	def next_action(self, board: Board, legal_actions: Actions) -> Action:
		q_values = self.dnn.predict(np.expand_dims(self.board_to_nn_input(board.board), axis=0))
		if self.train_mode:
			action: Action = self.train_policy.get_action(legal_actions, q_values)
		else:
			action: Action = self.test_policy.get_action(legal_actions, q_values)

		return action

	def load_weights(self):
		self.dnn.load_weights(self.weights_path)
		print(f'Loaded weights from {self.weights_path}')

	def save_weights(self):
		self.dnn.save_weights(self.weights_path)
		print(f'Saved weights to {self.weights_path}')

	@abstractmethod
	def create_model(self, verbose: bool = False, lr: float = 0.00025) -> Sequential:
		raise NotImplementedError

	@abstractmethod
	def board_to_nn_input(self, board: np.array) -> np.array:
		raise NotImplementedError
