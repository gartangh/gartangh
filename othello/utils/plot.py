import os
from typing import List

import matplotlib.pyplot as plt

from agents.trainable_agent import TrainableAgent
from policies.epsilon_greedy_annealing_trainable_policy import EpsilonGreedyAnnealingTrainablePolicy


class Plot:
	def __init__(self, black: TrainableAgent) -> None:
		self.black: TrainableAgent = black
		self.win_rates: List[float] = [0.0]
		self.episodes: List[int] = [0]
		if isinstance(self.black.train_policy, EpsilonGreedyAnnealingTrainablePolicy):
			self.epsilons: List[float] = [black.train_policy.inner_policy.epsilon]
		self.last_matches: List[int] = []
		self.live_plot: bool = False
		self.index_opponent_switch: List[int] = []  # list of episode numbers on which the opponent changed

		plt.title('Win ratio and epsilon (green, dotted) of black')
		plt.xlabel('number of games played')
		plt.ylabel('win ratio and epsilon')

	def set_plot_live(self, on: bool) -> None:
		if on:
			self.live_plot = True
			plt.ion()
			plt.draw()
			plt.pause(0.001)
		else:
			self.live_plot = False
			plt.ioff()
			plt.close('all')  # close the window

		self.index_opponent_switch.append(len(self.episodes) - 1)

	def save_plot(self) -> None:
		path = "plots/"
		if not os.path.exists(path):
			os.makedirs(path)

		fig, ax = plt.subplots()
		ax.set_title('Win ratio and epsilon (green, dotted) of black')
		ax.set_xlabel('number of games played')
		ax.set_ylabel('win ratio and epsilon')
		for i in range(len(self.index_opponent_switch)):
			indices = slice(self.index_opponent_switch[i],
			                self.index_opponent_switch[i + 1] if i + 1 < len(self.index_opponent_switch) else -1)
			ax.plot(self.episodes[indices], self.win_rates[indices])
		if isinstance(self.black.train_policy, EpsilonGreedyAnnealingTrainablePolicy):
			ax.plot(self.episodes, self.epsilons, color='green', linestyle='--')
		# uncomment the line below to make the name contain the time
		# fig.savefig('{}winratio{}.png'.format(path, datetime.datetime.now().strftime('%y%m%d%H%M%S')))
		# else:
		fig.savefig(path + 'winratio.png')

	def update(self, num_black_disks: int, num_white_disks: int, episode: int, plot_every_n_episodes: int) -> None:
		if num_black_disks > num_white_disks:
			self.last_matches.append(1)
		elif num_black_disks < num_white_disks:
			self.last_matches.append(0)
		else:
			self.last_matches.append(0.5)

		if episode % plot_every_n_episodes == 0 and len(self.last_matches) > 0:
			self.win_rates.append(sum(self.last_matches) / len(self.last_matches))
			if isinstance(self.black.train_policy, EpsilonGreedyAnnealingTrainablePolicy):
				self.epsilons.append(self.black.train_policy.inner_policy.epsilon)
			self.episodes.append(self.episodes[self.index_opponent_switch[-1]] + episode)

			if self.live_plot:
				# give different colors for different opponents by plotting their winrates separately
				plt.cla()  # clear axes
				plt.title('Win ratio and epsilon (green, dotted) of black')
				plt.xlabel('number of games played')
				plt.ylabel('win ratio and epsilon')
				if isinstance(self.black.train_policy, EpsilonGreedyAnnealingTrainablePolicy):
					plt.plot(self.episodes, self.epsilons, color='green', linestyle='--')
				for i in range(len(self.index_opponent_switch)):
					indices = slice(self.index_opponent_switch[i],
					                self.index_opponent_switch[i + 1] if i + 1 < len(
						                self.index_opponent_switch) else len(self.win_rates))
					plt.plot(self.episodes[indices], self.win_rates[indices])
				plt.draw()
				plt.pause(0.001)

			self.last_matches = []
