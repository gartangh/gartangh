import os
import datetime
import matplotlib.pyplot as plt

from game_logic.agents.dqn_trainable_agent import DQNTrainableAgent


class Plot:
	def __init__(self, black: DQNTrainableAgent = None) -> None:
		self.black: DQNTrainableAgent = black
		self.win_rates: list = [0.0]
		self.episodes: list = [0]
		self.epsilons: list = [black.training_policy.current_eps_value]
		self.last_matches: list = []
		self.live_plot: bool = False
		self.episode_opponent_switch: list = []  # list of episode numbers on which the opponent changed

		plt.title('Win ratio and epsilon (green, dotted) of black')
		plt.xlabel('number of games played')
		plt.ylabel('win ratio and epsilon')

	def toggle_live_plot(self, on: bool) -> None:
		if on:
			self.live_plot = True
			plt.ion()
			plt.draw()
			plt.pause(0.001)
		else:
			self.live_plot = False
			plt.ioff()
			plt.close('all')  # close the window

		self.episode_opponent_switch.append(self.episodes[-1])

	def save_plot(self) -> None:
		path = "plots/"
		if not os.path.exists(path):
			os.makedirs(path)

		fig, ax = plt.subplots()
		ax.set_title('Win ratio and epsilon (green, dotted) of black')
		ax.set_xlabel('number of games played')
		ax.set_ylabel('win ratio and epsilon')
		for i in range(len(self.episode_opponent_switch)):
			indices = slice(self.episode_opponent_switch[i],
			                self.episode_opponent_switch[i + 1] if i + 1 < len(self.episode_opponent_switch) else -1)
			ax.plot(self.episodes[indices], self.win_rates[indices])
		ax.plot(self.episodes, self.epsilons, color='green', linestyle='--')
		# uncomment the line below to make the name contain the time
		# fig.savefig('{}winratio{}.png'.format(path, datetime.datetime.now().strftime('%y%m%d%H%M%S')))
		# else:
		fig.savefig(path + 'winratio.png')

	def update(self, num_black_disks: int, num_white_disks: int, episode: int, plot_every_n_episodes: int):
		if num_black_disks > num_white_disks:
			self.last_matches.append(1)
		elif num_black_disks < num_white_disks:
			self.last_matches.append(-1)
		else:
			self.last_matches.append(0)

		if episode % plot_every_n_episodes == plot_every_n_episodes - 1 and len(self.last_matches) > 0:
			self.win_rates.append(sum(self.last_matches) / len(self.last_matches))
			self.epsilons.append(self.black.training_policy.current_eps_value)
			self.episodes.append(self.episode_opponent_switch[-1] + episode)

			if self.live_plot:
				# give different colors for different opponents by plotting their winrates separately
				plt.cla()  # clear axes
				plt.title('Win ratio and epsilon (green, dotted) of black')
				plt.xlabel('number of games played')
				plt.ylabel('win ratio and epsilon')
				for i in range(len(self.episode_opponent_switch)):
					indices = slice(self.episode_opponent_switch[i],
					                self.episode_opponent_switch[i+1] if i+1 < len(self.episode_opponent_switch) else -1)
					plt.plot(self.episodes[indices], self.win_rates[indices])
				plt.plot(self.episodes, self.epsilons, color='green', linestyle='--')
				plt.draw()
				plt.pause(0.001)

			self.last_matches = []