from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class Plot:
	def __init__(self) -> None:
		self.episodes: List[int] = []

	def update(self, episode: int, scores: defaultdict) -> None:
		self.episodes.append(episode)

		plt.plot(self.episodes, scores['epsilon'], linestyle='--')

		for key, win_ratios in scores.items():
			if key != 'epsilon':
				plt.plot(self.episodes, win_ratios)

		# draw
		plt.title('Evaluation')
		plt.xlabel('episode')
		plt.ylabel('win ratio')
		plt.yticks(np.arange(0, 100 + 1, 10))
		fig = plt.gcf()
		plt.draw()
		plt.pause(0.001)

		# save plot
		fig.savefig('plots\\plot.png')
