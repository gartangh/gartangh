from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class Plot:
	def __init__(self) -> None:
		self.episodes: List[int] = []

	def update(self, episode: int, scores: defaultdict) -> None:
		self.episodes.append(episode)
		for win_ratios in scores.values():
			plt.plot(self.episodes, win_ratios)

		# draw
		plt.title('Evaluation')
		plt.xlabel('episode')
		plt.ylabel('win ratio')
		plt.yticks(np.arange(0, 100 + 1, 10))
		plt.draw()
		plt.pause(0.001)
		plt.savefig('plots/Evaluation.png')
