from game_logic.board import Board
from game_logic.agents.agent import Agent
from utils.color import Color
from utils.intermediate_reward_functions.difference_with_prev_board import heur
import random
import copy


class MinimaxAgent(Agent):
    def __init__(self, color: Color, immediate_reward):
        super().__init__(color, immediate_reward)
        self.name = "minimax"

    def __str__(self):
        return f'{self.name}{super().__str__()}'

    def minimax(self, board: Board, player, legal_directions: dict, maxLevel: int = 2, level: int = 0, prev_best_points: dict = None) -> tuple:
        cur_best_points = None
        cur_best_move = None
        player2 = 1 - player

        for move in legal_directions:
            new_board = copy.deepcopy(board)
            done = new_board.take_action(move, legal_directions[move], player)
            points = 0
            if level < maxLevel:
                new_legal_actions: dict = new_board.get_legal_actions(player2)
                if not new_legal_actions: # pass -> player plays again
                    new_legal_actions: dict = new_board.get_legal_actions(player)
                    points, _ = self.minimax(new_board, new_legal_actions, player, maxLevel, level + 1, cur_best_points)
                else:
                    points, _ = self.minimax(new_board, new_legal_actions, player2, maxLevel, level + 1, cur_best_points)
            else:
                points = self.immediate_reward.immediate_reward(new_board, player)

            if player == self.color.value: #max_step
                if cur_best_points < points:
                    cur_best_points = points
                    cur_best_move = move
            elif player2 == self.color.value: #min step
                if cur_best_points > points:
                    cur_best_points = points
                    cur_best_move = move

        return cur_best_points, cur_best_move


    def get_next_action(self, board: Board, legal_directions: dict, maxLevel: int = 2, level: int = 0, prev_best_points: dict = None) -> tuple:
        if "pass" in legal_directions:
            return "pass", []
        player = self.color.value

        action = random.choice(list(legal_directions.keys()))
        legal_actions = legal_directions[action]

        return action, legal_actions

