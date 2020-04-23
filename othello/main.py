import matplotlib.pyplot as plt
from colorama import init
from termcolor import colored
from tqdm import tqdm

from game_logic.agents.agent import Agent
from game_logic.agents.cnn_dqn_trainable_agent import CNNDQNTrainableAgent
from game_logic.agents.dqn_trainable_agent import DQNTrainableAgent
from game_logic.agents.human_agent import HumanAgent
from game_logic.agents.random_agent import RandomAgent
from game_logic.agents.minimax_agent import MinimaxAgent
from game_logic.agents.risk_regions_agent import RiskRegionsAgent
from game_logic.agents.trainable_agent import TrainableAgent
from game_logic.game import Game
from gui.controller import Controller
from utils.color import Color
from utils.config import Config
from utils.global_config import GlobalConfig
from utils.immediate_rewards.minimax_heuristic import MinimaxHeuristic


def main() -> None:
    # initialize colors
    init()

    # agents
    black = config.black
    white = config.white
    if config.train_all_agents:
        white = 'Rotatable'
    print(f'\nAgents:\n\tBlack:\t{black}\n\tWhite:\t{white}\n')

    win_rates = [0.0]
    if isinstance(black, DQNTrainableAgent):
        epsilons = [black.training_policy.current_eps_value]
    last_matches = []

    # initialize live plot
    if config.plot_win_ratio_live:
        plt.ion()  # non-blocking plot
        plt.title('Win ratio of black (red), epsilon (green)')
        plt.xlabel('number of games played')
        plt.ylabel('win ratio and epsilon')
        plt.show()

    if config.train_all_agents:
        agents_rotation = {1: MinimaxAgent(Color.WHITE,
            immediate_reward=MinimaxHeuristic(global_config.board_size),
            depth=1),
                           2: RandomAgent(Color.WHITE),
                           3: MinimaxAgent(Color.WHITE,
            immediate_reward=MinimaxHeuristic(global_config.board_size),
            depth=1),
                           4: RiskRegionsAgent(Color.WHITE, global_config.board_size),
                           5: MinimaxAgent(Color.WHITE,
            immediate_reward=MinimaxHeuristic(global_config.board_size),
            depth=1)}
        black_wins = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        rotation_num_episodes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        assert(len(agents_rotation) == len(black_wins))
        assert(len(agents_rotation) == len(rotation_num_episodes))
        
    for episode in tqdm(range(1, config.num_episodes + 1)):
        if config.train_all_agents:
            opponent = agents_rotation[1+episode%len(agents_rotation)]
            config.white = opponent
        # create new game
        game: Game = Game(global_config, config, episode)
        if isinstance(white, HumanAgent):
            # create GUI controller
            controller: Controller = Controller(game)
            controller.start()
        elif isinstance(black, DQNTrainableAgent):
            # Update epsilon annealing policy
            black.training_policy.update_policy(episode)
            # play game
            game.play()
        else:
            # play game
            game.play()
            
        if config.train_all_agents:
            rotation_num_episodes[1+episode%len(agents_rotation)] += 1

        # plot win ratio
        if config.plot_win_ratio:
            if game.board.num_black_disks > game.board.num_white_disks:
                last_matches.append(1)
                if config.train_all_agents:
                    black_wins[1+episode%len(agents_rotation)] += 1
            elif game.board.num_black_disks < game.board.num_white_disks:
                last_matches.append(-1)
            else:
                last_matches.append(0)

            if ((episode-1) % 100 == 0 or (episode % config.plot_every_n_episodes == config.plot_every_n_episodes - 1)) and len(last_matches) > 0:
                win_rates.append(sum(last_matches) / len(last_matches))
                if isinstance(black, DQNTrainableAgent):
                    epsilons.append(black.training_policy.current_eps_value)
                if config.plot_win_ratio_live:
                    plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], win_rates,
                             color='red')
                    if isinstance(black, TrainableAgent):
                        plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], epsilons,
                                 color='green')
                    plt.draw()
                    plt.pause(0.001)
                last_matches = []
                
    print()
    print()

    # print end score
    if config.train_all_agents:
        for key in agents_rotation:
            print(f'White: {agents_rotation[key]}')
            print_scores(rotation_num_episodes[key], black_wins[key], agents_rotation[key].num_games_won)
    else:
        print_scores(config.num_episodes, black.num_games_won, white.num_games_won)

    # plot win ratio
    if config.plot_win_ratio_live:
        # keep showing live plot
        plt.ioff()
        plt.show()
    elif config.plot_win_ratio:
        # show plot
        plt.ion()  # non-blocking plot
        plt.title('Win ratio of black (red), epsilon (green)')
        plt.xlabel('number of games played')
        plt.ylabel('win ratio and epsilon')
        plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], win_rates, color='red')
        plt.plot([i * config.plot_every_n_episodes for i in range(len(win_rates))], epsilons, color='green')
        plt.show()
        plt.draw()

    # save models
    if isinstance(black, TrainableAgent) and black.train_mode:
        black.final_save()
    if isinstance(white, TrainableAgent) and white.train_mode:
        white.final_save()

def print_scores(num_episodes: int, black_won: int, white_won: int):
    ties: int = num_episodes - black_won - white_won
    if black_won > white_won:
        print(colored(
            f'\nBLACK {black_won:>5}/{num_episodes:>5} ({black_won:>5}|{white_won:>5}|{ties:>5})\n',
            'red'))
    elif black_won < white_won:
        print(colored(
            f'\nWHITE {white_won:>5}/{num_episodes:>5} ({black_won:>5}|{white_won:>5}|{ties:>5})\n',
            'green'))
    else:
        print(colored(
            f'\nDRAW  {black_won:>5}/{num_episodes:>5} ({black_won:>5}|{white_won:>5}|{ties:>5})\n',
            'cyan'))

def log(logline: str, path: str = 'log.txt'):
    with open(path, 'a') as f:
        f.write(f'{logline}\n')


def hardcore_training(black, white, board_size, total_iterations: int = 100_000, interval_log: int = 5000):
    black_dqn = black
    white_dqn = white

    total_runs = total_iterations // interval_log
    for i in range(total_runs):
        black = black_dqn
        white = white_dqn
        black.num_games_won = 0
        white.num_games_won = 0

        black.set_train_mode(True)
        white.set_train_mode(True)
        num_episodes = interval_log
        black = black_dqn
        white = white_dqn
        # TODO: use configs
        # main(num_episodes, black, white, board_size, False, False, False)

        black.final_save()
        white.final_save()

        black.set_train_mode(False)
        white.set_train_mode(False)

        black_dqn = black
        white_dqn = white

        tournament_mode = True

        if isinstance(black, TrainableAgent):
            # test against random white
            print('test ' + str(i) + ', BLACK DQN VS WHITE RANDOM')
            black.num_games_won = 0
            white.num_games_won = 0
            num_episodes = 244
            white = RandomAgent(color=Color.WHITE)
            # TODO: use configs
            # main(num_episodes, black, white, board_size, False, tournament_mode, False)
            log('test ' + str(i) + '\tBLACK DQN VS WHITE RANDOM\t' + str(black.num_games_won) + '\t' + str(
                white.num_games_won))

        if isinstance(white, TrainableAgent):
            # test against random black
            black = black_dqn
            white = white_dqn
            print('test ' + str(i) + ', BLACK RANDOM VS WHITE DQN')
            black.num_games_won = 0
            white.num_games_won = 0
            num_episodes = 244
            black = RandomAgent(color=Color.BLACK)
            # TODO: use configs
            # main(num_episodes, black, white, board_size, False, tournament_mode, False)
            log('test ' + str(i) + '\tBLACK RANDOM VS WHITE DQN\t' + str(black.num_games_won) + '\t' + str(
                white.num_games_won))

        if isinstance(black, TrainableAgent):
            # test against risk region white
            black = black_dqn
            white = white_dqn
            print('test ' + str(i) + ', BLACK DQN VS WHITE RISK_REGION')
            black.num_games_won = 0
            white.num_games_won = 0
            num_episodes = 244
            white = RiskRegionsAgent(color=Color.WHITE)
            # TODO: use configs
            # main(num_episodes, black, white, board_size, False, tournament_mode, False)
            log('test ' + str(i) + '\tBLACK DQN VS WHITE RISK_REGION\t' + str(black.num_games_won) + '\t' + str(
                white.num_games_won))

        if isinstance(white, TrainableAgent):
            # test against risk region white
            black = black_dqn
            white = white_dqn
            print('test ' + str(i) + ', BLACK RISK_REGION VS WHITE DQN')
            black.num_games_won = 0
            white.num_games_won = 0
            num_episodes = 244
            black = RiskRegionsAgent(color=Color.BLACK)
            # TODO: use configs
            # main(num_episodes, black, white, board_size, False, tournament_mode, False)
            log('test ' + str(i) + '\tBLACK RISK_REGION VS WHITE DQN\t' + str(black.num_games_won) + '\t' + str(
                white.num_games_won))


if __name__ == '__main__':
    # one-time global configuration
    global_config: GlobalConfig = GlobalConfig(board_size=8, gui_size=400)  # global config

    # TRAIN
    config: Config = Config(
        black=CNNDQNTrainableAgent(
            Color.BLACK,
            immediate_reward=MinimaxHeuristic(global_config.board_size),
            board_size=global_config.board_size,
            start_epsilon = 0.99,
            end_epsilon = 0.01,
            epsilon_steps = 20_000,
            allow_exploration = True
        ),
        train_black=True,
        white=RandomAgent(Color.WHITE),
        train_white=False,
        num_episodes=150,
        plot_win_ratio=True,
        plot_win_ratio_live=True,
        verbose=False,
        verbose_live=False,
        tournament_mode=False,
        random_start = True,
        train_all_agents=True,
    )
    main()

    # EVALUATE
    config: Config = Config(
        black=config.black,
        train_black=False,
        white=RandomAgent(Color.WHITE),
        train_white=False,
        num_episodes=100,
        plot_win_ratio=False,
        plot_win_ratio_live=False,
        verbose=True,
        verbose_live=False,
        tournament_mode=False,
        train_all_agents=False,
        random_start=False,
    )
    main()

    # HUMAN
    config: Config = Config(
        black=config.black,
        train_black=False,
        white=HumanAgent(Color.WHITE),
        train_white=False,
        num_episodes=2,
        plot_win_ratio=False,
        plot_win_ratio_live=False,
        verbose=True,
        verbose_live=False,
        tournament_mode=False,
        train_all_agents=False,
        random_start=False,
    )
    main()
