# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from typing import List
from collections import deque
from time import time as get_time
# from game import Board, Game
# from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_guandan import MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras
from guandan_net_tensorflow import GuandanNetForTwo
from guandan_game import GDGame2P

class TrainPipeline():
    def __init__(self):
        # training params
        self.game_batch_num = 1
        self.guandan_model = GuandanNetForTwo()
        self.game = GDGame2P()
        self.mcts_player1 = MCTSPlayer(self.guandan_model.policy_value_function, c_puct=5, n_playout=300)
        self.mcts_player2 = MCTSPlayer(self.guandan_model.policy_value_function, c_puct=5, n_playout=300)
        self.epoch_num = 10000
        self.episode_len = 0
        
        self.play_batch_size = 1
        self.train_batch = 2
        self.buffer_size = 10000
        self.training_data = deque(maxlen=self.buffer_size)
        self.train_time = 5
        self.check_freq = 1

    def collect_selfplay_data(self, game_round : int = 1):
        """collect self-play data for training in one game episode"""
        for _ in range(game_round):
            states, mcts_probs, current_players, winner, size = self.game.start_play(self.mcts_player1, self.mcts_player2)
            self.episode_len = size
            for j in range(size):
                data = list()
                data.extend(states[j])
                data.append(mcts_probs[j])
                val = 1 if current_players[j] == winner else -1
                temp = np.array([val], dtype=np.float32)
                data.append(temp)
                self.training_data.append(data)
                

    def policy_update(self) -> np.ndarray:
        """update the policy-value net"""
        mini_batch = random.sample(self.training_data, self.train_batch)

        self_state_batch1 = [data[0] for data in mini_batch]
        self_state_batch2 = [data[1] for data in mini_batch]
        oppo_state_batch1 = [data[2] for data in mini_batch]
        oppo_state_batch2 = [data[3] for data in mini_batch]
        print(oppo_state_batch2)
        last_action_batch1 = [data[4] for data in mini_batch]
        last_action_batch2 = [data[5] for data in mini_batch]
        level_batch = [data[6] for data in mini_batch]
        mcts_probs_batch = [data[7] for data in mini_batch]
        winner_batch = [data[8] for data in mini_batch]
        t = get_time()
        losses = list()
        for _ in range(self.train_time):
            loss = self.guandan_model.train_step(self_state_batch1, self_state_batch2, oppo_state_batch1, oppo_state_batch2,
                                                        last_action_batch1, last_action_batch2, level_batch, mcts_probs_batch, winner_batch)
            losses.append(loss)
        print(f"Training time = {(get_time() - t) / self.train_time}")
        return losses

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        pass
        # return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            save_time = 0
            for i in range(self.game_batch_num):
                print(f"Now it is batch {i + 1}!")
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.training_data) > self.train_batch:
                    losses = self.policy_update()
                # # check the performance of the current model,
                # # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    # win_ratio = self.policy_evaluate()
                    self.guandan_model.save_model(f'./saved_model/guandan_model_v1_{save_time}.model')
                    save_time += 1
                #     if win_ratio > self.best_win_ratio:
                #         print("New best policy!!!!!!!!")
                #         self.best_win_ratio = win_ratio
                #         # update the best_policy
                #         self.policy_value_net.save_model('./best_policy.model')
                #         if (self.best_win_ratio == 1.0 and
                #                 self.pure_mcts_playout_num < 5000):
                #             self.pure_mcts_playout_num += 1000
                #             self.best_win_ratio = 0.0
                pass
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()