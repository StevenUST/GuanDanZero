# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
# from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_guandan import *
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras


class TrainPipeline():
    def __init__(self, init_model=None):
        # training params
        self.game_batch_num = 10000

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        pass

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        pass

    def policy_update(self):
        """update the policy-value net"""
        pass
        # return loss, entropy

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
            for i in range(self.game_batch_num):
                # self.collect_selfplay_data(self.play_batch_size)
                # print("batch i:{}, episode_len:{}".format(
                #         i+1, self.episode_len))
                # if len(self.data_buffer) > self.batch_size:
                #     loss, entropy = self.policy_update()
                # # check the performance of the current model,
                # # and save the model params
                # if (i+1) % self.check_freq == 0:
                #     print("current self-play batch: {}".format(i+1))
                #     win_ratio = self.policy_evaluate()
                #     self.policy_value_net.save_model('./current_policy.model')
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