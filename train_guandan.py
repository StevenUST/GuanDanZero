# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from typing import Final
from collections import deque
from time import time as get_time
from mcts_guandan import MCTSPlayer
from guandan_net_tensorflow import GuandanNetForTwo
from guandan_game import GDGame2P

class TrainPipeline():
    
    base_path : Final[str] = "./saved_model/"
    model_name_base : Final[str] = "GDModel2P_v1"
    zfill_size : Final[int] = 7
    
    def __init__(self):
        # training params
        self.game_batch_num = 5
        self.guandan_model = GuandanNetForTwo(max_file_num=2)
        self.baseline_model = GuandanNetForTwo()
        self.game = GDGame2P()
        self.mcts_player = MCTSPlayer(self.guandan_model.policy_value_function, c_puct=5, n_playout=25)
        self.epoch_num = 10000
        self.episode_len = 0
        
        self.play_batch_size = 1
        self.train_batch = 2
        self.buffer_size = 10000
        self.training_data = deque(maxlen=self.buffer_size)
        self.check_freq = 1
        self.train_time = 0
        
        self.best_win_ratio = -1.0
        self.best_model_index = 0
        
        self.guandan_model.save_model(f"{TrainPipeline.base_path}{TrainPipeline.model_name_base}_{str.zfill(str(self.train_time), TrainPipeline.zfill_size)}")

    def collect_selfplay_data(self, game_round : int = 1):
        """collect self-play data for training in one game episode"""
        for _ in range(game_round):
            states, mcts_probs, current_players, winner, size = self.game.start_self_play(self.mcts_player)
            self.episode_len = size
            for j in range(size):
                data = list()
                data.extend(states[j])
                data.append(mcts_probs[j])
                val = 1 if current_players[j] == winner else -1
                temp = np.array([val], dtype=np.float32)
                data.append(temp)
                self.training_data.append(data)
                

    def policy_update(self, repeat_time : int = 3) -> np.ndarray:
        """update the policy-value net"""
        if len(self.training_data) == 0:
            raise BufferError("The data buffer is empty!")
        random.shuffle(self.training_data)
        mini_batch = list()
        num = 0
        size = min(len(self.training_data), self.train_batch)
        
        while num < size:
            mini_batch.append(self.training_data.pop())
            num += 1
            
        self_state_batch1, self_state_batch2, oppo_state_batch1, oppo_state_batch2, last_action_batch1, last_action_batch2,\
            level_batch, mcts_probs_batch, winner_batch = [], [], [], [], [], [], [], [], []
            
        for data in mini_batch:
            self_state_batch1.append(data[0])
            self_state_batch2.append(data[1])
            oppo_state_batch1.append(data[2])
            oppo_state_batch2.append(data[3])
            last_action_batch1.append(data[4])
            last_action_batch2.append(data[5])
            level_batch.append(data[6])
            mcts_probs_batch.append(data[7])
            winner_batch.append(data[8])
        print("Start Training!")
        t = get_time()
        losses = list()
        for _ in range(repeat_time):
            loss = self.guandan_model.train_step(self_state_batch1, self_state_batch2, oppo_state_batch1, oppo_state_batch2,
                                                        last_action_batch1, last_action_batch2, level_batch, mcts_probs_batch, winner_batch)
            losses.append(loss)
            print(f"[prob_loss, value_loss] = {loss.tolist()}")
        print(f"Training Batch = {size}; Training time = {get_time() - t} seconds!")
        return losses
    
    def policy_evaluate_previous_model(self, n_games : int = 2) -> float:
        """
        Evaluate the trained policy by playing against the previous model
        Note: this is only for monitoring the progress of training
        """
        self.baseline_model.restore_model(TrainPipeline.base_path, f"{TrainPipeline.model_name_base}_{str.zfill(str(self.best_model_index), TrainPipeline.zfill_size)}.meta")
        baseline_player = MCTSPlayer(self.baseline_model.policy_value_function, n_playout=25)
        
        win_time = 0
        win_ratio = 0.0
        
        for i in range(n_games):
            result = self.game.start_play_against_other(self.mcts_player, baseline_player)
            if result == 1:
                win_time += 1
            if i == n_games - 1:
                win_ratio = float(win_time) / float(n_games)
                break
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                print(f"Now it is batch {i + 1}!")
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.training_data) > self.train_batch:
                    _ = self.policy_update()
                if (i+1) % self.check_freq == 0:
                    self.guandan_model.save_model(f"{TrainPipeline.base_path}{TrainPipeline.model_name_base}_{str.zfill(str(self.train_time + 1), TrainPipeline.zfill_size)}")
                    self.train_time += 1
                    # print("current self-play batch: {}".format(i+1))
                    # win_ratio = self.policy_evaluate_previous_model()
                    # print(f"win_ratio = {win_ratio}")
                    # if win_ratio > self.best_win_ratio:
                    #     self.best_win_ratio = win_ratio
                    #     self.guandan_model.save_model(f"{TrainPipeline.base_path}{TrainPipeline.model_name_base}_{str.zfill(str(self.train_time + 1), TrainPipeline.zfill_size)}")
                    #     self.train_time += 1
                    #     self.best_model_index = self.train_time
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()