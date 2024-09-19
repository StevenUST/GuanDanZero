# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from typing import Final, Tuple, List
from guandan_net_tensorflow import GuandanNetForTwo
from cardcomb import CardComb

from utils import cardsToDict, cardDictToModelList, findAllCombs, getFlagsForActions_Flatten, filterActions, updateCardCombsAfterAction, indexOfCombBase
from utils2 import generate_2_random_card_lists, get_training_data_from_raw_data, simulate_two_card_dicts, get_progress_from_node, get_action_from_progress_nodes

import pickle

class TrainPipeline():
    
    data_base_path : Final[str] = "./collected_data/basic/"
    base_path : Final[str] = "./saved_model/basic/"
    model_name_base : Final[str] = "GDModel2P_basic_v1"
    zfill_size : Final[int] = 4
    
    def __init__(self):
        self.guandan_model = GuandanNetForTwo(max_file_num=2)
        self.batch_size = 64
        self.train_time = 0

    def collect_selfplay_data(self, num1 : int, num2 : int, level : int, size : int = 1000000) -> Tuple[str, int]:
        """collect self-play data for training in one game episode"""
        """This function returns the path of file that stores the training data"""
        file_name = f"basic_training-({num1},{num2})-({level}).txt"
        level_flags = [0] * 13
        level_flags[level - 1] = 1
        level_data = str(level_flags).replace(" ", '')
        data_size = 0
        with open(f"{TrainPipeline.data_base_path}{file_name}", "a") as file:
            for _ in range(size):
                data = generate_2_random_card_lists(num1, num2)
                cd1 = cardsToDict(data[0])
                cd2 = cardsToDict(data[1])
                combs1 = findAllCombs(cd1, level)
                combs2 = findAllCombs(cd2, level)
                last_action : CardComb = CardComb.pass_cardcomb()
                top_node, _ = simulate_two_card_dicts(cd1, cd2, level)
                val = top_node.update_recursively()
                if val == 1:
                    progress = get_progress_from_node(top_node)
                    final_progress = get_action_from_progress_nodes(progress)[0]
                    for step in final_progress:
                        if step[0] == 0:
                            # Self-data
                            self_condition = cardDictToModelList(cd1, level)
                            self_action_flags = getFlagsForActions_Flatten(combs1, last_action, level)
                            # Oppo-data
                            oppo_condition = cardDictToModelList(cd2, level)
                            oppo_action_flags = getFlagsForActions_Flatten(combs2, last_action, level)
                            
                            # LastAction-data
                            action_type = last_action.type_index()
                            action_type_flags = [0] * 17
                            action_type_flags[action_type] = 1
                            
                            action_rank = last_action.rank
                            action_rank_flags = [0] * 15
                            action_rank_flags[action_rank] = 1
                            
                            # Prob Label
                            comb = step[1]
                            index = indexOfCombBase(comb)
                            prob_label = [0.0] * 194
                            prob_label[index] = 1.0
                            
                            # Generating Training Data
                            self_data = str(self_condition).replace(" ", '')
                            self_action_flag_data = str(self_action_flags).replace(" ", '')
                            
                            oppo_data = str(oppo_condition).replace(" ", '')
                            oppo_action_flag_data = str(oppo_action_flags).replace(" ", '')
                            
                            action_type_data = str(action_type_flags).replace(" ", '')
                            action_rank_data = str(action_rank_flags).replace(" ", '')
                            
                            prob_label_data = str(prob_label).replace(" ", '')
                            
                            val_data = str([round(float(val), 1)])
                            
                            training_data = f"{self_data}|{self_action_flag_data}|{oppo_data}|{oppo_action_flag_data}|{action_type_data}|{action_rank_data}|{level_data}|{prob_label_data}|{val_data}"
                            file.write(f"{training_data}\n")
                            data_size += 1
                            if data_size % 1000 == 0:
                                print(f"Data Size is {data_size}")
                            
                            _, combs1 = updateCardCombsAfterAction(combs1, cd1, comb)
                            last_action = CardComb.pass_cardcomb() if step[1] == None else step[1]
                        else:
                            _, combs2 = updateCardCombsAfterAction(combs2, cd2, step[1])
                            last_action = CardComb.pass_cardcomb() if step[1] == None else step[1]
                else:
                    # Self-data
                    self_condition = cardDictToModelList(cd1, level)
                    self_action_flags = getFlagsForActions_Flatten(combs1, last_action, level)
                    # Oppo-data
                    oppo_condition = cardDictToModelList(cd2, level)
                    oppo_action_flags = getFlagsForActions_Flatten(combs2, last_action, level)
                    # LastAction-data
                    action_type = last_action.type_index()
                    action_type_flags = [0] * 17
                    action_type_flags[action_type] = 1
                    
                    action_rank = last_action.rank
                    action_rank_flags = [0] * 15
                    action_rank_flags[action_rank] = 1
                    
                    # Prob Label
                    prob_label = [0.0] * 194
                    legal_actions = filterActions(combs1, last_action, level)
                    value = 1.0 / float(len(legal_actions))
                    for comb in legal_actions:
                        index = indexOfCombBase(comb)
                        prob_label[index] = value
                    prob_label_data = "["
                    for val in prob_label:
                        prob_label_data = prob_label_data + str(round(val, 4)) + ','
                    prob_label_data = prob_label_data[:-1]
                    prob_label_data = prob_label_data + "]"
                    
                    self_data = str(self_condition).replace(" ", '')
                    self_action_flag_data = str(self_action_flags).replace(" ", '')
                    
                    oppo_data = str(oppo_condition).replace(" ", '')
                    oppo_action_flag_data = str(oppo_action_flags).replace(" ", '')
                    
                    action_type_data = str(action_type_flags).replace(" ", '')
                    action_rank_data = str(action_rank_flags).replace(" ", '')
                    
                    val_data = str([round(float(-1), 1)])
                    
                    training_data = f"{self_data}|{self_action_flag_data}|{oppo_data}|{oppo_action_flag_data}|{action_type_data}|{action_rank_data}|{level_data}|{prob_label_data}|{val_data}"
                    file.write(f"{training_data}\n")
                    data_size += 1
                    if data_size % 1000 == 0:
                        print(f"Data Size is {data_size}")
            file.close()
        return (f"{TrainPipeline.data_base_path}{file_name}", data_size)

    def collect_selfplay_data_csv(self, num1 : int, num2 : int, level : int, size : int = 1000000) -> Tuple[str, int]:
        """collect self-play data for training in one game episode"""
        """This function returns the path of file that stores the training data"""
        file_name = f"basic_training-({num1},{num2})-({level}).txt"
        level_flags = [0] * 13
        level_flags[level - 1] = 1
        level_data = str(level_flags).replace(" ", '')
        data_size = 0
        with open(f"{TrainPipeline.data_base_path}{file_name}", "a") as file:
            for _ in range(size):
                data = generate_2_random_card_lists(num1, num2)
                cd1 = cardsToDict(data[0])
                cd2 = cardsToDict(data[1])
                combs1 = findAllCombs(cd1, level)
                combs2 = findAllCombs(cd2, level)
                last_action : CardComb = CardComb.pass_cardcomb()
                top_node, _ = simulate_two_card_dicts(cd1, cd2, level)
                val = top_node.update_recursively()
                if val == 1:
                    progress = get_progress_from_node(top_node)
                    final_progress = get_action_from_progress_nodes(progress)[0]
                    for step in final_progress:
                        if step[0] == 0:
                            # Self-data
                            self_condition = cardDictToModelList(cd1, level)
                            self_action_flags = getFlagsForActions_Flatten(combs1, last_action, level)
                            # Oppo-data
                            oppo_condition = cardDictToModelList(cd2, level)
                            oppo_action_flags = getFlagsForActions_Flatten(combs2, last_action, level)
                            
                            # LastAction-data
                            action_type = last_action.type_index()
                            action_type_flags = [0] * 17
                            action_type_flags[action_type] = 1
                            
                            action_rank = last_action.rank
                            action_rank_flags = [0] * 15
                            action_rank_flags[action_rank] = 1
                            
                            # Prob Label
                            comb = step[1]
                            index = indexOfCombBase(comb)
                            prob_label = [0.0] * 194
                            prob_label[index] = 1.0
                            
                            # Generating Training Data
                            self_data = str(self_condition).replace(" ", '')
                            self_action_flag_data = str(self_action_flags).replace(" ", '')
                            
                            oppo_data = str(oppo_condition).replace(" ", '')
                            oppo_action_flag_data = str(oppo_action_flags).replace(" ", '')
                            
                            action_type_data = str(action_type_flags).replace(" ", '')
                            action_rank_data = str(action_rank_flags).replace(" ", '')
                            
                            prob_label_data = str(prob_label).replace(" ", '')
                            
                            val_data = str([round(float(val), 1)])
                            
                            training_data = f"{self_data}|{self_action_flag_data}|{oppo_data}|{oppo_action_flag_data}|{action_type_data}|{action_rank_data}|{level_data}|{prob_label_data}|{val_data}"
                            file.write(f"{training_data}\n")
                            data_size += 1
                            if data_size % 1000 == 0:
                                print(f"Data Size is {data_size}")
                            
                            _, combs1 = updateCardCombsAfterAction(combs1, cd1, comb)
                            last_action = CardComb.pass_cardcomb() if step[1] == None else step[1]
                        else:
                            _, combs2 = updateCardCombsAfterAction(combs2, cd2, step[1])
                            last_action = CardComb.pass_cardcomb() if step[1] == None else step[1]
                else:
                    # Self-data
                    self_condition = cardDictToModelList(cd1, level)
                    self_action_flags = getFlagsForActions_Flatten(combs1, last_action, level)
                    # Oppo-data
                    oppo_condition = cardDictToModelList(cd2, level)
                    oppo_action_flags = getFlagsForActions_Flatten(combs2, last_action, level)
                    # LastAction-data
                    action_type = last_action.type_index()
                    action_type_flags = [0] * 17
                    action_type_flags[action_type] = 1
                    
                    action_rank = last_action.rank
                    action_rank_flags = [0] * 15
                    action_rank_flags[action_rank] = 1
                    
                    # Prob Label
                    prob_label = [0.0] * 194
                    legal_actions = filterActions(combs1, last_action, level)
                    value = 1.0 / float(len(legal_actions))
                    for comb in legal_actions:
                        index = indexOfCombBase(comb)
                        prob_label[index] = value
                    prob_label_data = "["
                    for val in prob_label:
                        prob_label_data = prob_label_data + str(round(val, 4)) + ','
                    prob_label_data = prob_label_data[:-1]
                    prob_label_data = prob_label_data + "]"
                    
                    self_data = str(self_condition).replace(" ", '')
                    self_action_flag_data = str(self_action_flags).replace(" ", '')
                    
                    oppo_data = str(oppo_condition).replace(" ", '')
                    oppo_action_flag_data = str(oppo_action_flags).replace(" ", '')
                    
                    action_type_data = str(action_type_flags).replace(" ", '')
                    action_rank_data = str(action_rank_flags).replace(" ", '')
                    
                    val_data = str([round(float(val), 1)])
                    
                    training_data = f"{self_data}|{self_action_flag_data}|{oppo_data}|{oppo_action_flag_data}|{action_type_data}|{action_rank_data}|{level_data}|{prob_label_data}|{val_data}"
                    file.write(f"{training_data}\n")
                    data_size += 1
                    if data_size % 1000 == 0:
                        print(f"Data Size is {data_size}")
            file.close()
        return (f"{TrainPipeline.data_base_path}{file_name}", data_size)

    def policy_update(self, trainng_file : str, data_size : int = -1, repeat_time : int = 3) -> None:
        """update the policy-value net"""
        with open(trainng_file, "r") as f:
            data = f.readline()
            self_hand_card_batch = list()
            self_action_flag_batch = list()
            oppo_hand_card_batch = list()
            oppo_action_flag_batch = list()
            action_type_batch = list()
            action_rank_batch = list()
            level_batch = list()
            prob_label_batch = list()
            q_label_batch = list()
            counter = 0
            total_counter = 0
            while data is not None and data != '':
                training_data = get_training_data_from_raw_data(data)
                
                self_hand_card_batch.append(training_data[0])
                self_action_flag_batch.append(training_data[1])
                oppo_hand_card_batch.append(training_data[2])
                oppo_action_flag_batch.append(training_data[3])
                action_type_batch.append(training_data[4])
                action_rank_batch.append(training_data[5])
                level_batch.append(training_data[6])
                prob_label_batch.append(training_data[7])
                q_label_batch.append(training_data[8])
                
                counter += 1
                total_counter += 1
                
                if counter == self.batch_size or total_counter == data_size:
                    loss = self.guandan_model.train_step(
                        np.array(self_hand_card_batch, dtype=np.float32),
                        np.array(self_action_flag_batch, dtype=np.float32),
                        np.array(oppo_hand_card_batch, dtype=np.float32),
                        np.array(oppo_action_flag_batch, dtype=np.float32),
                        np.array(action_type_batch, dtype=np.float32),
                        np.array(action_rank_batch, dtype=np.float32),
                        np.array(level_batch, dtype=np.float32),
                        np.array(prob_label_batch, dtype=np.float32),
                        np.array(q_label_batch, dtype=np.float32),
                    )
                    
                    print(f"Loss = {loss.tolist()}")
                    
                    self_hand_card_batch.clear()
                    self_action_flag_batch.clear()
                    oppo_hand_card_batch.clear()
                    oppo_action_flag_batch.clear()
                    action_type_batch.clear()
                    action_rank_batch.clear()
                    level_batch.clear()
                    prob_label_batch.clear()
                    q_label_batch.clear()
                    counter = 0
                
                data = f.readline()
        f.close()
        self.guandan_model.save_model(f"{TrainPipeline.base_path}", f"{TrainPipeline.model_name_base}_{str.zfill(str(1), TrainPipeline.zfill_size)}.ckpt")

    def policy_update_csv(self, trainng_file : str, data_size : int = -1, repeat_time : int = 3) -> None:
        """update the policy-value net"""
        with open(trainng_file, "r") as f:
            data = f.readline()
            self_hand_card_batch = list()
            self_action_flag_batch = list()
            oppo_hand_card_batch = list()
            oppo_action_flag_batch = list()
            action_type_batch = list()
            action_rank_batch = list()
            level_batch = list()
            prob_label_batch = list()
            q_label_batch = list()
            counter = 0
            total_counter = 0
            while data is not None and data != '':
                training_data = get_training_data_from_raw_data(data)
                
                self_hand_card_batch.append(training_data[0])
                self_action_flag_batch.append(training_data[1])
                oppo_hand_card_batch.append(training_data[2])
                oppo_action_flag_batch.append(training_data[3])
                action_type_batch.append(training_data[4])
                action_rank_batch.append(training_data[5])
                level_batch.append(training_data[6])
                prob_label_batch.append(training_data[7])
                q_label_batch.append(training_data[8])
                
                counter += 1
                total_counter += 1
                
                if counter == self.batch_size or total_counter == data_size:
                    loss = self.guandan_model.train_step(
                        np.array(self_hand_card_batch, dtype=np.float32),
                        np.array(self_action_flag_batch, dtype=np.float32),
                        np.array(oppo_hand_card_batch, dtype=np.float32),
                        np.array(oppo_action_flag_batch, dtype=np.float32),
                        np.array(action_type_batch, dtype=np.float32),
                        np.array(action_rank_batch, dtype=np.float32),
                        np.array(level_batch, dtype=np.float32),
                        np.array(prob_label_batch, dtype=np.float32),
                        np.array(q_label_batch, dtype=np.float32),
                    )
                    
                    print(f"Loss = {loss.tolist()}")
                    
                    self_hand_card_batch.clear()
                    self_action_flag_batch.clear()
                    oppo_hand_card_batch.clear()
                    oppo_action_flag_batch.clear()
                    action_type_batch.clear()
                    action_rank_batch.clear()
                    level_batch.clear()
                    prob_label_batch.clear()
                    q_label_batch.clear()
                    counter = 0
                
                data = f.readline()
        f.close()
        self.guandan_model.save_model(f"{TrainPipeline.base_path}", f"{TrainPipeline.model_name_base}_{str.zfill(str(1), TrainPipeline.zfill_size)}.ckpt")

    def policy_evaluate(self, file_name : str, num : int) -> float:
        """
        Evaluate the trained policy by playing against the previous model
        Note: this is only for monitoring the progress of training
        """
        with open(f"{TrainPipeline.data_base_path}{file_name}", "r") as file:
            data = file.readline()
            overall_loss = 0.0
            counter = 0
            while data is not None and data != "":
                training_data = get_training_data_from_raw_data(data)
                prob = self.guandan_model.get_prob(
                    training_data[0], training_data[1], training_data[2], training_data[3],
                    training_data[4], training_data[5], training_data[6]
                    ).tolist()[0]

    def file_size(self, file_name : str) -> int:
        num = 0
        with open(file_name, "rb") as f:
            num = sum(1 for _ in f)
        return num
    
    @staticmethod
    def dummy_test() -> List[np.ndarray]:
        answer = list()
        answer.append(np.random.rand(1, 69))
        answer.append(np.random.rand(1, 193))
        answer.append(np.random.rand(1, 69))
        answer.append(np.random.rand(1, 193))
        answer.append(np.random.rand(1, 17))
        answer.append(np.random.rand(1, 15))
        answer.append(np.random.rand(1, 13))
        return answer
    
    @staticmethod
    def dummy_result() -> List[np.ndarray]:
        answer = list()
        answer.append(np.random.rand(1, 194))
        answer.append(np.random.rand(1, 1))
        return answer

    def dummy_loss_testing(self):
        data : List[np.ndarray] = list()
        
        for i in range(4):
            temp = training_pipeline.dummy_test()
            temp2 = training_pipeline.dummy_result()
            
            if i == 0:
                for j in range(7):
                    data.append(temp[j])
                for j in range(2):
                    data.append(temp2[j])
            else:
                for j in range(7):
                    data[j] = np.concatenate((data[j], temp[j]), axis=0)
                for j in range(2):
                    data[j + 7] = np.concatenate((data[j + 7], temp2[j]), axis=0)
        
        loss = self.guandan_model.get_loss(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8])
        return loss

if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    # # print(training_pipeline.file_size("/home/steventse7340/AlphaZero_Gomoku-master/collected_data/basic/basic_training-(2,2)-(1).txt"))
    # # 148677
    # # 148670
    # file_name, data_size = training_pipeline.collect_selfplay_data(4, 4, 1, 100)
    # # training_pipeline.policy_update(file_name, data_size, repeat_time=2)