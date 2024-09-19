from typing import Final

from guandan_net_tensorflow import GuandanNetForTwo
from utils2 import get_training_data_from_raw_data, get_indices_of_expected_actions, get_indices_of_predicted_actions

import numpy as np

class LossEvaluator:
    
    base_path : Final[str] = "/home/steventse7340/AlphaZero_Gomoku-master/saved_model/basic/"
    meta_name : Final[str] = "test1.ckpt"
    
    def __init__(self) -> None:
        self.model = GuandanNetForTwo()
        self.model.restore_model(LossEvaluator.base_path, LossEvaluator.meta_name)
    
    def loss_evaluation_txt(self, file_name : str) -> None:
        test_num = 100
        n = 0
        
        with open(file_name, "r") as f:
            data = f.readline()
            
            num = 0
            prob_fail_num = 0
            val_loss = 0.0
            fail_example_num = 0
            
            while data is not None and data != "" and n < test_num:
                training_data = get_training_data_from_raw_data(data)
                
                probs, val = self.model.get_prob(
                    np.reshape(training_data[0], (1, 69)),
                    np.reshape(training_data[1], (1, 193)),
                    np.reshape(training_data[2], (1, 69)),
                    np.reshape(training_data[3], (1, 193)),
                    np.reshape(training_data[4], (1, 17)),
                    np.reshape(training_data[5], (1, 15)),
                    np.reshape(training_data[6], (1, 13))
                    )
                
                probs_label = training_data[7]
                val_label = training_data[8].tolist()[0]
                
                expected_indices = get_indices_of_expected_actions(probs_label)
                predicted_indices = get_indices_of_predicted_actions(np.reshape(probs, (194, )), len(expected_indices))
                predicted_val = val.tolist()[0][0]
                
                prob_fail = False
                
                if len(expected_indices) == 1 and predicted_indices[0] != expected_indices[0]:
                    prob_fail = True
                elif not (predicted_indices[0] in expected_indices):
                    prob_fail = True
                
                print(f"predicted = {predicted_val}, expected = {val_label}")
                val_loss += np.sqrt((predicted_val - val_label) ** 2)
                
                if prob_fail:
                    fail_example_num += 1
                    prob_fail_num += 1
                
                num += 1
                n += 1
                data = f.readline()
            
            print(f"Fail num = {fail_example_num}")
            print(f"Prob Fail num = {prob_fail_num}")
            print(f"Average val loss = {val_loss / num}")
            
            f.close()

if __name__ == "__main__":
    loss_evaluator = LossEvaluator()
    loss_evaluator.loss_evaluation_txt("./collected_data/basic/basic_training-(4,4)-(1).txt")