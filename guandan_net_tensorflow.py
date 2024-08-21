import tensorflow as tf

from typing import Tuple, Optional
from numpy import ndarray as nplist, exp as npexp, reshape

tf.compat.v1.disable_eager_execution()

class GuandanNetForTwo:

    def __init__(self, lr : float = 0.01, max_file_num : int = 4) -> None:
        self.lr = lr
        # [
            # S2, S3, S4, ... SA, 
            # H2, H3, H4, ... HA,
            # C2, C3, C4, ... CA,
            # D2, D3, D4, ... DA,
            # SB, HR,
            # #2, #3, #4, ... #A,
            # #WC, #ALL, #STAGE
        # ]
        self.input_my_hand_card = tf.compat.v1.placeholder(tf.float32, shape=[None, 70])
        # (Flag for each action)
        # (Remark: '-' implies always 0 because such action does not exist)
        # (Given the last action is A. If the action can be played freely, value is 1; If the action can be played after action A, value is 2)
        # [
            # Single        [2, 3, 4, ... A, B, R]
            # Pair          [2, 3, 4, ... A, B, R]
            # Trip          [2, 3, 4, ... A, -, -]
            # 3+2           [2, 3, 4, ... A, -, -]
            # TwoTrips      [A, 2, 3, ... K, -, -]
            # ThreePairs    [A, 2, 3, ... Q, -, -, -]
            # Straight      [A, 2, 3, ... T, -, -, -, -, -]
            # Bomb(4)       [2, 3, 4, ... A, -, -]
            # Bomb(5)       [2, 3, 4, ... A, -, -]
            # SF            [A, 2, 3, ... T, -, -, -, -, -]
            # Bomb(6)       [2, 3, 4, ... A, -, -]
            # ...
            # Bomb(9)       [2, 3, 4, ... A, -, -]
            # Bomb(10)      [2, 3, 4, ... A, -, -]
            # JOKERBOMB(Remark: If JOKERBOMB exists, the vector is [2, 2, 2, ..., 2] with length of 15. Else it is all zero)
        # ]
        self.input_my_action_flags = tf.compat.v1.placeholder(tf.float32, shape=[None, 16, 15, 1])
        # [
            # S2, S3, S4, ... SA, 
            # H2, H3, H4, ... HA,
            # C2, C3, C4, ... CA,
            # D2, D3, D4, ... DA,
            # SB, HR,
            # #2, #3, #4, ... #A,
            # #WC, #ALL, #STAGE
        # ]
        self.input_oppo_hand_card = tf.compat.v1.placeholder(tf.float32, shape=[None, 70])
        # (Flag for each action)
        # (Remark: '-' implies always 0 because such action does not exist)
        # (Given the last action is A. If the action can be played freely, value is 1; If the action can be played after action A, value is 2)
        # [
            # Single        [2, 3, 4, ... A, B, R]
            # Pair          [2, 3, 4, ... A, B, R]
            # Trip          [2, 3, 4, ... A, -, -]
            # 3+2           [2, 3, 4, ... A, -, -]
            # TwoTrips      [A, 2, 3, ... K, -, -]
            # ThreePairs    [A, 2, 3, ... Q, -, -, -]
            # Straight      [A, 2, 3, ... T, -, -, -, -, -]
            # Bomb(4)       [2, 3, 4, ... A, -, -]
            # Bomb(5)       [2, 3, 4, ... A, -, -]
            # SF            [A, 2, 3, ... T, -, -, -, -, -]
            # Bomb(6)       [2, 3, 4, ... A, -, -]
            # ...
            # Bomb(9)       [2, 3, 4, ... A, -, -]
            # Bomb(10)      [2, 3, 4, ... A, -, -]
            # JOKERBOMB(Remark: If JOKERBOMB exists, the vector is [2, 2, 2, ..., 2] with length of 15. Else it is all zero)
        # ]
        self.input_oppo_action_flags = tf.compat.v1.placeholder(tf.float32, shape=[None, 16, 15, 1])
        
        # (PASS, Single, Pair, Trip, ThreePairs, TwoTrips, ThreeWithTwo, Straight, StraightFlush, Bomb(4-10), JOKERBOMB)
        self.last_move_type = tf.compat.v1.placeholder(tf.float32, shape=[None, 17])
        # (From 2 to A, then SB and HR)
        self.last_move_rank = tf.compat.v1.placeholder(tf.float32, shape=[None, 15])
        # Level (From 2 to A)
        self.current_level = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
        
        self.my_conv_layer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu)(self.input_my_action_flags)
        self.my_max_pool_layer1 = tf.keras.layers.MaxPool2D(strides=1)(self.my_conv_layer1)
        self.my_conv_layer2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.keras.activations.relu)(self.my_max_pool_layer1)
        self.my_max_pool_layer2 = tf.keras.layers.MaxPool2D(strides=1)(self.my_conv_layer2)
        self.my_conv_layer3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.keras.activations.relu)(self.my_conv_layer2)
        self.my_max_pool_layer3 = tf.keras.layers.MaxPool2D(strides=1)(self.my_conv_layer3)
        self.my_flatten_layer = tf.keras.layers.Flatten()(self.my_max_pool_layer3)
        
        self.oppo_conv_layer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu)(self.input_oppo_action_flags)
        self.oppo_max_pool_layer1 = tf.keras.layers.MaxPool2D(strides=1)(self.oppo_conv_layer1)
        self.oppo_conv_layer2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.keras.activations.relu)(self.oppo_max_pool_layer1)
        self.oppo_max_pool_layer2 = tf.keras.layers.MaxPool2D(strides=1)(self.oppo_conv_layer2)
        self.oppo_conv_layer3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.keras.activations.relu)(self.oppo_conv_layer2)
        self.oppo_max_pool_layer3 = tf.keras.layers.MaxPool2D(strides=1)(self.oppo_conv_layer3)
        self.oppo_flatten_layer = tf.keras.layers.Flatten()(self.oppo_max_pool_layer3)
        
        self.layer1 = tf.keras.layers.Concatenate()([self.input_my_hand_card, self.my_flatten_layer,
                                                         self.input_oppo_hand_card, self.oppo_flatten_layer,
                                                         self.last_move_type, self.last_move_rank, self.current_level])
        
        self.layer2 = tf.keras.layers.Dense(units=1024, activation=tf.keras.activations.relu)(self.layer1)
        self.layer3 = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu)(self.layer2)
        self.layer4 = tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu)(self.layer3)
        
        self.q_layer2 = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu)(self.layer1)
        self.q_layer3 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)(self.q_layer2)
        self.q_layer4 = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(self.q_layer3)
        
        self.policy_prob = tf.keras.layers.Dense(units=194, activation=tf.nn.log_softmax)(self.layer4)
        self.q_value = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)(self.q_layer4)
        
        self.policy_prob_label = tf.compat.v1.placeholder(tf.float32, shape=[None, 194])
        self.q_label = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.policy_prob, self.policy_prob_label), 1)))
        self.q_loss = tf.losses.mean_squared_error(self.q_label, self.q_value)
        self.loss = self.policy_loss + self.q_loss
        self.learning_rate = tf.compat.v1.placeholder(tf.float32)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # # Make a session
        self.session = tf.compat.v1.Session()

        # Initialize variables
        self.session.run(tf.compat.v1.global_variables_initializer())

        # For saving and restoring
        self.saver = tf.compat.v1.train.Saver(max_to_keep=max_file_num)
    
    def update_learning_rate(self, lr : float) -> None:
        assert lr > 0.0
        self.lr = lr

    def save_model(self, model_path : str):
        self.saver.save(self.session, model_path)

    def restore_model(self, base_path : str, meta_file : str):
        new_saver = tf.compat.v1.train.import_meta_graph(f"{base_path}{meta_file}")
        new_saver.restore(self.session, tf.train.latest_checkpoint(base_path))
    
    def policy_value_function(self, my_states : Tuple[nplist, nplist], oppo_states : Tuple[nplist, nplist], last_action : Tuple[nplist, nplist], level : nplist) -> nplist:
        return self.get_prob(my_states[0], my_states[1], oppo_states[0], oppo_states[1], last_action[0], last_action[1], level)
    
    def get_prob(self, my_state1 : nplist, my_state2 : nplist,\
                        oppo_state1 : nplist, oppo_state2 : nplist,\
                            last_action1 : nplist, last_action2 : nplist, level : nplist) -> Tuple[nplist, int]:
        log_act_prob, value = self.session.run(
            [self.policy_prob, self.q_value],
            feed_dict={self.input_my_hand_card : my_state1,
                       self.input_my_action_flags : my_state2,
                       self.input_oppo_hand_card : oppo_state1,
                       self.input_oppo_action_flags : oppo_state2,
                       self.last_move_type : last_action1,
                       self.last_move_rank : last_action2,
                       self.current_level : level
                       }
        )
        act_probs = npexp(log_act_prob)
        return act_probs, value
    
    def get_value(self, my_state1 : nplist, my_state2 : nplist,\
                        oppo_state1 : nplist, oppo_state2 : nplist,\
                            last_action1 : nplist, last_action2 : nplist, level : nplist) -> nplist:
        v = self.session.run(
            [self.q_value],
            feed_dict={self.input_my_hand_card : my_state1,
                       self.input_my_action_flags : my_state2,
                       self.input_oppo_hand_card : oppo_state1,
                       self.input_oppo_action_flags : oppo_state2,
                       self.last_move_type : last_action1,
                       self.last_move_rank : last_action2,
                       self.current_level : level
                       }
        )
        return v
    
    def train_step(self, my_state1 : nplist, my_state2 : nplist,
                            oppo_state1 : nplist, oppo_state2 : nplist,
                            last_action1 : nplist, last_action2 : nplist, level : nplist,
                            prob_label : nplist, final_score : nplist) -> nplist:
        # """perform a training step"""
        loss, _ = self.session.run(
            [self.loss, self.optimizer],
            feed_dict={self.input_my_hand_card : my_state1,
                       self.input_my_action_flags : my_state2,
                       self.input_oppo_hand_card : oppo_state1,
                       self.input_oppo_action_flags : oppo_state2,
                       self.last_move_type : last_action1,
                       self.last_move_rank : last_action2,
                       self.current_level : level,
                       self.policy_prob_label : prob_label,
                       self.q_label : final_score,
                       self.learning_rate : self.lr}
        )
        return loss