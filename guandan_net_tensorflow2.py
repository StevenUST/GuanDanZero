import tensorflow as tf

from typing import List, Tuple, Optional
from numpy import ndarray as nplist

tf.compat.v1.disable_eager_execution()

class GuandanNetBase():
    
    def __init__(self, learning_rate : float = 0.01) -> None:
        self.lr = learning_rate
        self.saver : Optional[tf.compat.v1.train.Saver] = None
        self.session : Optional[tf.compat.v1.Session] = None

    def update_learning_rate(self, lr : float) -> None:
        assert lr > 0.0
        self.lr = lr

    def save_model(self, model_path):
        if self.saver is not None:
            self.saver.save(self.session, model_path)
        else:
            raise RuntimeError("The saver is not initialized yet.")

    def restore_model(self, model_path):
        if self.saver is not None:
            self.saver.restore(self.session, model_path)
            raise RuntimeError("The saver is not initialized yet.")

class GuandanNetForTwo_M1(GuandanNetBase):

    def __init__(self, lr : float = 0.01, model_file=None):
        super().__init__(learning_rate=lr)
        # (Self hand cards. Starts from 2 to A, then SB and HR)
        self.input_my_hand_card = tf.compat.v1.placeholder(tf.float32, shape=[None, 15])
        # (The StraightFlush that can be played. From A to T, each slot is ranged from 0 to 4)
        self.input_my_straight_flush_flags = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
        # (Number of Wild Card, From 0 to 2)
        self.input_my_other_states = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
        
        # (Self hand cards. Starts from 2 to A, then SB and HR)
        self.input_oppo_hand_card = tf.compat.v1.placeholder(tf.float32, shape=[None, 15])
        # (The StraightFlush that can be played. From A to T, each slot is ranged from 0 to 4)
        self.input_oppo_straight_flush_flags = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
        # (Number of Wild Card, From 0 to 2)
        self.input_oppo_other_states = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
        
        # (PASS, Single, Pair, Trip, ThreePairs, TwoTrips, ThreeWithTwo, Straight, StraightFlush, Bomb(4-10), JOKERBOMB)
        self.last_move_type = tf.compat.v1.placeholder(tf.float32, shape=[None, 17])
        # (From 2 to A, then SB and HR)
        self.last_move_rank = tf.compat.v1.placeholder(tf.float32, shape=[None, 15])
        # Level (From 2 to A)
        self.current_level = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
        
        self.layer1 = tf.keras.layers.Concatenate()([self.input_my_hand_card, self.input_my_straight_flush_flags, self.input_my_other_states,
                                                         self.input_oppo_hand_card, self.input_oppo_straight_flush_flags, self.input_oppo_other_states,
                                                         self.last_move_type, self.last_move_rank, self.current_level
                                                        ])
        
        self.layer2 = tf.keras.layers.Dense(units=128, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(self.layer1)
        self.layer3 = tf.keras.layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(self.layer2)
        self.layer4 = tf.keras.layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(self.layer3)
        self.layer5 = tf.keras.layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(self.layer4)
        self.layer6 = tf.keras.layers.Dense(units=128, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(self.layer5)
        
        self.policy_prob = tf.keras.layers.Dense(units=194)(self.layer6)
        self.q_value = tf.keras.layers.Dense(units=1)(self.layer6)
        
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
        init = tf.compat.v1.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.compat.v1.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    
    def get_prob(self, my_states : List, oppo_states : List, last_action : List, level : List) -> nplist:
        log_act_prob = self.session.run(
            [self.policy_prob],
            feed_dict={self.input_my_hand_card : my_states[0],
                       self.input_my_straight_flush_flags : my_states[1],
                       self.input_my_other_states : my_states[2],
                       self.input_oppo_hand_card : oppo_states[0],
                       self.input_oppo_straight_flush_flags : oppo_states[1],
                       self.input_oppo_other_states : oppo_states[2],
                       self.last_move_type : last_action[0],
                       self.last_move_rank : last_action[1],
                       self.current_level : level
                       }
        )
        return log_act_prob
    
    def train_step(self, my_states : List, oppo_states : List, last_action : List, level : List, prob_label : List, final_score : List):
        # """perform a training step"""
        loss, _ = self.session.run(
            [self.loss, self.optimizer],
            feed_dict={self.input_my_hand_card : my_states[0],
                       self.input_my_straight_flush_flags : my_states[1],
                       self.input_my_other_states : my_states[2],
                       self.input_oppo_hand_card : oppo_states[0],
                       self.input_oppo_straight_flush_flags : oppo_states[1],
                       self.input_oppo_other_states : oppo_states[2],
                       self.last_move_type : last_action[0],
                       self.last_move_rank : last_action[1],
                       self.current_level : level,
                       self.policy_prob_label : prob_label,
                       self.q_label : final_score,
                       self.learning_rate : self.lr}
        )
        return loss

pass