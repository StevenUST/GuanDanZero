import tensorflow as tf

from typing import List
from numpy import ndarray as nplist

class GuandanNetBase():
    
    def __init__(self, learning_rate : float = 0.01) -> None:
        self.lr = learning_rate
        self.saver = tf.compat.v1.train.Saver()
        self.session = tf.compat.v1.Session()
        
        init = tf.compat.v1.global_variables_initializer()
        self.session.run(init)

    def update_learning_rate(self, lr : float) -> None:
        assert lr > 0.0
        self.lr = lr

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)

class GuandanNetForTwo_M1(GuandanNetBase):

    def __init__(self, model_file=None):
        super().__init__()
        tf.compat.v1.disable_eager_execution()
        # (smallest number card to biggest number card, then SB and HR)
        self.input_my_hand_card1 = tf.compat.v1.placeholder(tf.float32, shape=[None, 15, 1])
        # (smallest number card to biggest number card, without SB and HR)
        self.input_my_hand_card2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 13, 1])
        # (From A to K then finally A, but with suits)
        self.input_my_hand_card3 = tf.compat.v1.placeholder(tf.float32, shape=[None, 14, 4])
        # (hand_card_num (>=20, 19-12, 11-7, 6, 5, 4, 3, 2, 1), wild_card_num(0, 1, 2), can_win_in_one_shot(cannot, can))
        self.input_my_other_states = tf.compat.v1.placeholder(tf.float32, shape=[None, 14])
        
        # (smallest number card to biggest number card, then SB and HR)
        self.input_oppo_hand_card1 = tf.compat.v1.placeholder(tf.float32, shape=[None, 15, 1])
        # (smallest number card to biggest number card, without SB and HR)
        self.input_oppo_hand_card2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 13, 1])
        # (From A to K then finally A, but with suits)
        self.input_oppo_hand_card3 = tf.compat.v1.placeholder(tf.float32, shape=[None, 14, 4])
        # (hand_card_num (>=20, 19-12, 11-7, 6, 5, 4, 3, 2, 1), wild_card_num(0, 1, 2), can_win_in_one_shot(cannot, can))
        self.input_oppo_other_states = tf.compat.v1.placeholder(tf.float32, shape=[None, 14])
        
        # (last action type (PASS, Single, Pair, Trip, ThreeWithTwo, TwoTrips, ThreePairs, Straight, Bomb, StraightFlush), 
        # last action ranking (0 to level card, then SB and HR; For PASS, it is 0))
        self.universal_states = tf.compat.v1.placeholder(tf.float32, shape=[None, 26])
        
        self.conv_hc1 = tf.keras.layers.Conv1D(filters=20, kernel_size=[1], padding="same", activation=tf.nn.relu)(self.input_my_hand_card1)
        self.conv_hc2 = tf.keras.layers.Conv1D(filters=20, kernel_size=[1], padding="same", activation=tf.nn.relu)(self.input_my_hand_card2)
        self.conv_hc3_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=[2], padding="same", activation=tf.nn.relu)(self.input_my_hand_card3)
        self.conv_hc3_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=[3], padding="same", activation=tf.nn.relu)(self.input_my_hand_card3)
        self.conv_hc3_3 = tf.keras.layers.Conv1D(filters=20, kernel_size=[5], padding="same", activation=tf.nn.relu)(self.input_my_hand_card3)
        
        self.conv_oppo_hc1 = tf.keras.layers.Conv1D(filters=20, kernel_size=[1], padding="same", activation=tf.nn.relu)(self.input_oppo_hand_card1)
        self.conv_oppo_hc2 = tf.keras.layers.Conv1D(filters=20, kernel_size=[1], padding="same", activation=tf.nn.relu)(self.input_oppo_hand_card2)
        self.conv_oppo_hc3_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=[2], padding="same", activation=tf.nn.relu)(self.input_oppo_hand_card3)
        self.conv_oppo_hc3_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=[3], padding="same", activation=tf.nn.relu)(self.input_oppo_hand_card3)
        self.conv_oppo_hc3_3 = tf.keras.layers.Conv1D(filters=20, kernel_size=[5], padding="same", activation=tf.nn.relu)(self.input_oppo_hand_card3)
        
        self.flatten_1 = tf.keras.layers.Flatten()(self.conv_hc1)
        self.flatten_2 = tf.keras.layers.Flatten()(self.conv_hc2)
        self.flatten_3_1 = tf.keras.layers.Flatten()(self.conv_hc3_1)
        self.flatten_3_2 = tf.keras.layers.Flatten()(self.conv_hc3_2)
        self.flatten_3_3 = tf.keras.layers.Flatten()(self.conv_hc3_3)
        
        self.flatten_oppo_1 = tf.keras.layers.Flatten()(self.conv_oppo_hc1)
        self.flatten_oppo_2 = tf.keras.layers.Flatten()(self.conv_oppo_hc2)
        self.flatten_oppo_3_1 = tf.keras.layers.Flatten()(self.conv_oppo_hc3_1)
        self.flatten_oppo_3_2 = tf.keras.layers.Flatten()(self.conv_oppo_hc3_2)
        self.flatten_oppo_3_3 = tf.keras.layers.Flatten()(self.conv_oppo_hc3_3)
        
        self.concatenated_layer = tf.keras.layers.Concatenate()([self.flatten_1, self.flatten_2, self.flatten_3_1, self.flatten_3_2, self.flatten_3_3])
        self.oppo_concatenated_layer = tf.keras.layers.Concatenate()([self.flatten_oppo_1, self.flatten_oppo_2, self.flatten_oppo_3_1, self.flatten_oppo_3_2, self.flatten_oppo_3_3])
        
        self.total_concatenated_layer = tf.keras.layers.Concatenate()([self.concatenated_layer, self.oppo_concatenated_layer, self.universal_states])
        self.layer1 = tf.keras.layers.Dense(units=2048, activation=tf.keras.layers.LeakyReLU(alpha=0.02))(self.total_concatenated_layer)
        self.layer2 = tf.keras.layers.Dense(units=1024, activation=tf.keras.layers.LeakyReLU(alpha=0.02))(self.layer1)
        self.layer3 = tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(alpha=0.02))(self.layer2)
        self.layer4 = tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(alpha=0.02))(self.layer3)
        self.layer5 = tf.keras.layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.02))(self.layer4)
        
        # (PASS, Single(15), Pair(15), Trip(13), ThreeWithTwo(13), TwoTrips(13), ThreePairs(12), Straight(10), Bomb(91), StraightFlush(10), JOKER BOMB(1))
        self.policy_prob = tf.keras.layers.Dense(units=194)(self.layer5)
        self.policy_prob_label = tf.compat.v1.placeholder(tf.float32, shape=[None, 194])
        self.loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.policy_prob, self.policy_prob_label), 1)))
        self.learning_rate = tf.compat.v1.placeholder(tf.float32)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # # Make a session
        # self.session = tf.compat.v1.Session()

        # Initialize variables
        # init = tf.compat.v1.global_variables_initializer()
        # self.session.run(init)

        # # For saving and restoring
        # self.saver = tf.compat.v1.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    
    def get_prob(self, my_states : List, oppo_states : List, u_states : List) -> nplist:
        log_act_prob = self.session.run(
            [self.policy_prob],
            feed_dict={self.input_my_hand_card1 : my_states[0],
                       self.input_my_hand_card2 : my_states[1],
                       self.input_my_hand_card3 : my_states[2],
                       self.input_my_other_states : my_states[3],
                       self.input_oppo_hand_card1 : oppo_states[0],
                       self.input_oppo_hand_card2 : oppo_states[1],
                       self.input_oppo_hand_card3 : oppo_states[2],
                       self.input_oppo_other_states : oppo_states[3],
                       self.universal_states : u_states}
        )
        return log_act_prob

class GuandanNetForTwo_M2(GuandanNetBase):
    
    def __init__(self, model_file = None) -> None:
        super().__init__()