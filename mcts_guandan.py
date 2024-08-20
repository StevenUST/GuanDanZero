# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy

from typing import Callable, Dict, Tuple, List
from cardcomb import CombBase
from utils import assignCombBaseToProbs, indexOfCombBase

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def reshaping_array(my_states : Tuple[np.ndarray, np.ndarray], oppo_states : Tuple[np.ndarray, np.ndarray],
        last_action : Tuple[np.ndarray, np.ndarray], level : np.ndarray)\
            -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    answer = list()
    
    t = list()
    t.append(np.reshape(my_states[0], (1, 70)))
    t.append(np.reshape(my_states[1], (1, 16, 15, 1)))
    answer.append(tuple(t))
    t.clear()
    
    t.append(np.reshape(oppo_states[0], (1, 70)))
    t.append(np.reshape(oppo_states[1], (1, 16, 15, 1)))
    answer.append(tuple(t))
    t.clear()
    
    t.append(np.reshape(last_action[0], (1, 17)))
    t.append(np.reshape(last_action[1], (1, 15)))
    answer.append(tuple(t))
    t.clear()
    
    answer.append(np.reshape(level, (1, 13)))
    
    return tuple(answer)

class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent : TreeNode = parent
        self._children : Dict[CombBase, TreeNode] = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy : Callable, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        
        self._root: TreeNode = TreeNode(None, 1.0)
        self._policy : Callable = policy
        self._c_puct: int = c_puct
        self._n_playout: int = n_playout

    def _playout(self, cards : object):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while (1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            cards.do_action(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        my_states = cards.current_states(True)
        oppo_states = cards.current_states(False)
        last_action = cards.last_move_list()
        current_level = cards.level_list()
        
        temp = reshaping_array(my_states, oppo_states, last_action, current_level)
        
        # Shape: (1, ...)
        action_probs, leaf_value = self._policy(temp[0], temp[1], temp[2], temp[3])
        action_probs_list = action_probs[0].tolist()
        action_prob_tuple_list = assignCombBaseToProbs(action_probs_list, cards.current_player_comb_indices(), cards.last_move, cards.level)
        
        # Check for end of game.
        is_end = cards.has_a_winner()
        if is_end == 0:
            node.expand(action_prob_tuple_list)
        else:
            # for end state，return the "true" leaf_value
            print("End")
            if is_end == cards.get_current_player():
                leaf_value = 1.0
            else:
                leaf_value = -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, cards : object, temp=1e-4) -> Tuple[List[int], List[CombBase], np.ndarray[float]]:
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for i in range(self._n_playout):
            state_copy = copy.deepcopy(cards)
            self._playout(state_copy)
            # if i == self._n_playout - 1:
            #     print(self._root._children.items())

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        acts_index = [indexOfCombBase(act) for act in acts]
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts_index, acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=1000):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, cards : object, temp=1e-3) -> Tuple[CombBase, np.ndarray[float]]:
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(194, dtype=np.float32)
        indices, acts, probs = self.mcts.get_move_probs(cards, temp)
        move_probs[indices] = probs

        # add Dirichlet Noise for exploration (needed for
        # self-play training)
        action = np.random.choice(
            acts,
            p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
        )
        # update the root node and reuse the search tree
        self.mcts.update_with_move(action)

        return action, move_probs

    def __str__(self):
        return "MCTS {}".format(self.player)