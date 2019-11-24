import numpy as np
import random
import torch
from torch import nn
import matplotlib as plt
from collections import namedtuple

class ReplayMemory:
     
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数
 
    def push(self, state, action, state_next, reward):
        """state, action, state_next, rewardをメモリに保存します"""
 
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す
 
        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward)
 
        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす
 
    def sample(self, batch_size):
        """batch_size分だけ、ランダムに保存内容を取り出します"""
        return random.sample(self.memory, batch_size)
 
    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

gamma = 0.99   #時間割引率
max_step = 6
num_episodes = 100
