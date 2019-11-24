import os
import sys
from collections import deque
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.optimizers import Adam
from keras.models import clone_model
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import pandas as pd

class Agent(object):
    def __init__(self, learning_rate = 0.01, state_size = (5,), 
                 action_size = 6, hidden_size = 24, dropout = 0.3,
                 ):
        model = Sequential()
        model.add(Dense(hidden_size, input_shape = state_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(action_size))
        model.add(Activation('softmax'))
        self.model = model

    def evaluate(self, state, model = None):
        _model = model if model else self.model
        _state = np.expand_dims(state, axis = 0)
        return _model.predict(_state)[0]
    
    def get_action(self, state, epoch):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+epoch)

        if epsilon <= np.random.uniform(0, 1):
            Q = self.evaluate(state)
            action = np.argmax(Q)  # 最大の報酬を返す行動を選択する

        else:
            action = np.random.choice([0, 1])  # ランダムに行動する

        return action

class Data:

    def __init__(self, heart_rate = 0, drink_time = 0 ,
                 sleep_time = 0, thi = 0, fatigue = 0, correct_blend = 0,
                 incorrect_blend = 0):
        self.heart_rate = heart_rate
        self.drink_time = drink_time
        self.sleep_time = sleep_time
        self.thi = thi
        self.fatigue = fatigue
        self.correct_blend = correct_blend
        self.incorrect_blend = correct_blend

    def get_state(self, epoch, data):
        self.heart_rate = data[epoch][0]
        self.drink_time = data[epoch][2]
        self.sleep_time = data[epoch][3]
        self.thi = data[epoch][4]
        self.fatigue = data[epoch][7]
        self.correct_blend = data[epoch][9]
        self.incorrect_blend = data[epoch][10]
        state = [self.heart_rate, self.drink_time, 
                 self.sleep_time, self.thi, self.fatigue]
        blend = [self.correct_blend, self.incorrect_blend]
        return [state, blend] 

class Trainer(object):
    
    def __init__(self, agent, optimizer, model_dir = ""):
        self.agent = agent
        self.experience = []
        self._target_model = clone_model(self.agent.model)
        self.model_dir = model_dir
        if not self.model_dir:
            self.model_dir = os.path.join(os.path.dirname(__file__), "model")
            if not os.path.isdir(self.model_dir):
                os.mkdir(self.model_dir)
        self.agent.model.compile(loss = "mse", optimizer = optimizer)
        self.callback = TensorBoard(self.model_dir)
        self.callback.set_model(self.agent.model)
    
    def get_batch(self, batch_size, gamma):
        batch_indices = np.random.randint(
            low = 0, high = len(self.experience), size = batch_size)
        # X : 訓練データ y:教師データ
        X = np.zeros((batch_size,) + self.agent.INPUT_SHAPE)
        y = np.zeros((batch_size, self.agent.num_actions))
        for i, b_i in enumerate(batch_indices):
            s, a, r, next_s, done = self.experience[b_i]
            X[i] = s
            y[i] = self.agent.evaluate(s)
            # future reward
            Q_sa = np.max(self.agent.evaluate(next_s,
                                              model=self._target_model))
            #報酬付与
            if done:
                y[i, a] = r
            else:
                y[i, a] = r + gamma * Q_sa
        return X, y

    def reward_judgement(self, taste_table, action, correct_blend):
        temp = 0
        for i in range(6):
            temp += abs(taste_table[action][i] - taste_table[correct_blend][i])
        return temp

    def write_log(self, index, loss, score):
        for name, value in zip(("loss", "score"), (loss, score)):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, index)
            self.callback.writer.flush()
    
    def train(self, data, taste_table, gamma = 0.99,
                initial_epsilon = 0.1, final_epsilon = 0.0001,
                memory_size = 50000,
                training_epochs = 100, batch_size = 32, render = True):

        self.experience = deque(maxlen = memory_size)
        epsilon = initial_epsilon
        model_path = os.path.join(self.model_dir, "agent_network_al15042.h5")
        epochs = training_epochs
        for epoch in range(epochs):
            loss = 0.0
            rewards = []
            sample_data = Data()
            r_state = sample_data.get_state(epoch, data)
            state = r_state[0]
            blend = r_state[1]
            done = False
            is_training = True if epoch == 100 else False

            while not is_training:
                if not done:
                    action = self.agent.get_action(state, epoch)
                else:
                    action = self.agent.get_action(state, epoch)

                if action == blend[0]:
                    reward = 1
                    done = True
                
                else:
                    if self.reward_judgement(taste_table, action, blend[0]) > 0.4 or action == blend[1]:
                        reward = -1
                    else:
                        reward = 0

                self.experience.append((state, action, reward, done))
                rewards.append(reward)

                if is_training:
                    X,y = self.get_batch(batch_size, gamma)
                    loss += self.agent.train_on_batch(X, y)
                loss = loss/ len(rewards)
                score = sum(rewards)

                if epsilon > final_epsilon:
                    epsilon -= (initial_epsilon - final_epsilon) / epochs

                print("Epoch {:04d}/{:d} | Loss {:.5f} | Score: {} | e={:.4f} train={}".format(epoch + 1, epochs, loss, score, epsilon, is_training))
                    
                if epoch % 100 == 0:
                    self.agent.model.save(model_path, overwrite=True)

        self.agent.model.save(model_path, overwrite=True)    

def main():
    lst1 = pd.read_csv("test_data.csv").values.tolist()
    lst2 = pd.read_csv("blend_table.csv").values.tolist()
    data = np.array(lst1)
    taste_table = np.array(lst2)
    agent = Agent()
    trainer = Trainer(agent, Adam(lr=1e-6))
    trainer.train(data, taste_table)

if __name__ == "__main__":
    main()

            
