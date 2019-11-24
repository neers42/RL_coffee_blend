# coding:utf-8
# [0]必要なライブラリのインポート
import numpy as np
import pandas as pd
import time
import keras.models
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Lambda, Input, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf

"""
[1]損失関数の定義
損失関数にhuber関数を使用します 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
"""
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

#correct_blendとの味覚値の差を計算
def reward_judgement(taste_table, action, correct_blend):
    temp = 0
    same_taste = []
    for i in range(6):
        if taste_table[action][i] == taste_table[correct_blend][i]:
            same_taste.append(i)
        temp += abs(taste_table[action][i] - taste_table[correct_blend][i])
    
    return_taste = [temp, same_taste]
    return return_taste

"""
[2]Q関数をディープラーニングのネットワークをクラスとして定義
"""
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=(5,), action_size=6, 
                 hidden_size=10, dropout = 0.3):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation = 'relu', input_shape = state_size))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_size, activation = 'relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(action_size, activation = 'softmax'))
        self.optimizer = Adam(lr=learning_rate)   # 誤差を減らす学習方法はAdam
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
        #self.model.compile(loss='mse', optimizer=self.optimizer)

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        #縦:batch_siz　横5のゼロ行列を作成、targetsも同様
        inputs = np.zeros((batch_size, 5))
        targets = np.zeros((batch_size, 6))
        #batch_size分のリストをランダムに用意する
        mini_batch = memory.sample(batch_size)
        """
        N個の訓練データのなかから一部、n個を取り出し、パラメータの更新をするのがミニバッチ学習です。
        取り出した訓練データをミニバッチと呼びます。また取り出すデータ数nをミニバッチサイズと呼びます。
        以下のように更新を行います。

        1 N個のデータからランダムにn個を取り出す。
        2 n個のデータを用いて地点x0における傾きを求める。
        3 新たな探索地点x1を傾きと学習率ηを用いて求める。
        4 新たにn個のデータを取り出して2-3の更新を行う。
        5 1-4の更新を繰り返す。
        """
        # enumerate : 要素のインデックスと要素を同時に取り出すことができる
        for i, (state_b, action_b, reward_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分ける）
                #main_QnetworkのQ値を計算する
                retmainQs = self.model.predict(state_b)[0]
                #ブレンド比率の6個のQ値をから最も高いQ値の比率をaction
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                #target : 正解データ
                target = reward_b + gamma * targetQN.model.predict(state_b)[0][next_action]
            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            targets[i][action_b] = target               # 教師信号
            print(target)

        self.model.fit(inputs, targets, epochs=50, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

"""
[3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
"""
class Memory:
    def __init__(self, max_size=1000):
        #最大バッファーメモリサイズを1000に設定
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        #経験値をバッファーメモリに保存
        self.buffer.append(experience)

    def sample(self, batch_size):
        #np.arange：等差数列を配列として出力する関数
        #np.random.choice：要素をランダムに返す
        """
        np.arange(stop))は,0 ≦n<stopで間隔は1の等差数列の
        配列リストを出力
        np.random.choiceでは第一引数に配列、第二引数に出力する配列のサイズ
        第三引数に配列に値を重複させるかかしないかを設定する
        （replase=falseと指定することで重複するのを防いでいる)
        """
        #idxにはバッチサイズの重複せずに
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        #バッファーメモリのサイズの大きさを返す関数
        return len(self.buffer)

"""
[4]ユーザーの状態に応じて、行動を決定するクラス
アドバイスいただき、引数にtargetQNを使用していたのをmainQNに修正しました
"""
class Actor:
    def get_action(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        #epsilonの値を最初は１（探索中心）その後は経験値を基にactionを決定していく
        epsilon = 0.1 + 0.9 / (1.0 + episode / 5.0)
        
        #np.random.uniform(0,1)で0以上1未満の乱数を1つ作成しepsilonと比較
        if epsilon <= np.random.uniform(0, 1): 
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            #それ以外の場合はランダムに行動する
            action = np.random.choice([0, 1, 2, 3, 4, 5])  # ランダムに行動する

        return action

"""
[5] メイン関数開始----------------------------------------------------
[5.1] 初期設定--------------------------------------------------------
"""
DQN_MODE = 1    # 1がDQN、0がDDQNです
LENDER_MODE = 1 # 0は学習後も描画なし、1は学習終了後に描画する
num_episodes = 100  # 総試行回数
max_number_of_steps = 10000  # 1試行のstep数
goal_average_reward = 195  # この報酬を超えると学習終了
num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
gamma = 0.99    # 割引係数
islearned = 0  # 学習が終わったフラグ
isrender = 0  # 描画フラグ
# ---
hidden_size = 64               # Q-networkの隠れ層のニューロンの数
learning_rate = 0.00001         # Q-networkの学習係数
memory_size = 50000            # バッファーメモリの大きさ
batch_size = 16                # Q-networkを更新するバッチの大記載
number = 1
# [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
memory = Memory(max_size=memory_size)
actor = Actor()

#csv読み取り部分
#data:実験データ
lst1 = pd.read_csv("test_data.csv").values.tolist()
#name:ブレンド味覚値
lst2 = pd.read_csv("blend_table.csv").values.tolist()
data = np.array(lst1)
#taste_table：味覚値表
taste_table = np.array(lst2)

 
# [5.3]メインルーチン--------------------------------------------------------
for episode in range(num_episodes + 1):  # 試行数分繰り返す 
    # 1step目は適当な行動をとる
    done = False
    heart_rate = data[episode][0]       #heart_rate : 心拍数
    drink_time = data[episode][2]       #drink_time : 喫飲時間帯
    sleep_time = data[episode][3]       #sleep_time : 睡眠時間
    thi =  data[episode][5]             #thi : 不快指数
    fatigue = data[episode][7]          #fatigue : 肉体的疲労度
    correct_blend = data[episode][9]    # 正解データ
    incorrect_blend = data[episode][10]  #actionデータ
    state = [heart_rate, drink_time, sleep_time, thi, fatigue]
    array_state = np.array(state)     #state配列をnumpy配列に変換
    state = np.reshape(state, [1, 5])   # list型のstateを、1行5列の行列に変換
    episode_reward = 0
    rewards = []
    

    # targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
    # ↓
    targetQN.model.set_weights(mainQN.model.get_weights())

    for step in range(max_number_of_steps):  # 1試行のループ
        action = actor.get_action(state, episode, mainQN)   # 時刻tでの行動を決定する
        judge = reward_judgement(taste_table, action, correct_blend)
        # 報酬を設定し、与える
        if action == correct_blend:   #blend:csvから読み取ったブレンド比率
            reward = 10
            done = True
        else:
            #reward_judgement関数で味覚値の差を計算
            if judge[0] > 0.6:
                reward = -0.5
            elif action == incorrect_blend:
                reward = -1
            elif len(judge[1]) != 0:
                length = len(judge[1])
                reward = 0.5 * length
            else:
                reward = 0
        
        rewards.append(reward)
        episode_reward += 1    #合計報酬を更新
        memory.add((state, action, reward))   #メモリの更新（ state, action, reward をタプル型でメモリに保存）

        # Qネットワークの重みを学習・更新する replay
        if (memory.len() > batch_size*number) and not islearned:
            mainQN.replay(memory, batch_size, gamma, targetQN)
            number += 1

        if DQN_MODE:
        # targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
            targetQN.model.set_weights(mainQN.model.get_weights())

        # 1施行終了時の処理
        if done:
            score = sum(rewards)
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
            print('%d Episode finished after %f time steps / mean %f  / score : %f' % (episode, step + 1, total_reward_vec.mean(), score))
            break
        elif(step == max_number_of_steps):
            print('%d Episode finished after %f time steps / mean %f' % (episode, step + 1, total_reward_vec.mean()))
        else:
            pass
            
    # 複数施行の平均報酬で終了を判断
    if episode == 100:
        print('agent train successfuly!')
mainQN.model.save('al15042_model.h5')  # h5モデルファイル作成
print("model save successfuly!")
