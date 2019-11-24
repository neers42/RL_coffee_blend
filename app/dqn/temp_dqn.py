import numpy as np
from collections import deque

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
        #idxにはバッチサイズの重複せずに
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        #バッファーメモリのサイズの大きさを返す関数
        return len(self.buffer)

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
hidden_size = 32               # Q-networkの隠れ層のニューロンの数
learning_rate = 0.00001         # Q-networkの学習係数
memory_size = 50000            # バッファーメモリの大きさ
batch_size = 16                # Q-networkを更新するバッチの大記載
number = 1

memory = Memory()
inputs = np.zeros((batch_size, 5))
targets = np.zeros((batch_size, 6))
#batch_size分のリストをランダムに用意する
mini_batch = memory.sample(batch_size)
# enumerate : 要素のインデックスと要素を同時に取り出すことができる
for i, (state_b, action_b, reward_b) in enumerate(mini_batch):
    inputs[i:i + 1] = state_b
    target = reward_b
    print(state_b)