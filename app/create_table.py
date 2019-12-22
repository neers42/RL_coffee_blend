# coding:utf-8-sig                                                                                                                                                              
# [0]ライブラリのインポート                                                                                                                                                         
import numpy as np
import time
import pandas as pd
from time import sleep
from tqdm import tqdm

print("Please Input User ID!!")
user_id = str(input("User_ID : "))
Q_table_path = "./EDB/Q_Table/" + user_id + "_EDB_Q_Table.csv"
EDB_path = "./EDB/" + user_id + "_EDB.csv"
test_path = "./test_data/" + user_id + "_data.csv"

#実験データ
lst2 = pd.read_csv(test_path, encoding = "ANSI").values.tolist()
#ブレンド味覚値
lst3 = pd.read_csv(EDB_path, encoding = "ANSI").values.tolist()


data = np.array(lst2)
name = np.array(lst3)
print(data)
# [1]Q関数を離散化して定義する関数　------------                                                                                                                                    
# 観測した状態を離散値にデジタル変換する                                                                                                                                            
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]
    #６分割するには仕切りが７つ                                                                                                                                                     

def convert_action(action):
    if action == 0:
        return "(6,1,3)"
    elif action == 1:
        return "(6,2,2)"
    elif action== 2:
        return "(6,3,1)"
    elif action== 3:
        return "(7,1,2)"
    elif action== 4:
        return "(7,2,1)"
    else:
        return "(8,1,1)"

def evaluation(judge,name,value,num,action):
    if value == 0:
        for i in range(6):
            if name[action][num] > name[i][num]:
                judge[i] += 0.5
            elif name[i][num] > name[action][num]:
                judge[i] -= 0.5
    elif value == 2:
        for i in range(6):
            if name[action][num] > name[i][num]:
                judge[i] -= 0.5
            elif name[i][num] > name[action][num]:
                judge[i] += 0.5
    return judge

# 各値を離散値に変換                                                                                                                                                                
def digitize_state(observation):
    discomfort_index,pressure,body_temp,sleep_time,wakeup_time,coffee_time = observation
    digitized = [
        np.digitize(discomfort_index, bins=bins(60.0, 80.0, num_dizitized)), #不快指数                                                                                              
        np.digitize(pressure, bins=bins(998.0, 1022.0, num_dizitized)), #気圧                                                                                                       
        np.digitize(body_temp, bins=bins(36.0, 38.0, num_dizitized)), #体温                                                                                                         
        np.digitize(sleep_time, bins=bins(0.0, 24.0, num_dizitized)), #睡眠時間                                                                                                     
        np.digitize(wakeup_time, bins=bins(0.0, 24.0, num_dizitized)),#起床してからの時間                                                                                           
        np.digitize(coffee_time, bins=bins(0.0, 24.0, num_dizitized)),#飲む時間帯                                                                                                   
    ]
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])

# [2]行動a(t)を求める関数 -------------------------------------                                                                                                                     
def get_action(next_state, episode):
           #徐々に最適行動のみをとる、ε-greedy法 #発生確率追加する                                                                                                                  
    epsilon = 0.5 * (1 / ((episode*0.1) + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1, 2, 3, 4, 5])
    return next_action

# [3]Qテーブルを更新する関数 -------------------------------------                                                                                                                  
def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    j = 5
    next_Max_Q = max(q_table[next_state][0],q_table[next_state][5])
    for i in range(j):
        q_table[state, i] = (1 - alpha) * q_table[state, action] + alpha * (reward[i] + gamma * next_Max_Q)
    return q_table

# [4]. メイン関数開始 パラメータ設定--------------------------------------------------------                                                                                        
max_number_of_steps = 6  #1試行のstep数                                                                                                                                            
num_episodes = 50  #総試行回数
num_consecutive_iterations = 3  
num_dizitized = 6  #分割数 
q_table = np.random.uniform(low=-1, high=1, size=(num_dizitized**6, 6))
total_reward_vec = np.zeros(num_consecutive_iterations)  #各試行の報酬を格納                                                                                                        

# [5] メインルーチン--------------------------------------------------
print(name)

feedback = []
for episode in range(num_episodes):  #試行数分繰り返す                                                                                                                              
    # 環境の初期化 
    #状態変数定義
    discomfort_index = float(data[episode][6])
    print("discomfort_index:" + str(discomfort_index)) 
    pressure = float(data[episode][7])
    print("pressure:" + str(pressure)) 
    body_temp = float(data[episode][2])
    print("body_temp:" + str(body_temp)) 
    sleep_time = float(data[episode][4])
    print("sleep_time:" + str(sleep_time)) 
    wakeup_time = float(data[episode][5])
    print("wakeup_time:" + str(wakeup_time))
    coffee_time = float(data[episode][3])
    print("coffee_time:" + str(coffee_time))

    observation = discomfort_index,pressure,body_temp,sleep_time,wakeup_time,coffee_time
    state = digitize_state(observation) #離散値の計算(6^6*6)                                                                                                                        
    action = np.argmax(q_table[state])  #rewardがmaxを引っ張って来る                                                                                                                
    reward_entire = [0,0,0,0,0,0]
    done = 0
    progressbar = tqdm(range(max_number_of_steps))
    for t in progressbar:  #1試行のループ
        progressbar.set_description("Processing %s" % t)
        time.sleep(1)
        # 行動a_tの実行により、s_{t+1}, r_{t}などを計算する
        if 5 < t:
            reward = -3
            break
        else:
            if data[episode][10] == action:
                reward = 1
                #うまい
                done = 2
                break
            elif data[episode][11] == action:
                #まずい
                reward = -1
            else :
                #普通
                reward = 0

        reward_entire[action] += reward  #報酬を追加

        if done < 2:
            #コクの味覚値をランダムで決定
            a = np.random.choice([0, 1, 2])
            num = 0
            evaluation(reward_entire,name,a,num,action)
            #酸味
            a =np.random.choice([0, 1, 2])
            num = 1
            evaluation(reward_entire,name,a,num,action)
            #苦味
            a =np.random.choice([0, 1, 2])
            num= 2
            evaluation(reward_entire,name,a,num,action)
            #甘味
            a =np.random.choice([0, 1, 2])
            num= 3
            evaluation(reward_entire,name,a,num,action)
            #香り
            a =np.random.choice([0, 1, 2])
            num= 4
            evaluation(reward_entire,name,a,num,action)
            #飲みやすさ
            a =np.random.choice([0, 1, 2])
            num= 5
            evaluation(reward_entire,name,a,num,action)

        # 離散状態s_{t+1}を求め、Q関数を更新する                                                                                                                                    
        next_state = digitize_state(observation)  #t+1での観測状態を、離散値に変換                                                                                                  
        q_table = update_Qtable(q_table, state, action, reward_entire, next_state)

        #  次の行動a_{t+1}を求める                                                                                                                                                  
        action = get_action(next_state, episode)    # a_{t+1} 
        state = next_state
progressbar.close()
np.savetxt(Q_table_path,q_table, delimiter=",") #Qtableの保存する場合                                                                                              

