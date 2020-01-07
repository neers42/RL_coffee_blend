# coding:utf-8-sig                                                                                                                                                              
# [0]ライブラリのインポート                                                                                                                                                         
import numpy as np
import time
import pandas as pd
import fitbit
import fitbit_api.fitbit_api as fa
from Weather_api.weather_api import Weather

# [1]Q関数を離散化して定義する関数　------------                                                                                                                                    
# 観測した状態を離散値にデジタル変換する                                                                                                                                            
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]
    #６分割するには仕切りが７つ                                                                                                                                                     

def convert_action(action):
    if action == 0:
        main_blend = 60
        sub1 = 10
        sub2 = 30
        return [main_blend, sub1, sub2]
    elif action == 1:
        main_blend = 60
        sub1 = 20
        sub2 = 20
        return [main_blend, sub1, sub2]
    elif action== 2:
        main_blend = 60
        sub1 = 30
        sub2 = 10
        return [main_blend, sub1, sub2]
    elif action== 3:
        main_blend = 70
        sub1 = 10
        sub2 = 20
        return [main_blend, sub1, sub2]
    elif action== 4:
        main_blend = 70
        sub1 = 20
        sub2 = 10
        return [main_blend, sub1, sub2]
    else:
        main_blend = 80
        sub1 = 10
        sub2 = 10
        return [main_blend, sub1, sub2]

def evaluation(judge,name,value,num,action):
    if value == 1:
        for i in range(6):
            if name[action][num] > name[i][num]:
                judge[i] += 0.5
            elif name[i][num] > name[action][num]:
                judge[i] -= 0.5
    elif value == 3:
        for i in range(6):
            if name[action][num] > name[i][num]:
                judge[i] -= 0.5
            elif name[i][num] > name[action][num]:
                judge[i] += 0.5
    return judge

# 各値を離散値に変換                                                                                                                                                                
def digitize_state(observation, num_dizitized):
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
def get_action(next_state, episode, q_table):
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

if __name__ == "__main__":
    print("学籍番号を入力してください\n")
    user_id = input("Student ID : ")
    Q_Table_path = "./EDB/Q_Table/" + user_id + "_EDB_Q_table.csv"
    test_data_path = "./test_data/" + user_id + "_data.csv"
    

    #Q_tableの読み込み
    lst_q_table = pd.read_csv(Q_Table_path, header = None).values.tolist()
    q_table = np.array(lst_q_table)
    lst_test_data = pd.read_csv(test_data_path, encoding="ANSI").values.tolist()
    test_data = np.array(lst_test_data)
    episode = 51
    print(test_data)
    for i in range(10):
        discomfort_index = int(float(test_data[episode + i][6]))
        pressure = int(float(test_data[episode + i][7]))
        body_temp = float(test_data[episode + i][2])
        sleep_time = int(float(test_data[episode + i][4]))
        wakeup_time = int(float(test_data[episode + i][5]))
        coffee_time = int(float(test_data[episode + i][3]))
        observation = discomfort_index,pressure,body_temp,sleep_time,wakeup_time,coffee_time
        state = digitize_state(observation, 6) #離散値の計算(6^6*6)                                                                                                                        
        action = np.argmax(q_table[state])  #rewardがmaxを引っ張って来る
        blend = convert_action(action)
        print("episode : ", episode + i)
        rec_blend = "メイン :  " + str(blend[0]) + "%  サブ1 : " + str(blend[1]) + "% サブ2 : " + str(blend[2]) + "%\n"
        print("Action : ", action)
        print("オススメのブレンドは" + rec_blend)
    print("episode 60 complete!")

