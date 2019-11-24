from flask import Flask, request, render_template
import numpy as np
import datetime
import pandas as pd
from Weather_api.weather_api import Weather
import fitbit
import fitbit_api.fitbit_api as fa
import System_Evaluation as se
import csv


#fitbit_api refresh_token更新用
def updateToken(token):
    f = open(TOKEN_FILE, 'w')
    f.write(str(token))
    f.close()
    return

#--------身体データ取得部分--------

TOKEN_FILE = "./fitbit_api/token.txt"
token_dict = fa.get_token(TOKEN_FILE)
CLIENT_ID =  "22DR8P"
CLIENT_SECRET  = "e1fc1be370e61cf466646a1460f9b446"
ACCESS_TOKEN = token_dict['access_token']
REFRESH_TOKEN = token_dict['refresh_token']
episode = 50
num_dizitized = 6
file_path = './state.csv'
#---------Http通信部分---------------

app = Flask(__name__)
my_port = 55555


# ブレンド比率取得
@app.route('/blend', methods = ['GET'])
def get_blend():
    DATE = datetime.date.today()
    # ID等の設定
    authd_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET
                                 ,access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN, refresh_cb = updateToken)
    
    
    #身体データ取得部分
    data = fa.get_data(authd_client,DATE)
    heart_rate = data[0]
    coffee_time = data[1]
    sleep_time = data[2]
    wakeup_time = data[3]
    body_temp = int(36.5 + (heart_rate - 65) / 10 * 0.55)
    
     #天候データ取得部分
    weather = Weather()
    weather.get_weather_data()
    discomfort_index = weather.thi
    pressure = weather.pressure
    
    observation = (discomfort_index,pressure,body_temp
                        ,sleep_time,wakeup_time,coffee_time)
    
    #----------csv読み出し部分-----------
    lst_q_table = pd.read_csv("EDB_Q_table.csv", header = None).values.tolist()
    q_table = np.array(lst_q_table)
    state = se.digitize_state(observation, num_dizitized)
    action = np.argmax(q_table[state])
    user_blend = se.convert_action(action)
    lst_observation = list(observation)
    lst_observation.append(state)
    lst_observation.append(action)
    try:
        state_csv = open(file_path, 'w', newline = '')
        writer = csv.writer(state_csv)
        writer.writerow(lst_observation)
    except Exception as e:
        print(e)
        return "failed to write"
    finally:
        state_csv.close()
    
    return_msg = "recomended coffee blend is {0}.".format(user_blend)
    return return_msg
    
@app.route('/blend', methods = ['POST'])
def post_feedback():
    lst_q_table = pd.read_csv("EDB_Q_table.csv", header = None).values.tolist()
    q_table = np.array(lst_q_table)
    state_csv = open(file_path, 'r')
    reader = csv.reader(state_csv)
    lst_state = [e for e in reader]
    lst_state_int = [int(s) for s in lst_state[0]]
    lst_name = pd.read_csv('EDB.csv').values.tolist()
    state = lst_state_int[6]
    action = lst_state_int[7]
    observation = tuple(lst_state_int[:6])
    name = np.array(lst_name)
    delicious = int(request.form["delicious"])         #ブレンド自体のおいしさ
    richness = int(request.form["richness"])           #コク
    sweet = int(request.form["sweet"])                 #甘み
    acidity = int(request.form["acidity"])             #酸味
    bitterness = int(request.form["bitterness"])       #苦味
    fragrance = int(request.form["fragrance"])         #香り
    easy_to_drink = int(request.form["easy_to_drink"]) #飲みやすさ
    feedbacks = [richness, sweet, acidity, bitterness, fragrance, easy_to_drink]
    reward_entire = [0,0,0,0,0,0]
    num = 0
    
    #おいしい :2 普通 : 1 まずい : 0
    if delicious == 2:
        reward = 1
    elif delicious == 1:
        reward = 0
    else:
        reward = -1
    reward_entire[action] += reward
    if delicious < 2:
        for feedback in feedbacks:
            while num < 6:
                se.evaluation(reward_entire, name, feedback, num, action)
                num += 1

    next_state = se.digitize_state(observation, num_dizitized)
    q_table = se.update_Qtable(q_table, state, action, reward_entire, next_state)    
    np.savetxt('EDB_Q_table.csv', q_table, delimiter = ",")
    return "succeeded update Q-table"
    
    
    
if __name__ == '__main__':
    app.run(debug = True, host = '160.16.210.86', port = my_port)
