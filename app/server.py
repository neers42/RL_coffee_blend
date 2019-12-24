from flask import Flask, request, render_template, jsonify
import numpy as np
import datetime
import pandas as pd
from Weather_api.weather_api import Weather
import fitbit
import fitbit_api.fitbit_api as fa
import System_Evaluation as se
import csv
import json
from ast import literal_eval


num_dizitized = 6
#flaskインスタンス生成、portはかぶらないようにする
app = Flask(__name__)
my_port = 55555
#fitbit_api refresh_token更新用
def updateToken(token):
    f = open('temp_token.txt', 'r')
    user_id = f.read()
    token_file_path = "./fitbit_api/" + user_id + "_token.txt"
    f = open(token_file_path, 'w')
    f.write(str(token))
    f.close()
    return

# ブレンド比率取得
@app.route('/blend', methods = ['GET'])
def get_blend():
    user_id = str(request.args.get("user_id"))
    client_id_path = "./fitbit_api/" + user_id + "_client_id.txt"
    client_id = open(client_id_path).read()
    #文字列を辞書に変換(literal_eval関数)
    client_id_dict = literal_eval(client_id)
    CLIENT_ID = client_id_dict['CLIENT_ID']
    CLIENT_SECRET = client_id_dict['CLIENT_SECRET']
    f = open('temp_token.txt', 'w')
    f.write(user_id)
    f.close()
    blend_data_path = "./blend_data/" + user_id + "data.csv"
    fitbit_data_path = "./fitbit_data/" + user_id + "_fitbit_data.csv"
    token_file_path = "./fitbit_api/" + user_id + "_token.txt"
    state_file_path = "./state/" + user_id + "_state.csv"
    Q_Table_path = "./EDB/Q_Table/" + user_id + "_EDB_Q_table.csv"
    DATE = datetime.date.today()
    token_dict = fa.get_token(token_file_path)
    ACCESS_TOKEN = token_dict['access_token']
    REFRESH_TOKEN = token_dict['refresh_token']
    # ID等の設定
    authd_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET
                                 ,access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN, refresh_cb = updateToken)
    
    
    #身体データ取得部分
    data = fa.get_data(authd_client,DATE)
    data_csv = open(fitbit_data_path, 'w', newline = '')
    data_writer = csv.writer(data_csv)
    data_writer.writerow(data)
    heart_rate = data[0]
    drink_time = data[1]
    sleep_time = data[2]
    wakeup_time = data[3]
    body_temp = int(36.5 + (heart_rate - 65) / 10 * 0.55)
    
     #天候データ取得部分
    weather = Weather()
    weather.get_weather_data()
    discomfort_index = weather.thi
    pressure = weather.pressure
    
    observation = (discomfort_index,pressure,body_temp
                        ,sleep_time,wakeup_time,drink_time)
    #----------csv読み出し部分-----------
    #ユーザのコーヒーブレンド種類の読み出し
    lst_blend = pd.read_csv(blend_data_path, header = None).values.tolist()
    user_coffee_blend = np.array(lst_blend)
    #Q_tableの読み出し
    lst_q_table = pd.read_csv(Q_Table_path, header = None).values.tolist()
    q_table = np.array(lst_q_table)
    #状態を離散化する
    state = se.digitize_state(observation, num_dizitized)
    action = np.argmax(q_table[state])
    #アクションからブレンド比率に変換する
    user_blend = se.convert_action(action)
    lst_observation = list(observation)
    lst_observation.append(state)
    lst_observation.append(action)
    #jsonデータ化する
    state_data = {"BodyTemperature" : str(body_temp), "SleepTime" : str(sleep_time), 
                    "WakeupTime" : str(wakeup_time), "Pressure" : str(pressure), 
                    "DiscomfortIndex" : str(discomfort_index),
                    "DrinkTime" : str(drink_time), "UserBlend" : user_blend,
                    "MainBlend" : str(user_coffee_blend[0][0]), "SubBlend1" : str(user_coffee_blend[0][1]),
                    "SubBlend2" : str(user_coffee_blend[0][2]),
                    "MainBlendPercentage" : str(user_blend[0]), "SubBlend1Percentage" : str(user_blend[1]),
                    "SubBlend2Percentage" : str(user_blend[2])}
    try:
        state_csv = open(state_file_path, 'w', newline = '')
        writer = csv.writer(state_csv)
        writer.writerow(lst_observation)
    except Exception as e:
        print(e)
        return "failed to write"
    finally:
        state_csv.close()
    
    return jsonify(state_data)
    
@app.route('/blend', methods = ['POST'])
def post_feedback():
    #Q-tableとstateのcsv読み取り
    user_id = str(request.args.get("user_id"))
    state_file_path = "./" + user_id + "_state.csv"
    Q_Table_path = "./EDB/Q_Table/" + user_id + "_EDB_Q_table.csv"
    EDB_file_path = "./EDB/" + user_id + "_EDB.csv"
    lst_q_table = pd.read_csv(Q_Table_path, header = None).values.tolist()
    q_table = np.array(lst_q_table)
    state_csv = open(state_file_path, 'r')
    reader = csv.reader(state_csv)
    lst_state = [e for e in reader]
    lst_state_int = [int(s) for s in lst_state[0]]
    lst_name = pd.read_csv(EDB_file_path).values.tolist()
    state = lst_state_int[6]
    action = lst_state_int[7]
    observation = tuple(lst_state_int[:6])
    name = np.array(lst_name)

    #postされた値を変換して各変数に格納する
    json_data = request.get_json()
    delicious = json_data["coffee_taste"]       #ブレンド自体のおいしさ
    richness = json_data["richness"]            #コク
    sweetness = json_data["sweetness"]          #甘み
    acidity = json_data["acidity"]              #酸味
    bitterness = json_data["bitterness"]        #苦味
    fragrance = json_data["fragrance"]          #香り
    easy_to_drink = json_data["easy_to_drink"]  #飲みやすさ
    feedbacks = [richness, sweetness, acidity, bitterness, fragrance, easy_to_drink]
    print("ok")
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
                if num == action:
                    num += 1
                    continue
                else:
                    reward_entire = se.evaluation(reward_entire, name, feedback, num, action)
                    num += 1

    next_state = se.digitize_state(observation, num_dizitized)
    q_table = se.update_Qtable(q_table, state, action, reward_entire, next_state)    
    np.savetxt(Q_Table_path, q_table, delimiter = ",")
    return "succeeded update Q-table"
    
    
    
if __name__ == '__main__':
    app.run(debug = True, host = '160.16.210.86', port = my_port)
