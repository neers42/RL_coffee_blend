import fitbit
import json
from ast import literal_eval
import datetime


def updateToken(token):
    f = open(TOKEN_FILE, 'w')
    f.write(str(token))
    f.close()
    return

#心拍数データ取得関数
def get_data(authd_client, DATE):
#心拍数データ
    data_sec = authd_client.intraday_time_series('activities/heart', DATE, detail_level='1min') #'1sec', '1min', or '15min'
    heart_sec = data_sec["activities-heart-intraday"]["dataset"]
    temp = heart_sec[-1]
    heart_data = list(temp.values())
    get_time_list = heart_data[0].split(':')
    get_time = int(get_time_list[0])
    heart_rate = int(heart_data[1])
    drink_time = get_time    
    try:    
        #睡眠データ取得
        sleep_data = authd_client.sleep(date = DATE)
        wakeup_time_list = list(sleep_data["sleep"][0]["minuteData"])
        #wakeup_time=起床時間
        wakeup_time = wakeup_time_list[-1]["dateTime"]
        wakeup_time_list = wakeup_time.split(':')
        wakeup_hour = int(wakeup_time_list[0])
        sleep_min_data = int(sleep_data["sleep"][0]["minutesAsleep"])
        #min=睡眠分数データ、hour＝睡眠時間データ
        min_data = sleep_min_data % 60
        sleep_hour_data = sleep_min_data // 60
        wakeup_to_drink = get_time - wakeup_hour
        data_list = [heart_rate, drink_time, sleep_hour_data, wakeup_to_drink]
        return data_list
    except Exception as e:
        data_list = [heart_rate, drink_time, 0 ,0]
        return data_list
def get_token(TOKEN_FILE):
    # tokenファイル読み込み
    tokens = open(TOKEN_FILE).read()
    #文字列を辞書に変換(literal_eval関数)
    token_dict = literal_eval(tokens)
    return token_dict


if __name__ == "__main__":
    #ID,Token設定
    TOKEN_FILE = "token.txt"
    token_dict = get_token(TOKEN_FILE)
    CLIENT_ID =  "22DR8P"
    CLIENT_SECRET  = "e1fc1be370e61cf466646a1460f9b446"
    ACCESS_TOKEN = token_dict['access_token']
    REFRESH_TOKEN = token_dict['refresh_token']

    # 取得したい日付
    DATE = datetime.date.today()
    # ID等の設定
    authd_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET
                                 ,access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN, refresh_cb = updateToken)
    # 心拍数、睡眠時間を取得（1秒単位）
    data = get_data(authd_client, DATE)
    print(data)