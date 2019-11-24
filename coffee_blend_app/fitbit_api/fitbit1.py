import fitbit
import seaborn as sns
import json
from ast import literal_eval
import datetime

class Fitbit_user():
    
    def __init__(self, heart_rate = None, 
                 date = None, wakeup_time = None, sleep_hour_data =None, drink_time = None, wakeup_to_drink = None, 
                 access_token = None, refresh_token = None, authd_client = None, get_data_time = None):
        self.date = date
        self.heart_rate = heart_rate
        self.wakeup_time = wakeup_time
        self.sleep_hour_data = sleep_hour_data
        self.drink_time = drink_time
        self.wakeup_to_drink = wakeup_to_drink
        self.client_id = "22DR8P"
        self.client_secret = "e1fc1be370e61cf466646a1460f9b446"
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.authd_client = authd_client
        self.get_data_time = get_data_time
    def get_data(self):
        try:
            DATE = "2019-07-01"
            #以下心拍数データ、取得時間データ
            data_h = self.authd_client.intraday_time_series('activities/heart', DATE, detail_level='1min')
            heart_min = data_h["activities-heart-intraday"]["dataset"]
            temp = heart_min[-1]
            heart_data = list(temp.values())
            self.get_data_time = heart_data[0]
            temp_drink_time_list = heart_data[0].split(":")
            self.drink_time = int(temp_drink_time_list[0])
            self.heart_rate = int(heart_data[1])
            
            #以下、睡眠時間、起床から喫飲までの時間、喫飲時間
            sleep_data = self.authd_client.sleep(date = DATE)
            wakeup_time_list = list(sleep_data["sleep"][0]["minuteData"])
            temp_wakeup = wakeup_time_list[-1]["dateTime"]
            temp_wakeup_list = temp_wakeup.split(":")
            self.wakeup_time = int(temp_wakeup_list[0])
            sleep_min_data = int(sleep_data["sleep"][0]["minutesAsleep"])
            self.sleep_hour_data = sleep_min_data // 60
            self.wakeup_to_drink = self.drink_time - self.wakeup_time
        except Exception as e:
            print(e)
                       
    def get_token(self):
        tokens = open(token_file).read()
        token_dict = literal_eval(tokens)
        self.access_token = token_dict['access_token']
        self.refresh_token = token_dict['refresh_token']
        self.authd_client = fitbit.Fitbit(self.client_id, self.client_secret
                                          , access_token = self.access_token, refresh_token = self.refresh_token, refresh_cb = update_token)
        
    
def update_token(token):
    f = open(token_file, 'W')
    f.write(str(token))
    f.close()
    return

token_file = "token.txt"

if __name__ == "__main__":
    fitbit_user = Fitbit_user()
    fitbit_user.get_token()
    fitbit_user.get_data()
    print(fitbit_user.drink_time, fitbit_user.wakeup_time)
    print(str(fitbit_user.wakeup_to_drink))