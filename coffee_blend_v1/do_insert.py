from coffee_database.database import init_db
from coffee_database.database import db_session
from coffee_database.models import dataContent
from Weather_api.weather_api import Weather
import fitbit
import fitbit_api.fitbit_api as fba
import datetime

def updateToken(token):
    f = open(TOKEN_FILE, 'w')
    f.write(str(token))
    f.close()
    return

print("Saving Data to DB......")
TOKEN_FILE = "fitbit_api/token.txt"
token_dict = fba.get_token(TOKEN_FILE)
CLIENT_ID =  "22DR8P"
CLIENT_SECRET  = "e1fc1be370e61cf466646a1460f9b446"
ACCESS_TOKEN = token_dict['access_token']
REFRESH_TOKEN = token_dict['refresh_token']

DATE = datetime.date.today()
# ID等の設定
authd_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET
                                 ,access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN, refresh_cb = updateToken)
# 心拍数、睡眠時間を取得（1秒単位）
fit_data = fba.get_data(authd_client, DATE)

weather = Weather()
weather.get_weather_data()

d1 = dataContent()
d1.student_number = "AL15042"
d1.date = DATE
d1.heart_rate = fit_data[0]
d1.drink_time = fit_data[1]
d1.sleep_hour_data = fit_data[2]
d1.wakeup_to_drink = fit_data[3]
d1.pressure = weather.pressure
d1.thi = weather.thi

db_session.add(d1)
db_session.commit()

print("Success to save!!")