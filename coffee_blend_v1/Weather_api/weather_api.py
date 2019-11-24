import requests
import json

class Weather():
    def __init__(self, degrees_k = None, degrees_c = None, humidity = None
                 , pressure = None, thi = None, api_key = None, api = None
                 , name = None, weather = None):
        self.degrees_k = degrees_k
        self.degrees_c = degrees_c
        self.humidity = humidity
        self.pressure = pressure
        self.thi = thi
        self.api_key = api_key
        self.api = api
        self.name = name
        self.weather = weather
    def convert_KtoC(degrees_k):
        c = degrees_k - 273.15
        return c
    def get_THI(degrees_c, humidity):
        THI = 0.81 * degrees_c + 0.01 * humidity * (0.99 * degrees_c - 14.3) + 46.3
        return int(THI)


    def get_weather_data(self):
        self.api_key = "977eed07ff1f4fc7f9c200fd2970f5df"
        self.api = "http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}"
        city_name = "Tokyo"
        url = self.api.format(city=city_name, key=self.api_key)
        r = requests.get(url)
        data = json.loads(r.text)

        if data["weather"][0]["description"] == "clear sky":
            self.weather = "快晴"
        elif data["weather"][0]["description"] == "few clouds":
            self.weather = "晴れ"
        elif data["weather"][0]["description"] == "scattered clouds" or "broken clouds":
            self.weather = "曇り"
        elif data["weather"][0]["description"] == "shower rain" or "rain" or "thnderstorm":
            self.weather ="雨"
        else:
            self.weather = "雪"
        #degrees_k=ケルビン温度、degrees_c=摂氏温度
        self.name = data["name"]
        self.degrees_k = float(data["main"]["temp"])
        self.degrees_c = int(Weather.convert_KtoC(self.degrees_k))
        self.humidity = data["main"]["humidity"]
        self.pressure = data["main"]["pressure"]
        self.thi = Weather.get_THI(self.degrees_c, self.humidity)
        
    def weather_data_display(self):    
        print("+ 都市= ", self.name)
        print("| 天気= ", self.weather)
        print("| 気温= ",str(self.degrees_c)+ "*C")
        print("| 湿度= ",str(self.humidity) + "%")
        print("| 気圧= ", str(self.pressure) + "hPa")
        print("| 不快指数 = ", str(self.thi))
    
if __name__ == '__main__':
    weather = Weather()
    weather.get_weather_data()
    weather.weather_data_display()