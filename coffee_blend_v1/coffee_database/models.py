from sqlalchemy import Column, Integer, String, Text
from coffee_database.database import Base
from datetime import datetime

class dataContent(Base):
    __tablename__ = 'fitbitcontents'
    id = Column(Integer, primary_key=True)
    student_number = Column(String(200))   #学番
    date = Column(String(200))             #日付
    heart_rate = Column(Integer)           #心拍数
    drink_time = Column(Integer)       #喫飲時間
    sleep_hour_data = Column(Integer)      #睡眠時間
    wakeup_to_drink = Column(Integer)      #起床から喫飲までの時間
    pressure = Column(Integer)             #気圧
    thi = Column(Integer)                  #不快指数
    
    

def __init__(self, student_number = None, date = None, heart_rate = None , drink_time = None, sleep_hour_data = None, wakeup_to_drink = None, pressure = None, thi = None):
    self.student_number = student_number
    self.date = date
    #以下が入力
    self.heart_rate = heart_rate
    self.drink_time = drink_time
    self.sleep_hour_data = sleep_hour_data
    self.wakeup_to_drink = wakeup_to_drink
    self.pressure = pressure
    self.thi = thi

def __repr__(self):
    return '<Date %r>' % (self.date)
   
    
    
    