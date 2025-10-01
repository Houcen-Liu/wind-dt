from datetime import datetime,timedelta
from typing import Dict
import random

class predictionFrame:
    def __init__(self,hour,day,week,month):
        self.hour = hour
        self.day = day
        self.week = week
        self.month = month
        return

class predictionHandler:
    def __init__(self):
        self.cur_hour_error_average =0.0
        self.hour_count = 0
        self.cur_day_error_average =0.0
        self.day_count = 0
        self.cur_week_error_average =0.0
        self.week_count = 0
        self.cur_month_error_average =0.0
        self.month_count = 0

        self.prediction_dict: Dict[datetime, predictionFrame] = {}


        return

    def _remove(self, ts: datetime):
        del self.prediction_dict[ts]

    def _calc_run_avg(self,mean,new,n):
        return (((n-1)*mean)+new)/n
    
    def receive(self,ts: datetime,data: Dict[str, float]):
        ret = {}
        hour_ts = ts - timedelta(hours=1)
        actual_power = float(data["Power (kW)"])
        if hour_ts in self.prediction_dict:
            ret["time"]=ts.strftime('%Y-%m-%d %H:%M:%S')
            ret["actual_power"]=actual_power

            self.hour_count +=1
            error = abs(self.prediction_dict[hour_ts].hour-actual_power)
            self.cur_hour_error_average = self._calc_run_avg(self.cur_hour_error_average,error,self.hour_count)
            ret["hour_error_average"]=self.cur_hour_error_average
            ret["pred_hour"]=self.prediction_dict[hour_ts].hour


        else:
            return None
        
        day_ts = ts - timedelta(days=1)
        if day_ts in self.prediction_dict:
            self.day_count +=1
            error = abs(self.prediction_dict[day_ts].day-actual_power)
            self.cur_day_error_average = self._calc_run_avg(self.cur_day_error_average,error,self.day_count)
            ret["day_error_average"]=self.cur_day_error_average
            ret["pred_day"]=self.prediction_dict[day_ts].day
        else:
            return ret
        

        week_ts = ts - timedelta(days=7)
        if week_ts in self.prediction_dict:
            self.week_count +=1
            error=abs(self.prediction_dict[week_ts].week-actual_power)
            self.cur_week_error_average = self._calc_run_avg(self.cur_week_error_average,error,self.week_count)
            ret["week_error_average"]=self.cur_week_error_average
            ret["pred_week"]=self.prediction_dict[week_ts].week
        else:
            return ret
        
        month_ts = ts - timedelta(days=30)
        if month_ts in self.prediction_dict:
            self.month_count +=1
            error = abs(self.prediction_dict[month_ts].month-actual_power)
            self.cur_month_error_average = self._calc_run_avg(self.cur_month_error_average,error,self.month_count)
            ret["month_error_average"]=self.cur_month_error_average
            ret["pred_month"]=self.prediction_dict[month_ts].month
            self._remove(month_ts)
        else:
            return ret

        return ret
    
    def add_prediction(self,ts: datetime,data: Dict[str, float]):
        #if (ts - timedelta(minutes=10)) not in self.prediction_dict:
            #print("predictionHandler [Warning]: Prediction data for last sample was lost or not received {}".format(ts))
        
        self.prediction_dict[ts]=predictionFrame(data["y"]+10.00,data["y"]+10.00,data["y"]+10.00,data["y"]+10.00)