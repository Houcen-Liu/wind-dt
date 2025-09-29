import time
import csv
import yaml
import threading
import json
from enum import IntEnum
import paho.mqtt.client as mqtt

class KelmarshIndexs(IntEnum):
    Date_and_time=0
    Wind_speed =1
    Wind_direction =15
    Energy_Export= 27
    Energy_Import= 29
    Power = 62
    Ambient_Temp_Converter = 106
    Rotor_Speed = 214
    Yaw_Bearing_Angle = 252

class MQTTManager():
    def __init__(self,cfg_path) :
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Get YAML Params
        pcfg = cfg["data_manager"]["options"]
        self.broker = pcfg["broker"]
        self.port = pcfg["port"]
        self.speedup = pcfg["speedup"]
        self.number_turbines = pcfg["number_turbines"]

        self.client = mqtt.Client()
        self.client.connect(self.broker, self.port, 60)

        self.running = False
        self.main_thread = None

        data_path = pcfg["data_path"]
        data_year = pcfg["data_year"]

        self.data_files = []
        self.turbine_topics = []
        for i in range(1,self.number_turbines+1):
            file_name = data_path+"Turbine_Data_Kelmarsh_"+str(i)+"_"+str(data_year)+"-01-01_-_"+str(int(data_year)+1)+"-01-01_"+str(227+i)+".csv"
            self.data_files.append(csv.reader(open(file_name)))
            self.turbine_topics.append("turbine/state"+str(i))

        #skip headers
        for f in self.data_files:
            for skip in range(10):
                next(f)

    def _fetch_line(self, f):
        line = next(f)
        dict_payload = {
            "Date and time": line[KelmarshIndexs.Date_and_time],
            "Wind speed (m/s)": line[KelmarshIndexs.Wind_speed],
            "Wind direction (°)": line[KelmarshIndexs.Wind_direction],
            "Energy Export (kWh)": line[KelmarshIndexs.Energy_Export],
            "Energy Import (kWh)": line[KelmarshIndexs.Energy_Import],
            "Power (kW)": line[KelmarshIndexs.Power],
            "Ambient temperature (converter) (°C)": line[KelmarshIndexs.Ambient_Temp_Converter],
            "Rotor speed (RPM)": line[KelmarshIndexs.Rotor_Speed],
            "Yaw bearing angle (°)": line[KelmarshIndexs.Yaw_Bearing_Angle]
        }
        payload = json.dumps(dict_payload)
        return payload
    
    def _work_loop(self):
        while self.running:
            for i in range(len(self.data_files)):
                payload = self._fetch_line(self.data_files[i])
                self.client.publish(self.turbine_topics[i], payload)
                #print(f"Published Turbine {i+1}: {payload}")

            time.sleep(600/self.speedup)


    def start(self):
        self.running = True
        self.main_thread = threading.Thread(target=self._work_loop)
        self.main_thread.start()

    def stop(self):
        self.running = False
        self.main_thread.join()

        








