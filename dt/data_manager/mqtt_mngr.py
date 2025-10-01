import time
import csv
import yaml
import threading
import json
import asyncio
import math
import paho.mqtt.client as mqtt
from .handlers import predictionHandler
from datetime import datetime

class MQTTManager():
    def __init__(self,cfg_path) :
        # Get YAML Params
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        pcfg = cfg["data_manager"]["options"]
        self.broker = pcfg["broker"]
        self.port = pcfg["port"]
        self.speedup = pcfg["speedup"]
        self.number_turbines = pcfg["number_turbines"]
        self.data_items = pcfg["data_items"] # List of column names

        # Setup MQTT
        self.client = mqtt.Client()
        self.client.connect(self.broker, self.port, 60)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()

        # Thread values
        self.running = False
        self.main_thread = None

        # Load files, topics and handlers
        data_path = pcfg["data_path"] # Path to CSV's
        data_year = pcfg["data_year"] # Year to use
        self.data_files = [] # List of csv readers per file

        self.turbine_topics = [] # List of output topics of raw turbine data
        self.cur_pred_topics = [] # List of output topics current predictions topics
        self.calc_pred_topics = [] # List of topics for calculated error of predictions

        self.input_topics = [] # List of topics to recieve data from
        self.predictionHandlers = [] # handlers for each trubines predictions
        for i in range(1,self.number_turbines+1):
            # Files
            file_name = data_path+"Turbine_Data_Kelmarsh_"+str(i)+"_"+str(data_year)+"-01-01_-_"+str(int(data_year)+1)+"-01-01_"+str(227+i)+".csv"
            self.data_files.append(csv.reader(open(file_name)))

            # Outputs
            self.turbine_topics.append("turbine/state"+str(i))
            self.cur_pred_topics.append("predictions/current"+str(i))
            self.calc_pred_topics.append("predictions/results"+str(i))


            # Inputs
            self.input_topics.append("ml/svrPredictions"+str(i)) 
            self.input_topics.append("ml/annPredictions"+str(i))

            # Predictions
            self.predictionHandlers.append(predictionHandler())


        #Skip headers and read column names
        for f in self.data_files:
            for skip in range(9):
                next(f)
            col_name = next(f)
            col_name[0] = col_name[0].lstrip("# ").strip() # remove '# '
            col_indexs = {name: idx for idx, name in enumerate(col_name)}

        # Build a list of column indexs
        self.data_indexs = [col_indexs[item] for item in self.data_items]



    # On connection subsribe to topics
    def _on_connect(self, client, userdata, flags, rc):
        for topic in self.input_topics:
            self.client.subscribe(topic)

    # Message reception handler
    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        data = json.loads(payload) 
        #asyncio.run_coroutine_threadsafe(
        #    self.queue.put((msg.topic, data)),
        #    self.loop
        #)

        # Add prediction to handler
        self.predictionHandlers[0].add_prediction(datetime.strptime(data["ts"], '%Y-%m-%dT%H:%M:%S'),data)
        # Send out prediction
        self.client.publish(self.cur_pred_topics[0], payload)


    def _message_data_handler(self,topic,data):
        
        return

    # Fetch line from CSV file
    def _fetch_line(self, f,turbine_i):
        line = next(f)   
        dict_payload = {}
        for idx, val in enumerate(self.data_indexs):
            if(line[val] == "NaN"):
                #print("MQTTManager [Warning]: dropping sample {} because of NaN value in {}".format(self.data_items[idx],line[0]))
                return None
            dict_payload[self.data_items[idx]]=line[val]

        dict_payload["Turbine"]=turbine_i
        return dict_payload
    
    # Main work loop to output turbine data
    def _work_loop(self):
        while self.running:
            for i in range(len(self.data_files)):
                payload = self._fetch_line(self.data_files[i],i+1)
                # Skip bad rows
                if(payload == None):
                    continue
                payload_json = json.dumps(payload)
                self.client.publish(self.turbine_topics[i], payload_json)

                # Calculate error of past predictions
                clean_payload = {k: v for k, v in payload.items() if k != "Date and time"}
                pred_data = self.predictionHandlers[i].receive(datetime.strptime(payload["Date and time"], '%Y-%m-%d %H:%M:%S'),clean_payload)
                
                # If there is data to publish
                if pred_data is not None:
                    print(pred_data)
                    res_payload_json = json.dumps(pred_data)
                    self.client.publish(self.calc_pred_topics[i], res_payload_json)


            time.sleep(600/self.speedup)

    # Start the data manager
    def start(self):
        self.running = True
        self.client.loop_start()
        self.main_thread = threading.Thread(target=self._work_loop)
        self.main_thread.start()

    # Stop the data Manager
    def stop(self):
        self.running = False
        self.client.disconnect()
        self.main_thread.join()

