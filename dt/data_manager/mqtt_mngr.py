import time
import csv
import yaml
import threading
import json
import asyncio
import paho.mqtt.client as mqtt
from .handlers import predictionHandler
from datetime import datetime

class MQTTManager():
    def __init__(self,cfg_path) :
        # Get YAML Params
        with open(cfg_path, 'r',encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        pcfg = cfg["data_manager"]["options"]
        self.broker = pcfg["broker"]
        self.port = pcfg["port"]
        self.speedup = pcfg["speedup"]
        self.number_turbines = pcfg["number_turbines"]
        self.data_items = pcfg["data_items"] # List of column names
        scada_output_topic = pcfg["scada_output_topic"]
        pred_input_topic = pcfg["prediction_input_topic"]
        prediction_output_topic = pcfg["prediction_output_topic"]
        prediction_result_output_topic = pcfg["prediction_result_output_topic"]
        self.control_topic = pcfg["control_topic"]

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
        self.stop_event = threading.Event()

        # Load files, topics and handlers
        data_path = pcfg["data_path"] # Path to CSV's
        data_year = pcfg["data_year"] # Year to use
        self.data_files = [] # List of csv readers per file

        self.turbine_topics = [] # List of output topics of raw turbine data
        self.cur_pred_topics = [] # List of output topics current predictions topics
        self.pred_result_topics = [] # List of topics for calculated error of predictions

        self.pred_input_topics = [] # List of topics to recieve data from
        self.predictionHandlers = [] # handlers for each trubines predictions
        for i in range(1,self.number_turbines+1):
            # Files
            file_name = data_path+"Turbine_Data_Kelmarsh_"+str(i)+"_"+str(data_year)+"-01-01_-_"+str(int(data_year)+1)+"-01-01_"+str(227+i)+".csv"
            self.data_files.append(csv.reader(open(file_name,encoding="utf-8")))

            # Outputs
            self.turbine_topics.append(scada_output_topic+str(i))
            self.cur_pred_topics.append(prediction_output_topic+str(i))
            self.pred_result_topics.append(prediction_result_output_topic+str(i))


            # Inputs
            self.pred_input_topics.append(pred_input_topic+str(i)) 

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
        for topic in self.pred_input_topics:
            self.client.subscribe(topic)
        self.client.subscribe(self.control_topic)

    # Message reception handler
    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        data = json.loads(payload) 

        # Handle Predictions inputs
        if(msg.topic in self.pred_input_topics):
            # Add prediction to handler
            self.predictionHandlers[0].add_prediction(datetime.strptime(data["ts"], '%Y-%m-%dT%H:%M:%S'),data)
            # Send out prediction
            #print(data)
            self.client.publish(self.cur_pred_topics[0], payload)
        
        # Handle Control Inputs
        elif(msg.topic  == self.control_topic):
            self.speedup = data["speedup"]
            self.stop_event.set()
            self.stop_event.clear()


    # Validate a value being used from CSV
    def _validate_val(self,val):
        if(val == "NaN"):
                return None
        else:
            try:
                f_val=float(val)
                if(f_val < 0.0):
                    return None
                
                return val
            except ValueError:
                return val # Not a float
            
    # Fetch line from CSV file
    def _fetch_line(self, f,turbine_i):
        try:
            line = next(f)
        except StopIteration:
            self.running = False
            return None

        dict_payload = {}
        for idx, val in enumerate(self.data_indexs):
            v_val = self._validate_val(line[val])
            if(v_val is None):
                #print("MQTTManager [Warning]: dropping sample {} because of {} value in {}".format(line[0],line[val],self.data_items[idx]))
                return None
            dict_payload[self.data_items[idx]]=v_val

        dict_payload["Turbine"]=turbine_i
        return dict_payload
    
    # Main work loop to output turbine data
    def _work_loop(self):
        while self.running:
            for i in range(len(self.data_files)):
                payload = self._fetch_line(self.data_files[i],i+1) # Get next line from CSV

                # Skip bad rows
                if(payload == None):
                    continue
                
                # Output SCADA Data
                payload_json = json.dumps(payload)
                self.client.publish(self.turbine_topics[i], payload_json)

                # Calculate error of past predictions with prediction handler for turbine
                clean_payload = {k: v for k, v in payload.items() if k != "Date and time"}
                pred_data = self.predictionHandlers[i].receive(datetime.strptime(payload["Date and time"], '%Y-%m-%d %H:%M:%S'),clean_payload)
                
                # Output results of previous predictions
                if pred_data is not None:
                    #print(pred_data)
                    res_payload_json = json.dumps(pred_data)
                    self.client.publish(self.pred_result_topics[i], res_payload_json)
            
            # For large sleep times use event to prevent blocking
            if((600/self.speedup) > 30):
                self.stop_event.wait(600/self.speedup)
            else:
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
        self.stop_event.set()
        self.client.disconnect()
        self.main_thread.join()

