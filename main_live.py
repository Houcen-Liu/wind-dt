import asyncio
import time
from dt.pipeline import Pipeline
from dt.data_manager.mqtt_mngr import MQTTManager

if __name__ == "__main__":
    ml_pl = Pipeline("config/config.yaml")
    ml_pl.start()
    mqttMngr = MQTTManager("config/mngr_config.yaml")
    mqttMngr.start()
    
    try:
        while True:
            time.sleep(1)
            if(not mqttMngr.running):
                mqttMngr.stop()
            if(not ml_pl.running):
                ml_pl.stop()
                break
    except KeyboardInterrupt:
        print("Stopping ... ")
        ml_pl.stop()
        mqttMngr.stop()
    