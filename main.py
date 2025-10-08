import asyncio
#import time
from dt.pipeline import Pipeline
#from dt.data_manager.mqtt_mngr import MQTTManager

if __name__ == "__main__":
    ml_pl = Pipeline("config/config.yaml")
    #mqttMngr = MQTTManager("config/mngr_config.yaml")
    #mqttMngr.start()

    asyncio.run(ml_pl.run())
    
    #try:
    #    while True:
    #        time.sleep(1)
    #except KeyboardInterrupt:
    #    print("Stopping ... ")
    #    mqttMngr.stop()
    
