import json
import paho.mqtt.client as mqtt
from .base import Sink

class MQTTSink(Sink):
    def __init__(self,broker,port):
        self.broker = broker
        self.port = port
        
        self.client = mqtt.Client()
        self.client.connect(broker, port, 60)
        return

    def write(self, msg: dict, topic):
        payload = json.dumps(msg)
        self.client.publish(topic, payload)
        return
