import paho.mqtt.client as mqtt
import json
import sys

client = mqtt.Client()
client.connect("localhost", 1883, 60)
d = {"speedup":int(sys.argv[1])}
payload = json.dumps(d)
client.publish("control/mngr_inputs", payload)