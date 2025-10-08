import asyncio
import json
import paho.mqtt.client as mqtt

from datetime import datetime
from .base import StreamProvider
from ..types import MeteoFrame

class MQTTStream(StreamProvider):
    def __init__(self,broker,port,topics):
        self.running = True
        self.broker = broker
        self.port = port
        self.topics = topics
        
        self.loop = asyncio.get_event_loop()

        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(broker, port, 60)
        self.queue = asyncio.Queue()
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        for topic in self.topics:
            self.client.subscribe(topic)

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        data = json.loads(payload) 
        asyncio.run_coroutine_threadsafe(
            self.queue.put((msg.topic, data)),
            self.loop
        )

    async def stream(self):
        while self.running:
            topic, payload = await self.queue.get()
            if("diagnostsic" in payload and payload["diagnostsic"]=="Done"):
                break
            clean_payload = {k: v for k, v in payload.items() if k != "Date and time"}
            yield MeteoFrame(ts=datetime.strptime(payload["Date and time"], '%Y-%m-%d %H:%M:%S'), payload=clean_payload)

    def close(self):
        self.running = False
        self.client.loop_stop()
        self.client.disconnect()
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)

