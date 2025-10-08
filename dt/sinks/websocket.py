# Minimal stub for a websocket sink; pair with a dashboard receiver
import asyncio, json, websockets
from .base import Sink

class WebSocketSink(Sink):
    def __init__(self, url):
        self.url = url
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._sender())

    async def _sender(self):
        while True:
            msg = await self._queue.get()
            try:
                async with websockets.connect(self.url, ping_interval=None) as ws:
                    await ws.send(json.dumps(msg))
            except Exception as e:
                # In dev, swallow and continue; in prod log this
                await asyncio.sleep(0.5)

    def write(self, msg: dict):
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass
