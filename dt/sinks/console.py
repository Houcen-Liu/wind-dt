from .base import Sink
class ConsoleSink(Sink):
    def write(self, msg: dict):
        print(msg)
