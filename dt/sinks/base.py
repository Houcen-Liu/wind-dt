import abc
class Sink(abc.ABC):
    @abc.abstractmethod
    def write(self, msg: dict,topic = None):
        ...
