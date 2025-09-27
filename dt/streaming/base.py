import abc
from typing import AsyncIterator
from ..types import MeteoFrame

class StreamProvider(abc.ABC):
    @abc.abstractmethod
    async def stream(self) -> AsyncIterator[MeteoFrame]:
        ...
