import abc
from typing import Dict, Optional

class ModelWrapper(abc.ABC):
    @abc.abstractmethod
    def fit(self, X, y): ...
    @abc.abstractmethod
    def predict(self, x: Dict[str, float]) -> float: ...
    def predict_quantiles(self, x: Dict[str, float], qs=(0.1,0.5,0.9)):
        return None
