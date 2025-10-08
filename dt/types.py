from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class MeteoFrame(BaseModel):
    ts: datetime
    payload: Dict[str, float]  # raw SCADA row

class FeatureFrame(BaseModel):
    ts: datetime
    x: Dict[str, float]        # features for model
    y_actual: Optional[float]  # actual power if available
