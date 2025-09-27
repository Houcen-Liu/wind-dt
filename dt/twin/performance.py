from typing import Optional

class PerformanceTwin:
    def __init__(self, model, guardrails_cfg):
        self.model = model
        self.guardrails_cfg = guardrails_cfg

    def step(self, feats: dict, y_actual: Optional[float] = None):
        v = feats.get("v", 0.0)
        y_hat = self.model.predict(feats)
        from ..processing.guardrails import apply_physics
        y_hat = apply_physics(y_hat, v, self.guardrails_cfg)
        out = {"y_hat": y_hat}
        if y_actual is not None and y_hat > 1e-6:
            out["pi"] = y_actual / y_hat
        return out
