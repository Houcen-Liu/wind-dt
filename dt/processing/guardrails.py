def apply_physics(y_hat: float, v: float, cfg: dict) -> float:
    if not cfg.get("enabled", True):
        return y_hat
    cut_in = cfg.get("cut_in", 3.0)
    rated  = cfg.get("rated_speed", 12.0)
    cut_out = cfg.get("cut_out", 25.0)
    p_rated = cfg.get("rated_power", 2000.0)
    if v < cut_in or v > cut_out:
        return 0.0
    return min(y_hat, p_rated) if v >= rated else y_hat
