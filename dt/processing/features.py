import math
from datetime import datetime

def time_features(ts: datetime):
    # hour & month as sin/cos
    h = ts.hour + ts.minute/60.0
    m = ts.month
    import math as _M
    return {
        "hour_sin": _M.sin(2*_M.pi*h/24.0),
        "hour_cos": _M.cos(2*_M.pi*h/24.0),
        "month_sin": _M.sin(2*_M.pi*m/12.0),
        "month_cos": _M.cos(2*_M.pi*m/12.0),
    }

def build_features(raw: dict, ts, cfg: dict) -> dict:
    x = {}
    v_col = cfg.get("primary_speed") or "Wind speed (m/s)"
    v = raw.get(v_col) or raw.get("Wind speed (m/s)")
    if v is None:
        return {}

    x["v"]  = v
    x["v2"] = v*v
    x["v3"] = v*v*v

    if cfg.get("use_direction"):
        wdir = raw.get("Wind direction (°)")
        if wdir is not None:
            rad = math.radians(wdir)
            x["wdir_sin"] = math.sin(rad)
            x["wdir_cos"] = math.cos(rad)

    if cfg.get("add_ti"):
        sigma = raw.get('Wind speed, Standard deviation (m/s)')
        if sigma and v > 0:
            ti = max(0.0, min(0.6, sigma / v))
            x["ti"] = ti

    if t_col := cfg.get("temperature_col"):
        t = raw.get(t_col)
        if t is not None:
            x["temp"] = t

    if cfg.get("add_time_cyclical") and ts is not None:
        x.update(time_features(ts))
    return x
