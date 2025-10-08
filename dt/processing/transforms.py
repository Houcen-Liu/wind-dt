from typing import Dict, Any, List
import math
import pandas as pd

def apply_transforms(raw: Dict[str, Any], rules: List[dict]) -> Dict[str, Any]:
    """
    Apply simple row-wise transforms to a SCADA dictionary.

    Supported rules in config.yaml:

      - drop_negatives: ["col1", "col2", ...]
          -> drop row if any of these columns < 0

      - drop_nonfinite: ["col1", "col2", ...]
          -> drop row if any of these are NaN/None/inf

      - clip: {col: "ColName", min: float, max: float}
          -> clamp a column's value between min/max

      - fillna: {col: "ColName", value: float}
          -> replace NaN/None with value

      - rename: {from: "OldName", to: "NewName"}
          -> rename a column

    Returns the modified row dict, or {} if the row should be dropped.
    """
    if not rules:
        return raw

    r = dict(raw)
    for rule in rules:
        if "drop_negatives" in rule:
            for c in rule["drop_negatives"]:
                v = r.get(c, None)
                try:
                    if v is not None and float(v) < 0:
                        return {}  # drop row
                except Exception:
                    continue

        elif "drop_nonfinite" in rule:
            for c in rule["drop_nonfinite"]:
                v = r.get(c, None)
                try:
                    fv = float(v)
                except Exception:
                    return {}  # drop if not castable
                if not math.isfinite(fv):
                    return {}

        elif "clip" in rule:
            info = rule["clip"]
            c = info["col"]
            lo = info.get("min", None)
            hi = info.get("max", None)
            if c in r and r[c] is not None:
                try:
                    v = float(r[c])
                    if lo is not None and v < lo: v = lo
                    if hi is not None and v > hi: v = hi
                    r[c] = v
                except Exception:
                    pass

        elif "fillna" in rule:
            info = rule["fillna"]
            c = info["col"]
            val = info.get("value", 0.0)
            if c in r:
                if r[c] is None or (isinstance(r[c], float) and pd.isna(r[c])):
                    r[c] = val

        elif "rename" in rule:
            info = rule["rename"]
            f = info.get("from")
            t = info.get("to")
            if f in r and t:
                r[t] = r.pop(f)

    return r
