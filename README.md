# wind-dt — Modular Performance Twin (Prototype)

A plug‑and‑play **digital twin** pipeline for wind power **performance monitoring** using **meteorological inputs only**. 
- Swappable **stream providers** (CSV replay now; Kafka/MQTT stubs included).
- Swappable **models** (LightGBM default; easily add SVR/Bagging).
- Centralized **feature builder** + **physics guardrails**.
- Pluggable **sinks** (Console + WebSocket stubs) for dashboards.

## Quickstart

```bash
# 1) Create & activate venv (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run (replays data/data_sample.csv ~1 day @ 60x speed)
python main.py
```

You should see streaming outputs like:
```
{'ts': '2020-01-01T00:00:00+00:00', 'v': 4.2, 'y': 120.0, 'y_hat': 110.5, 'pi': 1.086}
```

## Configuration
Everything is driven by `config/config.yaml`. Switch stream/model/sinks without code changes.

- **Stream**: `csv_replay` now; later swap to `kafka` by editing YAML.
- **Model**: `lgbm` default with monotone constraints on speed-derived features.
- **Sinks**: `console` and/or `websocket` (stub here; pair with your dashboard).

## Project layout
```
wind-dt/
  config/config.yaml          # choose stream/model/features/guardrails/sinks
  dt/                         # pipeline framework
  data/                       # drop in Turbine_Data
  main.py                     # entrypoint
  requirements.txt
```

## Notes
- This is a *prototype*: online/partial_fit is not implemented; we do a warmup fit on first N records, then stream.
- Drop in Turbine_Data CSV and update `rated_power` in `config.yaml`.
- Add more models in `dt/models/` by subclassing `ModelWrapper`.
