# wind-dt — Modular Performance Twin (Prototype)

A plug‑and‑play **digital twin** pipeline for wind power **performance monitoring** using **meteorological inputs only**. 

## Quickstart
1. Install [mosquitto](https://mosquitto.org/download/)
2. Place SCADA Data in `data/`
3. Update `data_year` in `config/mngr_config.yaml` to match the data you placed

4. Then run the following from your terminal:
```bash
#Create & activate venv (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run live replay
python main_live.py

```
5. In a new console run:

```bash
python -m dt.visualization.dashboard <turbine number>
```
<i>\<turbine number\> is the number of the turbine you want to visualize. I.e. if you want to see turbine 3, replace it with 3.</i>

After launching the visualization you should see:
```bash
Dash is running on http://127.0.0.1:8051/
```

6. Open the link in your browser to see the visualization.


## Configuration
The Data manager is configured by `config/mngr_config.yaml`. 
- **speedup**: default data replay speed (600 (seconds) / \<speedup\>)
- **number_turbines**: number of turbines to manage (1-6)
- **data_path**: Path to SCADA data
- **data_year**: Year of data to use
- **data_items**: list of columns to include in representation model


The ML pipeline is configured by `config/config.yaml`. Switch stream/model/sinks without code changes.

- **Stream**: Source of data for the ML pipeline
- **Model**: `lgbm` default with monotone constraints on speed-derived features.
- **Sinks**: Where to output the results of the predictions

## Project layout
```
wind-dt/
  config/mngr_config.yaml     # Configure the Data Manager
  config/config.yaml          # Configure the ML pipeline
  dt/                         # Digital Twin Source code
  data/                       # Drop in Turbine_Data
  main_live.py                # Live Replay entrypoint
  requirements.txt
```

## Notes
- This is a *prototype*: online/partial_fit is not implemented; we do a warmup fit on first N records, then stream.
- Drop in Turbine_Data CSV and update `rated_power` in `config.yaml`.
- Add more models in `dt/models/` by subclassing `ModelWrapper`.