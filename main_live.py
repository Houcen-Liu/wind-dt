# main_live.py
# This script runs the full live pipeline: reading MQTT data, processing it,
# building features, applying ML model, and sending results to sinks (console, CSV, MQTT).

import asyncio
from dt.pipeline import Pipeline# --------------------------
# Entry point for live run
# --------------------------
if __name__ == "__main__":
    # Create the pipeline using the live config
    pipeline = Pipeline(cfg_path="config/config.yaml")

    # Run the pipeline in asyncio event loop
    # It will continuously listen to MQTT topics and process incoming SCADA data
    asyncio.run(pipeline.run())