#### Layout ve callback fonksiyonları (live update) ####

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import json
import threading
import paho.mqtt.client as mqtt
import sys

from .plots import (
    plot_energy_flow,
    plot_environmental_conditions,
    plot_wind_conditions,
    plot_turbine_status,
    plot_power_vs_temperature,
    plot_predicted_vs_actual_power,
    table_latest_turbine_data,
    table_errors,
    table_prediction
)

# -------------------------------
# MQTT broker information
TURBINE = sys.argv[1]
BROKER = "localhost"
PORT = 1883
TOPICS = [
    "Kelmarsh/Turbine/state",
    "Kelmarsh/Predictions/current",
    "Kelmarsh/Predictions/results"
]

# -------------------------------
# Dash App
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Wind Turbine Dashboard"),
    
    dcc.Graph(id='energy-flow'),
    dcc.Graph(id='environmental-conditions'),
    dcc.Graph(id='wind-conditions'),
    dcc.Graph(id='turbine-status'),
    dcc.Graph(id='power-vs-temp'),
    dcc.Graph(id='predicted-vs-actual'),

    dcc.Graph(id='latest-turbine-table'),
    dcc.Graph(id='average-error-table'),
    dcc.Graph(id='predicitons-table'),


    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)  # Interval component triggers callback every 5 seconds
])

# -------------------------------
# Global data store (in-memory)
data_store = {
    "data": [],
    "prediction": [],
    "results": []
}

# -------------------------------
# MQTT Listener
def on_connect(client, userdata, flags, rc):
    for topic in TOPICS:
        client.subscribe(topic+TURBINE)

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())

    if msg.topic == "Kelmarsh/Turbine/state"+TURBINE:
        for k, v in payload.items():
            if k != "Date and time":
                payload[k] = float(v)
        data_store["data"].append(payload)
    elif msg.topic == "Kelmarsh/Predictions/current"+TURBINE:
        data_store["prediction"].append(payload)
    elif msg.topic == "Kelmarsh/Predictions/results"+TURBINE:
        for k, v in payload.items():
            if k != "Date and time":
                payload[k] = float(v)
        data_store["results"].append(payload)

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(BROKER, PORT, 60)
threading.Thread(target=mqtt_client.loop_forever, daemon=True).start()

# -------------------------------
# Dash callbacks
@app.callback(
    Output('energy-flow', 'figure'),
    Output('environmental-conditions', 'figure'),
    Output('wind-conditions', 'figure'),
    Output('turbine-status', 'figure'),
    Output('power-vs-temp', 'figure'),
    Output('predicted-vs-actual', 'figure'),
    Output('latest-turbine-table', 'figure'), 
    Output('average-error-table', 'figure'), 
    Output('predicitons-table', 'figure'), 

    Input('interval-component', 'n_intervals'),
)
def update_graphs(n):
    df_data = data_store["data"]
    df_pred = data_store["prediction"]
    df_results = data_store["results"]
    
    df_data_copy = df_data.copy()
    df_pred_copy = df_pred.copy()
    df_results_copy = df_results.copy()
    df_data_sample = None
    df_error_sample = None
    df_pred_sample = None
    if(len(df_data_copy)):
        df_data_sample=df_data_copy[-1]
    if(len(df_results_copy)):
        df_error_sample=df_results_copy[-1]
    if(len(df_pred_copy)):
        df_pred_sample=df_pred_copy[-1]

    fig_energy = plot_energy_flow(df_data_copy)
    fig_env = plot_environmental_conditions(df_data_copy)
    fig_wind = plot_wind_conditions(df_data_copy)
    fig_status = plot_turbine_status(df_data_copy)
    fig_power_temp = plot_power_vs_temperature(df_data_copy)
    fig_pred_actual = plot_predicted_vs_actual_power(df_results_copy)
    fig_latest_trubine_table = table_latest_turbine_data(df_data_sample)
    fig_latest_error_table = table_errors(df_error_sample)
    fig_latest_pred_table = table_prediction(None)

    return fig_energy, fig_env, fig_wind, fig_status, fig_power_temp, fig_pred_actual,fig_latest_trubine_table,fig_latest_error_table,fig_latest_pred_table

# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8050+int(TURBINE))