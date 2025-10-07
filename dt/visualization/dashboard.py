#### Layout ve callback fonksiyonları (live update) ####

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import json
import threading
import paho.mqtt.client as mqtt

from .plots import (
    plot_energy_flow,
    plot_environmental_conditions,
    plot_wind_conditions,
    plot_turbine_status,
    plot_power_vs_temperature,
    plot_predicted_vs_actual_power
)

# -------------------------------
# MQTT broker information
BROKER = "localhost"
PORT = 1883
TOPICS = [
    "Kelmarsh/Turbine1/Data",
    "Kelmarsh/Turbine1/Prediction",
    "Kelmarsh/Turbine1/Results"
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
        client.subscribe(topic)

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    if msg.topic == "Kelmarsh/Turbine1/Data":
        data_store["data"].append(payload)
    elif msg.topic == "Kelmarsh/Turbine1/Prediction":
        data_store["prediction"].append(payload)
    elif msg.topic == "Kelmarsh/Turbine1/Results":
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
    Input('interval-component', 'n_intervals')
)
def update_graphs(n):
    df_data = data_store["data"]
    df_pred = data_store["prediction"]
    df_results = data_store["results"]

    fig_energy = plot_energy_flow(df_data)
    fig_env = plot_environmental_conditions(df_data)
    fig_wind = plot_wind_conditions(df_data)
    fig_status = plot_turbine_status(df_data)
    fig_power_temp = plot_power_vs_temperature(df_data)
    fig_pred_actual = plot_predicted_vs_actual_power(df_pred, df_results)

    return fig_energy, fig_env, fig_wind, fig_status, fig_power_temp, fig_pred_actual

# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8050)