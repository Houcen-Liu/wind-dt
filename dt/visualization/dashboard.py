#### Layout ve callback fonksiyonları (live update) ####

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import json
import threading
import paho.mqtt.client as mqtt
import sys
import yaml

from .plots import (
    plot_energy_flow,
    plot_environmental_conditions,
    plot_wind_conditions,
    plot_turbine_status,
    plot_power_vs_windspeed,
    plot_predicted_vs_actual_power,
    table_latest_turbine_data,
    table_errors,
    table_prediction
)
# Load speedup value from config
with open("config/mngr_config.yaml", "r") as f:
    config = yaml.safe_load(f)

speedup_value = config["data_manager"]["options"]["speedup"]
interval_value = config["data_manager"]["options"].get("interval_ms", 1000)  # default 1000 ms
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

    
     # Speed / Interval Slider
    html.Div([
        html.Label("Simulation Speed (ms interval)", style={
            'fontSize': '20px',
            'fontWeight': 'bold',
            'color': '#003366',
            'display': 'block',
            'marginBottom': '10px'
        }),

        dcc.Slider(
            id='speed-slider',
            min=100,      # minimum interval 100 ms
            max=5000,     # maximum interval 5000 ms
            step=100,
            value= interval_value,
            marks={100: '100ms', 1000: '1s', 5000: '5s'},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),

        html.Div(id='speed-display', style={
            'marginTop': '15px',
            'fontSize': '18px',
            'fontWeight': '500',
            'color': '#1F618D'
        })
    ], style={
        'width': '70%',
        'margin': 'auto',
        'padding': '20px',
        'borderRadius': '10px',
        'backgroundColor': '#F7FBFF',
        'boxShadow': '0px 2px 6px rgba(0,0,0,0.1)',
        'textAlign': 'center'
    }),
    
    html.Br(),
    
    dcc.Graph(id='wind-conditions'),
    dcc.Graph(id='turbine-status'),
    dcc.Graph(id='energy-flow'),
    dcc.Graph(id='environmental-conditions'),
    dcc.Graph(id='power-vs-temp'),
    dcc.Graph(id='predicted-vs-actual'),

    dcc.Graph(id='latest-turbine-table'),
    dcc.Graph(id='average-error-table'),
    dcc.Graph(id='predicitons-table'),


    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)  # Interval component triggers callback every 5 seconds
])
#------------------------------------------
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



# -------------------------------
# Interval slider callback
@app.callback(
    Output('speed-display', 'children'),
    Output('interval-component', 'interval'),
    Input('speed-slider', 'value')
)
def update_interval(value):
    global interval_value
    interval_value = value

    # Update YAML
    with open("config/mngr_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["data_manager"]["options"]["interval_ms"] = value
    with open("config/mngr_config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # Optionally send control message
    mqtt_client.publish("Kelmarsh/Control/Speedup", json.dumps({"interval_ms": value}))

    return f"Current interval: {value} ms", value
#-----------------------------------------------------------------------------------
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

    State('turbine-status', 'relayoutData'),
    State('turbine-status', 'figure'),
    State('predicted-vs-actual', 'relayoutData'),
    State('predicted-vs-actual', 'figure')
)
def update_graphs(n,turbine_relayout, turbine_prev_fig,pred_v_actual_relayout, pred_v_actual_prev_fig):
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
    fig_power_temp = plot_power_vs_windspeed(df_data_copy)
    fig_pred_actual = plot_predicted_vs_actual_power(df_results_copy)
    fig_latest_trubine_table = table_latest_turbine_data(df_data_sample)
    fig_latest_error_table = table_errors(df_error_sample)
    fig_latest_pred_table = table_prediction(None)

    # Preserve User selected layouts
    if turbine_relayout:
        x0 = turbine_relayout.get("xaxis.range[0]")
        x1 = turbine_relayout.get("xaxis.range[1]")
        y0 = turbine_relayout.get("yaxis.range[0]")
        y1 = turbine_relayout.get("yaxis.range[1]")
        if x0 and x1:
            fig_status.update_xaxes(range=[x0, x1])
        if y0 and y1:
            fig_status.update_yaxes(range=[y0, y1])
    if pred_v_actual_relayout:
        x0 = pred_v_actual_relayout.get("xaxis.range[0]")
        x1 = pred_v_actual_relayout.get("xaxis.range[1]")
        y0 = pred_v_actual_relayout.get("yaxis.range[0]")
        y1 = pred_v_actual_relayout.get("yaxis.range[1]")
        if x0 and x1:
            fig_pred_actual.update_xaxes(range=[x0, x1])
        if y0 and y1:
            fig_pred_actual.update_yaxes(range=[y0, y1])

    # Preserve Lines that were select/hidden
    if turbine_prev_fig and "data" in turbine_prev_fig:
        for i, trace in enumerate(fig_status.data):
            if i < len(turbine_prev_fig["data"]):
                trace.visible = turbine_prev_fig["data"][i].get("visible", True)

    if pred_v_actual_prev_fig and "data" in pred_v_actual_prev_fig:
        for i, trace in enumerate(fig_pred_actual.data):
            if i < len(pred_v_actual_prev_fig["data"]):
                trace.visible = pred_v_actual_prev_fig["data"][i].get("visible", True)

                
    return fig_energy, fig_env, fig_wind, fig_status, fig_power_temp, fig_pred_actual,fig_latest_trubine_table,fig_latest_error_table,fig_latest_pred_table

# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8050+int(TURBINE))