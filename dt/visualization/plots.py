 # Individual graphical functions (energy flow, wind rose, turbine status, etc.) 


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# -------------------------------
# Energy Flow (Sankey veya Bar)
def plot_energy_flow(df_data):
    if not df_data:
        return go.Figure()

    df = pd.DataFrame(df_data)
    if "Energy Export (kWh)" not in df.columns or "Energy Import (kWh)" not in df.columns:
        return go.Figure()

    fig = go.Figure(go.Sankey(
        node=dict(label=["Export", "Import"]),
        link=dict(
            source=[0, 1],
            target=[1, 0],
            value=[df["Energy Export (kWh)"].sum(), df["Energy Export (kWh)"].sum()]
        )
    ))
    fig.update_layout(title_text="Energy Flow", font_size=12)
    return fig

# -------------------------------
# Environmental Conditions (Line)
def plot_environmental_conditions(df_data):
    if not df_data:
        return px.line()
    df = pd.DataFrame(df_data)
    if "Nacelle ambient temperature (°C)" not in df.columns or "Date and time" not in df.columns:
        return px.line()
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    fig = px.line(df, x="Date and time", y="Nacelle ambient temperature (°C)", title="Ambient Temperature")
    return fig

# -------------------------------
# Wind Conditions (Polar)
def plot_wind_conditions(df_data):
    if not df_data:
        return px.scatter_polar()
    df = pd.DataFrame(df_data)

    if "Wind speed (m/s)" not in df.columns or "Wind direction (°)" not in df.columns:
        return px.scatter_polar()
    fig = px.scatter_polar(df, r="Wind speed (m/s)", theta="Wind direction (°)", title="Wind Conditions", color="Wind speed (m/s)")
    return fig

# -------------------------------
# Turbine Status (Multi-line)
def plot_turbine_status(df_data):
    if not df_data:
        return px.line()
    df = pd.DataFrame(df_data)
    if not all(col in df.columns for col in ["Power (kW)", "Rotor speed (RPM)", "Yaw bearing angle (°)"]):
        return px.line()
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    fig = px.line(df, x="Date and time", y=["Power (kW)", "Rotor speed (RPM)", "Yaw bearing angle (°)"],
                  title="Turbine Status")
    return fig

# -------------------------------
# Power vs Temperature (Scatter + Trendline)
def plot_power_vs_temperature(df_data):
    if not df_data:
        return px.scatter()
    df = pd.DataFrame(df_data)
    if not all(col in df.columns for col in ["Power (kW)", "Nacelle ambient temperature (°C)"]):
        return px.scatter()
    fig = px.scatter(df, x="Nacelle ambient temperature (°C)", y="Power (kW)", trendline="ols",
                     title="Power vs Temperature")
    return fig

# -------------------------------
# Predicted vs Actual Power (Dual Line + Residual)
def plot_predicted_vs_actual_power(df_results):
    if not df_results:
        return px.line(title="No results data")

    df_r = pd.DataFrame(df_results)

    required_cols = [
        "Date and time",
        "actual_power",
        "pred_hour",
        "pred_day",
        "pred_week",
        "pred_month"
    ]
    missing = [c for c in required_cols if c not in df_r.columns]
    if missing:
        return px.line(title=f"Missing columns: {', '.join(missing)}")

    df_melt = df_r.melt(
        id_vars=["Date and time"],
        value_vars=["actual_power", "pred_hour", "pred_day", "pred_week", "pred_month"],
        var_name="Type",
        value_name="Power"
    )
    
    color_map = {
        "actual_power": "red",
        "pred_hour": "#65b5ee",
        "pred_day": "#0400ff",
        "pred_week": "#17becf",
        "pred_month": "#0066ff"
    }

    fig = px.line(
        df_melt,
        x="Date and time",
        y="Power",
        color="Type",
        title="Predicted vs Actual Power Over Time",
        color_discrete_map=color_map
    )

    fig.update_layout(
        xaxis_title="Date and time",
        yaxis_title="Power",
        legend_title="Type"
    )

    return fig

# -------------------------------
# Current Turbine state
def table_latest_turbine_data(latest_sample: dict):
    if not latest_sample or not isinstance(latest_sample, dict):
        return go.Figure()

    # Convert the single dictionary to a DataFrame for the Plotly table
    df = pd.DataFrame(list(latest_sample.items()), columns=["Parameter", "Value"])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Parameter", "Value"],
                    fill_color="lightgrey",
                    align="left"
                ),
                cells=dict(
                    values=[df["Parameter"], df["Value"]],
                    fill_color="white",
                    align="left"
                )
            )
        ]
    )

    fig.update_layout(title="Latest Turbine Data", height=500)
    return fig

# -------------------------------
# Table of prediction error averages 
def table_errors(latest_sample: dict):
    if not latest_sample or not isinstance(latest_sample, dict):
        return go.Figure()
    
    keys_to_show = [
        "hour_error_average",
        "day_error_average",
        "week_error_average",
        "month_error_average"
    ]

    filtered_data = {k: latest_sample[k] for k in keys_to_show}
    df = pd.DataFrame(list(filtered_data.items()), columns=["Parameter", "Value"])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Range", "Error (MAE)"],
                    fill_color="lightgrey",
                    align="left"
                ),
                cells=dict(
                    values=[df["Parameter"], df["Value"]],
                    fill_color="white",
                    align="left"
                )
            )
        ]
    )

    fig.update_layout(title="Prediction Average Errors (MAE)", height=500)
    return fig

# -------------------------------
# Table of current prediction
def table_prediction(latest_sample: dict):
    if not latest_sample or not isinstance(latest_sample, dict):
        return go.Figure()
    
    keys_to_show = [
        "hour",
        "day",
        "week",
        "month"
    ]

    filtered_data = {k: latest_sample[k] for k in keys_to_show}
    df = pd.DataFrame(list(filtered_data.items()), columns=["Parameter", "Value"])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Range", "Prediction (kW)"],
                    fill_color="lightgrey",
                    align="left"
                ),
                cells=dict(
                    values=[df["Parameter"], df["Value"]],
                    fill_color="white",
                    align="left"
                )
            )
        ]
    )

    fig.update_layout(title="Prediction Average Errors (MAE)", height=500)
    return fig