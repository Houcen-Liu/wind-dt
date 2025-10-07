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
    if "Energy Export" not in df.columns or "Energy Import" not in df.columns:
        return go.Figure()

    fig = go.Figure(go.Sankey(
        node=dict(label=["Export", "Import"]),
        link=dict(
            source=[0, 1],
            target=[1, 0],
            value=[df["Energy Export"].sum(), df["Energy Import"].sum()]
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
    if "Nacelle Ambient Temperature" not in df.columns or "Date and time" not in df.columns:
        return px.line()
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    fig = px.line(df, x="Date and time", y="Nacelle Ambient Temperature", title="Ambient Temperature")
    return fig

# -------------------------------
# Wind Conditions (Polar)
def plot_wind_conditions(df_data):
    if not df_data:
        return px.scatter_polar()
    df = pd.DataFrame(df_data)
    if "Wind Speed" not in df.columns or "Wind Direction" not in df.columns:
        return px.scatter_polar()
    fig = px.scatter_polar(df, r="Wind Speed", theta="Wind Direction", title="Wind Conditions", color="Wind Speed")
    return fig

# -------------------------------
# Turbine Status (Multi-line)
def plot_turbine_status(df_data):
    if not df_data:
        return px.line()
    df = pd.DataFrame(df_data)
    if not all(col in df.columns for col in ["Power (kW)", "Rotor Speed (rpm)", "Yaw Bearing Angle"]):
        return px.line()
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    fig = px.line(df, x="Date and time", y=["Power (kW)", "Rotor Speed (rpm)", "Yaw Bearing Angle"],
                  title="Turbine Status")
    return fig

# -------------------------------
# Power vs Temperature (Scatter + Trendline)
def plot_power_vs_temperature(df_data):
    if not df_data:
        return px.scatter()
    df = pd.DataFrame(df_data)
    if not all(col in df.columns for col in ["Power (kW)", "Nacelle Ambient Temperature"]):
        return px.scatter()
    fig = px.scatter(df, x="Nacelle Ambient Temperature", y="Power (kW)", trendline="ols",
                     title="Power vs Temperature")
    return fig

# -------------------------------
# Predicted vs Actual Power (Dual Line + Residual)
def plot_predicted_vs_actual_power(df_pred, df_results):
    if not df_pred or not df_results:
        return px.line()
    df_p = pd.DataFrame(df_pred)
    df_r = pd.DataFrame(df_results)
    if not all(col in df_p.columns for col in ["y"]) or "actual_power" not in df_r.columns:
        return px.line()
    df_plot = pd.merge(df_p, df_r, left_on="ts", right_on="time", how="inner")
    fig = px.line(df_plot, x="time", y=["y", "actual_power"], title="Predicted vs Actual Power")
    # Residuals
    df_plot["residual"] = df_plot["actual_power"] - df_plot["y"]
    fig_res = px.line(df_plot, x="time", y="residual", title="Residuals (Actual - Predicted)")
    fig.add_trace(fig_res.data[0])
    return fig