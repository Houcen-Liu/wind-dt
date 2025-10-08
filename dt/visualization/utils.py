
def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())

def add_trendline(fig, df, x, y):
    import plotly.express as px
    fig2 = px.scatter(df, x=x, y=y, trendline="ols")
    for trace in fig2.data:
        fig.add_trace(trace)
    return fig