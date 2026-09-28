import plotly.express as px
import plotly.graph_objects as go

def plot_kpis(df):
    return {
        'Rows': df.shape[0],
        'Columns': df.shape[1],
        'Missing Values': df.isna().sum().sum()
    }

def plot_bar_chart(df, x_col, y_col):
    if df[x_col].nunique() > 50:
        df_agg = df.groupby(x_col)[y_col].mean().reset_index().head(50)
    else:
        df_agg = df.groupby(x_col)[y_col].mean().reset_index()
    return px.bar(df_agg, x=x_col, y=y_col, title=f"Average {y_col} by {x_col}")

def plot_scatter(df, x_col, y_col, color_col=None):
    return px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter: {x_col} vs {y_col}")

def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    return px.imshow(corr, title="Correlation Heatmap", text_auto=True, aspect='auto')

def plot_trend_line(df, time_col, val_col):
    df_sorted = df.sort_values(by=time_col)
    return px.line(df_sorted, x=time_col, y=val_col, title=f"Trend of {val_col} over {time_col}")
