# src/visualization.py

import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_candlestick(df, title="XAUUSD Candlestick Chart"):
    """Creates an interactive candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                           open=df['Open'],
                                           high=df['High'],
                                           low=df['Low'],
                                           close=df['Close'],
                                           name='Candles')])

    fig.update_layout(
        title=title,
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    fig.show()

def plot_predictions(historical_data, predictions, title="Model Predictions vs Actuals"):
    """Plots historical data and model predictions."""
    hist_df = historical_data.copy()
    pred_df = predictions.copy()

    fig = go.Figure()

    # Plot historical actuals
    fig.add_trace(go.Scatter(
        x=hist_df.index,
        y=hist_df['Close'],
        mode='lines',
        name='Historical Actual Price'
    ))

    # Plot predicted values
    fig.add_trace(go.Scatter(
        x=pred_df.index,
        y=pred_df['prediction'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='orange', dash='dot')
    ))
    
    # Plot prediction quantiles (confidence interval)
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df['p90'],
        fill=None, mode='lines', line_color='rgba(255,165,0,0.3)', name='90th Quantile'
    ))
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df['p10'],
        fill='tonexty', mode='lines', line_color='rgba(255,165,0,0.3)', name='10th Quantile'
    ))

    fig.update_layout(
        title=title,
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    fig.show()

def plot_feature_importance(interpretation, feature_names):
    """Plots feature importance from the TFT model."""
    importance = interpretation['importance']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=importance.mean(axis=1), y=feature_names, ax=ax, orient='h')
    ax.set_title("TFT Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig