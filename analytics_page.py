import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def visualize_performance_trend():
    fig = go.Figure()
    metrics = ['avg_relevance', 'consistency', 'response_time']
    
    for model, data in st.session_state.analytics.items():
        df = pd.DataFrame(data)
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=df['interaction'],
                y=df[metric],
                mode='lines+markers',
                name=f'{model} - {metric}',
                visible='legendonly' if metric != 'avg_relevance' else None
            ))
    
    fig.update_layout(
        title="Model Performance Trend",
        xaxis_title="Interaction",
        yaxis_title="Score / Time",
        height=600,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def show():
    st.title("Analytics Dashboard")
    
    if not st.session_state.analytics:
        st.info("No analytics data available yet. Please chat with the document to generate analytics.")
        return
    
    st.plotly_chart(visualize_performance_trend(), use_container_width=True)
    
    # Display latest metrics
    st.subheader("Latest Metrics")
    latest_metrics = {model: data[-1] if data else {} for model, data in st.session_state.analytics.items()}
    df_latest = pd.DataFrame(latest_metrics).T
    if not df_latest.empty:
        st.dataframe(df_latest)
    
    # Display full analytics data
    st.subheader("Full Analytics Data")
    for model, data in st.session_state.analytics.items():
        st.write(f"**{model}**")
        st.dataframe(pd.DataFrame(data))