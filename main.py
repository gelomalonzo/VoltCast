import streamlit as st
from datetime import timedelta
import io

import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import load_model
from tcn import TCN
import plotly.graph_objects as go

# PAGE SETUP
st.set_page_config(
    page_title="VoltCast",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/styles.css") as f:
    st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.sidebar.container():
    st.image("assets/banner.png")
    
# GLOBALS
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True
if "show_results" not in st.session_state:
    st.session_state.show_results = False

def deseason(ts_df:pd.DataFrame, season_df:pd.DataFrame) -> pd.DataFrame:
    deseasoned_df = ts_df.copy()
    for index, row in ts_df.iterrows():
        for col in ts_df.columns:
            deseasoned_df.loc[index, col] = deseasoned_df.loc[index, col] - season_df.loc[index.strftime("%m-%d"), col]
    return deseasoned_df

def revert_deseason(ts_df:pd.DataFrame, season_df:pd.DataFrame) -> pd.DataFrame:
    deseasoned_df = ts_df.copy()
    for index, row in ts_df.iterrows():
        for col in ts_df.columns:
            deseasoned_df.loc[index, col] = deseasoned_df.loc[index, col] + season_df.loc[index.strftime("%m-%d"), col]
    return deseasoned_df

def revert_detrend(df:pd.DataFrame, trend_scaler:MinMaxScaler, trend_df:pd.DataFrame, trend_model:object=None) -> pd.DataFrame:
    detrended_df = df.copy()

    X = get_input_sequences(trend_df)
    preds = trend_model.predict(X)
    preds = compress_to_timestep(preds)[-180:]
    h_dates = [trend_df.index[-1] + timedelta(days=i) for i in range(1, 180 + 1)]
    temp_trend_df = pd.DataFrame(preds, columns=df.columns, index=h_dates)

    temp_trend_df[df.columns] = trend_scaler.inverse_transform(temp_trend_df[df.columns])
    for index, row in df.iterrows():
        for col in df.columns:
            detrended_df.loc[index, col] = detrended_df.loc[index, col] + temp_trend_df.loc[index, col]

    return detrended_df

def get_input_sequences(df:pd.DataFrame) -> np.ndarray:
    X = []
    for i in range(len(df) - 180 + 1):
        X.append(df.iloc[i : i + 180].values)
    return np.array(X, dtype=np.float64)

def compress_to_timestep(seq_arr:np.ndarray) -> np.ndarray:
    temp_arr = []
    # Extract the first prediction from each sequence
    if len(seq_arr) > 0:
        for sequence in seq_arr:
            temp_arr.append(sequence[0])
    # Extract the last time step from the last sequence
    for timestep in seq_arr[len(seq_arr) - 1]:
        temp_arr.append(timestep)
    return np.array(temp_arr, dtype=np.float64)

# INPUT PANE
with st.sidebar.container():
    left_col, input_col, right_col = st.columns((0.05, 0.90, 0.05))
    with input_col:
        with st.form("input_form"):
            past_data_file = st.file_uploader("Upload historical data here:", type="csv")
            actual_data_file = st.file_uploader("Upload actual data here (optional):", type="csv")
            msg = st.container()
            submit_btn = st.form_submit_button("Forecast")
            
            if submit_btn:
                if past_data_file:
                    with msg:
                        msg.empty()
                        st.session_state.show_welcome = False
                        st.session_state.show_results = True
                else:
                    with msg:
                        st.error("Historical data cannot be empty.")
                        st.session_state.show_welcome = True
                        st.session_state.show_results = False

# WELCOME PAGE
if st.session_state.show_welcome:
    st.header(":wave: Welcome")
    st.write("---")
    
    about_col, instructions_col = st.columns((0.4, 0.6))
    with about_col:
        st.subheader("About")
    with instructions_col:
        st.subheader("Instructions")

# RESULTS PAGE
if st.session_state.show_results:
    results_msg = st.container()
    st.header(":chart_with_upwards_trend: Results")
    st.write("---")
    
    # store input files to DataFrame
    past_data_text = past_data_file.read()
    past_data_text_str = str(past_data_text, "utf-8")
    past_data_df = pd.read_csv(io.StringIO(past_data_text_str))
    past_data_df["DATE"] = pd.to_datetime(past_data_df["DATE"])
    past_data_df["DATE"] = past_data_df["DATE"].dt.date
    past_data_df.set_index("DATE", inplace=True)
    if actual_data_file:
        actual_data_text = actual_data_file.read()
        actual_data_text_str = str(actual_data_text, "utf-8")
        actual_data_df = pd.read_csv(io.StringIO(actual_data_text_str))
        actual_data_df["DATE"] = pd.to_datetime(actual_data_df["DATE"])
        actual_data_df["DATE"] = actual_data_df["DATE"].dt.date
        actual_data_df.set_index("DATE", inplace=True)
    
    # import models
    numeric_cols = ["LUZVIS PRICE (PHP/KWH)", "LUZON DEMAND (MW)", "VISAYAS DEMAND (MW)", "LUZON SUPPLY (MW)", "VISAYAS SUPPLY (MW)"]
    season_df = pd.read_csv("assets/season_df.csv").set_index("DATE")
    trend_scaler = joblib.load("assets/trend_scaler.pkl")
    scaler = joblib.load("assets/scaler.pkl")
    forecast_model = load_model("assets/tcn.h5", custom_objects={"TCN": TCN}, compile=False)
    # forecast_model.compile()
    trend_model = load_model("assets/lstm_trend.h5")
    
    # preprocess
    actual_past_data_df = past_data_df.copy()
    past_data_df = pd.get_dummies(past_data_df, columns=["BLOCK"])
    trend_df = past_data_df.copy()
    for col in numeric_cols:
        decomposition = seasonal_decompose(past_data_df[col], model="additive", period=90, extrapolate_trend=180)
        trend_df[col] = decomposition.trend
    past_data_df[numeric_cols] = deseason(past_data_df[numeric_cols], season_df) # deseasonalize
    past_data_df[numeric_cols] = scaler.transform(past_data_df[numeric_cols])
    X = get_input_sequences(past_data_df)
    
    # forecast 180 days ahead
    forecast_dates = [past_data_df.index[-1] + timedelta(days=i) for i in range(1, 180 + 1)]
    preds = forecast_model.predict(X)
    preds = compress_to_timestep(preds)[-180:]
    preds_df = pd.DataFrame(preds, index=pd.to_datetime(forecast_dates), columns=numeric_cols)
    preds_df.index = preds_df.index.date
    preds_df[numeric_cols] = revert_deseason(preds_df[numeric_cols], season_df)
    preds_df = revert_detrend(preds_df, trend_scaler=trend_scaler, trend_df=trend_df, trend_model=trend_model)
    
    st.subheader("Forecasted Values")
    st.dataframe(preds_df, use_container_width=True)
    
    for col in numeric_cols:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual_past_data_df.index,
            y=actual_past_data_df[col],
            mode="lines",
            name="Past Data"
        ))
        fig.add_trace(go.Scatter(
            x=preds_df.index,
            y=preds_df[col],
            mode="lines",
            name="Forecasted"
        ))
        fig.update_layout(
            title=f"Forecasts for {col}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)