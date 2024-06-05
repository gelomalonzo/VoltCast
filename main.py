import streamlit as st
from datetime import timedelta
import io

import pandas as pd
import numpy as np
import joblib
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

def get_input_sequences(df:pd.DataFrame):
    X = []
    for i in range(len(df) - 180 + 1):
        X.append(df.iloc[i : i + 180].values)
    return np.array(X, dtype=np.float64)

def compress_to_timestep(seq_arr:np.ndarray):
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
    
    about_col, mid_col, instructions_col = st.columns((0.375, 0.050, 0.575))
    with about_col:
        st.subheader("About")
        st.write("\
            **VoltCast** is a simple web application designed for time series \
            electricity forecasting. It integrates the recurrent neural \
            networks GRU and LSTM to forecast electricity demand, price, and \
            supply for up to 180 days ahead. Its goal is to incorporate deep \
            learning in forecasting electricity data in the Philippines to \
            provide insights that could help in the electricity operations \
            (e.g., management, maintenance, costs) through its forecasts.\
        ")
        st.write("\
            This web application was developed by Angelo Malonzo \
            (abmalonzo@up.edu.ph) as an output of a comparative study on \
            deep learning algorithms for electricity forecasting.\
        ")
    with instructions_col:
        st.subheader("Instructions")
        st.write("\
            The sidebar on the left contains the input form where \
            users can upload the input data. \
        ")
        st.write("\
            The first file upload bin is allotted for uploading the CSV file of \
            the past data to be used as input to the model for it to draw its forecasts.\
        ")
        st.write("\
            The second file upload bin is for uploading the CSV file of the actual \
            data if available. This is mainly for comparing the forecasts with the \
            actual values to check the performance.\
        ")
        st.write("\
            Once the input files are uploaded, click the **Forecast** button below \
            the file upload bins to generate forecasts. The Results page will then \
            be displayed showing the table and plots of the generated forecasts. \
            Users can download the forecast table as a CSV file and the plots as PNG files.\
        ")
        st.subheader("Notes")
        st.write("\
            1. The input file for the historical data is required.\
        ")
        st.write("\
            2. The historical data must have at least 180 days of \
            (consecutive) electricity data.\
        ")
        st.write("\
            3. The input data (historical and actual) must be in \
            daily time steps."\
        )
        st.write("\
            4. The input data (historical and actual) must have the following \
            columns: DATE, BLOCK, LUZVIS PRICE, LUZON DEMAND, VISAYAS DEMAND, \
            LUZON SUPPLY, VISAYAS SUPPLY.\
        ")
        st.write("5. Ensure that there are no missing and absurd input data values.")

# RESULTS PAGE
if st.session_state.show_results:
    results_msg = st.container()
    st.header(":chart_with_upwards_trend: Results")
    st.write("---")
    
    # store input files to DataFrame
    past_data_text = past_data_file.read()
    past_data_text_str = str(past_data_text, "utf-8")
    past_data_df = pd.read_csv(io.StringIO(past_data_text_str))
    past_data_df.columns = past_data_df.columns\
                           .str.upper()\
                           .str.strip()\
                           .str.replace(r'\s+', ' ', regex=True)
    past_data_df.replace(',', '', regex=True, inplace=True)
    past_data_df["DATE"] = pd.to_datetime(past_data_df["DATE"])
    past_data_df["DATE"] = past_data_df["DATE"].dt.date
    past_data_df.set_index("DATE", inplace=True)
    if actual_data_file:
        actual_data_text = actual_data_file.read()
        actual_data_text_str = str(actual_data_text, "utf-8")
        actual_data_df = pd.read_csv(io.StringIO(actual_data_text_str))
        actual_data_df.columns = actual_data_df.columns\
                                 .str.upper()\
                                 .str.strip()\
                                 .str.replace(r'\s+', ' ', regex=True)
        actual_data_df.replace(',', '', regex=True, inplace=True)
        actual_data_df["DATE"] = pd.to_datetime(actual_data_df["DATE"])
        actual_data_df["DATE"] = actual_data_df["DATE"].dt.date
        actual_data_df.set_index("DATE", inplace=True)
    
    # import models
    numeric_cols = [
        "LUZVIS PRICE (PHP/KWH)", 
        "LUZON DEMAND (MW)", 
        "VISAYAS DEMAND (MW)", 
        "LUZON SUPPLY (MW)", 
        "VISAYAS SUPPLY (MW)"
    ]
    scaler = joblib.load("assets/scaler2.pkl")
    # for "LUZON DEMAND (MW)", "VISAYAS DEMAND (MW)", "LUZON SUPPLY (MW)"
    init_gru_model = load_model("assets/init_gru.h5") 
    # for "LUZVIS PRICE (PHP/KWH)"
    tuned_gru_model = load_model("assets/tuned_gru.h5") 
    # for "VISAYAS SUPPLY (MW)"
    init_lstm_model = load_model("assets/init_lstm.h5") 
    
    # preprocess
    actual_past_data_df = past_data_df.copy()
    past_data_df = pd.get_dummies(past_data_df, columns=["BLOCK"])
    past_data_df[numeric_cols] = scaler.transform(past_data_df[numeric_cols])
    X = get_input_sequences(past_data_df)
    
    # forecast 180 days ahead
    forecast_dates = [past_data_df.index[-1] + timedelta(days=i) for i in range(1, 180 + 1)]
    
    init_gru_preds = init_gru_model.predict(X)
    init_gru_preds = compress_to_timestep(init_gru_preds)[-180:]
    init_gru_preds_df = pd.DataFrame(
        init_gru_preds, 
        index=pd.to_datetime(forecast_dates), 
        columns=numeric_cols
    )
    init_gru_preds_df.index = init_gru_preds_df.index.date
    init_gru_preds_df[numeric_cols] = scaler.inverse_transform(init_gru_preds_df[numeric_cols])
    
    tuned_gru_preds = tuned_gru_model.predict(X)
    tuned_gru_preds = compress_to_timestep(tuned_gru_preds)[-180:]
    tuned_gru_preds_df = pd.DataFrame(
        tuned_gru_preds, 
        index=pd.to_datetime(forecast_dates), 
        columns=numeric_cols
    )
    tuned_gru_preds_df.index = tuned_gru_preds_df.index.date
    tuned_gru_preds_df[numeric_cols] = scaler.inverse_transform(tuned_gru_preds_df[numeric_cols])
    
    init_lstm_preds = init_lstm_model.predict(X)
    init_lstm_preds = compress_to_timestep(init_lstm_preds)[-180:]
    init_lstm_preds_df = pd.DataFrame(
        init_lstm_preds, 
        index=pd.to_datetime(forecast_dates), 
        columns=numeric_cols
    )
    init_lstm_preds_df.index = init_lstm_preds_df.index.date
    init_lstm_preds_df[numeric_cols] = scaler.inverse_transform(init_lstm_preds_df[numeric_cols])
    
    preds_df = init_gru_preds_df.copy()
    # preds_df["LUZON DEMAND (MW)"] = init_gru_preds_df["LUZON DEMAND (MW)"]
    # preds_df["VISAYAS DEMAND (MW)"] = init_gru_preds_df["VISAYAS DEMAND (MW)"]
    preds_df["LUZVIS PRICE (PHP/KWH)"] = init_lstm_preds_df["LUZVIS PRICE (PHP/KWH)"]
    # preds_df["LUZON SUPPLY (MW)"] = init_gru_preds_df["LUZON SUPPLY (MW)"]
    preds_df["VISAYAS SUPPLY (MW)"] = init_lstm_preds_df["VISAYAS SUPPLY (MW)"]
    
    del init_gru_preds, init_gru_preds_df
    del tuned_gru_preds, tuned_gru_preds_df
    del init_lstm_preds, init_lstm_preds_df
    
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
        if actual_data_file:
            fig.add_trace(go.Scatter(
                x=actual_data_df.index,
                y=actual_data_df[col],
                mode="lines",
                name="Actual"
            ))
        fig.update_layout(
            title=f"Forecasts for {col}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        