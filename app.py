import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import plotly.express as px

from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TinyTimeMixerForPrediction,
)
from tsfm_public.toolkit.visualization import plot_predictions

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Energy Consumption Forecast [UPDATED]",
    page_icon="⚡",
    layout="wide",
)

# ------------------ CONFIG ------------------
DATA_FILE_PATH = "hf://datasets/vitaliy-sharandin/energy-consumption-hourly-spain/energy_dataset.csv"
timestamp_column = "time"
default_targets = ["total load actual", "price actual"]

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[timestamp_column])
    df = df.ffill()
    return df

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_pipeline(num_targets):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",
        num_input_channels=num_targets,
    )
    pipeline = TimeSeriesForecastingPipeline(
        model,
        timestamp_column=timestamp_column,
        id_columns=[],
        target_columns=target_columns,
        explode_forecasts=False,
        freq="h",
        device=device,
    )
    return pipeline

# ------------------ STREAMLIT APP ------------------
st.title("⚡ Energy Consumption Forecast Dashboard")
st.markdown(
    """
    Welcome to the **Energy Consumption & Price Forecasting Dashboard**.  
    This tool uses **IBM Granite TinyTimeMixers (TTM)** for deep learning–based forecasting.  
    """
)

st.divider()

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Configuration")
context_length = st.sidebar.number_input(
    "Context length", min_value=32, max_value=1024, value=512, step=32
)

# Load data
df = load_data()
st.write("### Last few rows of dataset")
st.dataframe(df.tail())

# All possible targets (exclude timestamp)
all_possible_targets = [col for col in df.columns if col != timestamp_column]

# Multiselect
target_columns = st.sidebar.multiselect(
    "Select target columns",
    options=all_possible_targets,
    default=[col for col in default_targets if col in all_possible_targets],
)

# ------------------ TABS ------------------
tab1, tab2 = st.tabs(["📂 Dataset", "📈 Forecasting"])

# ------------------ TAB 1: DATASET ------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.tail(20), use_container_width=True)

    # Quick stats
    st.markdown("### 🔍 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    col2.metric("Start Date", str(df[timestamp_column].min().date()))
    col3.metric("End Date", str(df[timestamp_column].max().date()))

# ------------------ TAB 2: FORECAST ------------------
with tab2:
    st.subheader("Forecasting")
    if st.button("🚀 Run Forecast"):
        if not target_columns:
            st.error("⚠️ Please select at least one target column.")
            st.stop()

        # Prepare input data
        input_df = df.iloc[
            -context_length:,
            [df.columns.get_loc(timestamp_column)] + [df.columns.get_loc(c) for c in target_columns],
        ]

        pipeline = load_pipeline(num_targets=len(target_columns))

        with st.spinner("Running forecast with TinyTimeMixers..."):
            forecast_df = pipeline(input_df)

        st.success("✅ Forecast completed")

        # Metrics
        latest_time = df[timestamp_column].max()
        st.markdown("### 📊 Key Metrics (Latest Actual Values)")
        kpi_cols = st.columns(len(target_columns))
        for i, col in enumerate(target_columns):
            latest_val = df[col].iloc[-1]
            kpi_cols[i].metric(col, f"{latest_val:,.2f}")

        st.divider()

        # ---------------- Matplotlib version ----------------
        #st.markdown("### 📈 Forecast Results (Matplotlib)")
        #for col in target_columns:
        #    st.markdown(f"#### 🔹 {col}")
        #    plot_predictions(
        #        input_df=input_df,
        #        predictions_df=forecast_df,
        #        freq="h",
        #        timestamp_column=timestamp_column,
        #        channel=col,
        #        indices=[-1],
        #        num_plots=1,
        #    )
        #    st.pyplot(plt.gcf())
        #    plt.clf()
        #    st.caption(f"Forecast generated using {context_length} past steps.")

        #st.divider()

        # ---------------- Plotly version ----------------
        st.markdown("### 📊 Forecast Results (Interactive Plotly)")
        for col in target_columns:
            st.markdown(f"#### 🔹 {col}")

            # Actual data
            actual_df = input_df[[timestamp_column, col]].copy()
            actual_df["type"] = "Actual"

            # Forecast values
            forecast_col = f"{col}_prediction"
            if forecast_col not in forecast_df.columns:
                st.warning(f"⚠️ No forecast column found for {col}")
                continue

            raw_forecast = forecast_df[forecast_col].iloc[0]

            # Expand forecasts if stored as list/array
            if isinstance(raw_forecast, (list, tuple)):
                forecast_values = raw_forecast
            else:
                forecast_values = forecast_df[forecast_col].values

            # Generate future timestamps
            last_time = input_df[timestamp_column].iloc[-1]
            future_times = pd.date_range(
                start=last_time + pd.Timedelta(hours=1),
                periods=len(forecast_values),
                freq="h"
            )

            forecast_part = pd.DataFrame({
                timestamp_column: future_times,
                col: forecast_values,
                "type": "Forecast"
            })

            # Merge actual + forecast
            plot_df = pd.concat([actual_df, forecast_part], ignore_index=True)
           

            # Ensure timezone-naive
            if pd.api.types.is_datetime64_any_dtype(plot_df[timestamp_column]):
                plot_df[timestamp_column] = plot_df[timestamp_column].dt.tz_localize(None)

            # Plot
            fig = px.line(
                plot_df,
                x=timestamp_column,
                y=col,
                color="type",
                title=f"Forecast vs Actual for {col}",
                markers=True,
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title=col,
                legend_title="Data Type",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- Download buttons ----------------
        st.divider()
        csv_data = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecast (CSV)",
            data=csv_data,
            file_name="forecast_results.csv",
            mime="text/csv",
        )
