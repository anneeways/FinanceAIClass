import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="📈 Revenue Forecasting Agent", page_icon="📉", layout="wide")

st.title("📈 AI-Powered Revenue Forecasting with Prophet")

# Upload Excel File
uploaded_file = st.file_uploader("📤 Upload Excel file with 'Date' and 'Revenue' columns", type=["xlsx", "xls"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Validate required columns
        if "Date" not in df.columns or "Revenue" not in df.columns:
            st.error("❌ The uploaded file must contain 'Date' and 'Revenue' columns.")
            st.stop()

        # Data preprocessing
        df = df[["Date", "Revenue"]].copy()
        df.columns = ["ds", "y"]  # Prophet requires 'ds' and 'y'
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")

        st.success("✅ Data loaded successfully!")

        # Show raw data
        with st.expander("🔍 Preview Uploaded Data"):
            st.dataframe(df)

        # Train Prophet Model
        model = Prophet()
        model.fit(df)

        # Forecasting
        periods_input = st.slider("🔮 Forecast Horizon (months)", 1, 24, 12)
        future = model.make_future_dataframe(periods=periods_input * 30)
        forecast = model.predict(future)

        # Plot forecast
        st.subheader("📊 Forecasted Revenue")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Components (trend, seasonality)
        st.subheader("🔍 Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # 📅 Forecast Summary by Quarter
        st.subheader("📋 Forecast Summary by Quarter (in $000s)")

        forecast_summary = forecast[["ds", "yhat"]].copy()
        forecast_summary = forecast_summary[forecast_summary["ds"] > df["ds"].max()]  # Only future
        forecast_summary["Quarter"] = forecast_summary["ds"].dt.to_period("Q")
        quarterly_summary = (
            forecast_summary.groupby("Quarter")["yhat"]
            .sum()
            .reset_index()
            .rename(columns={"yhat": "Forecasted Revenue ($000s)"})
        )
        quarterly_summary["Forecasted Revenue ($000s)"] = (quarterly_summary["Forecasted Revenue ($000s)"] / 1000).round(1)

        st.dataframe(quarterly_summary)

        # AI Analysis with GROQ
        st.subheader("🤖 AI-Generated Forecast Commentary")

        data_for_ai = df.tail(60).to_json(orient="records")

        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        You are an FP&A expert. Based on the historical revenue data provided, generate:
        - Key insights about revenue trends.
        - Any seasonality or anomalies.
        - Forecast outlook based on the Prophet model.
        - Actionable recommendations for leadership.

        Here is the recent dataset in JSON format:
        {data_for_ai}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial forecasting expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"
        )

        ai_commentary = response.choices[0].message.content
        st.markdown(ai_commentary)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
