# 📅 Forecast Summary by Quarter
st.subheader("📋 Forecast Summary by Quarter (in $000s)")

forecast_summary = forecast[["ds", "yhat"]].copy()
forecast_summary = forecast_summary[forecast_summary["ds"] > df["ds"].max()]  # Only future
forecast_summary["Quarter"] = forecast_summary["ds"].dt.to_period("Q")
quarterly = (
    forecast_summary.groupby("Quarter")["yhat"]
    .sum()
    .reset_index()
    .rename(columns={"yhat": "Forecasted Revenue"})
)

# Add past actuals for comparison
historical = df[["ds", "y"]].copy()
historical["Quarter"] = historical["ds"].dt.to_period("Q")
historical_quarterly = (
    historical.groupby("Quarter")["y"]
    .sum()
    .reset_index()
    .rename(columns={"y": "Actual Revenue"})
)

# Combine historical and forecast
combined = pd.merge(quarterly, historical_quarterly, on="Quarter", how="left")
combined["Revenue ($000s)"] = (combined["Forecasted Revenue"] / 1000).round(1)

# Calculate % change vs prior quarter
combined["QoQ % Change"] = combined["Forecasted Revenue"].pct_change() * 100

# Calculate % change vs same quarter last year
combined["YoY % Change"] = combined["Forecasted Revenue"].pct_change(periods=4) * 100

# Format percentage columns
combined["QoQ % Change"] = combined["QoQ % Change"].round(1)
combined["YoY % Change"] = combined["YoY % Change"].round(1)

# Select and rename columns for display
summary_table = combined[["Quarter", "Revenue ($000s)", "QoQ % Change", "YoY % Change"]]

# Display
st.dataframe(summary_table)
