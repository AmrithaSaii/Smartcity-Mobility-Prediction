import streamlit as st
import pandas as pd
import numpy as np
import pickle, json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bangalore Mobility Dashboard",
    page_icon="🚖",
    layout="wide"
)

@st.cache_resource
def load_models():
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("models/features.json", "r") as f:
        features = json.load(f)
    return model, le, features

@st.cache_data
def load_data():
    demand    = pd.read_csv("data/demand_features.csv")
    streaming = pd.read_csv("data/streaming_predictions.csv")
    return demand, streaming

model, le, FEATURES = load_models()
demand_df, streaming_df = load_data()

def get_season(month):
    if month in [6,7,8,9]: return 2
    elif month in [3,4,5]: return 1
    else: return 0

def predict_demand(zone, hour, day_of_week, month,
                   is_weekend, temperature, precipitation, windspeed):
    zone_encoded = int(le.transform([zone])[0])
    row = {
        "zone_encoded":       zone_encoded,
        "hour":               hour,
        "day_of_week":        day_of_week,
        "month":              month,
        "season":             get_season(month),
        "is_weekend":         is_weekend,
        "is_peak_hour":       1 if (8<=hour<=10 or 17<=hour<=20) else 0,
        "is_raining":         1 if precipitation > 0 else 0,
        "is_heavy_rain":      1 if precipitation > 5 else 0,
        "temperature_real":   temperature,
        "precipitation_real": precipitation,
        "windspeed_real":     windspeed
    }
    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    return round(float(model.predict(X)[0]), 2)

# ── Sidebar ───────────────────────────────────────────
st.sidebar.title("🚖 Bangalore Mobility")
st.sidebar.markdown("**Urban Transport Demand Prediction**")
st.sidebar.markdown("*Big Data Analytics — 2025*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔮 Live Prediction",
    "📊 Batch Analytics",
    "📡 Streaming Simulation"
])

zones = list(le.classes_)

# ════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🚖 City-Scale Mobility Demand Prediction")
    st.markdown("### Bangalore Urban Transport Analytics Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rides Processed",   "1,035,200+", "Bangalore 2025 Dataset")
    col2.metric("Zones Monitored",   "15",         "Key Bangalore Areas")
    col3.metric("Model Accuracy",    "R² = 0.76",  "XGBoost Best Model")
    col4.metric("Peak Demand Zone",  "Whitefield", "27.93 rides/hr peak")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔧 System Pipeline")
        st.info("""
        **Data Processing**
        Apache PySpark processes 1M+ mobility records
        across 15 Bangalore zones

        **Batch Analytics**
        Historical demand patterns by hour, zone,
        weather and city events

        **ML Prediction**
        Random Forest vs XGBoost comparison
        Best model: XGBoost (R² = 0.76)

        **Streaming Simulation**
        Spark Structured Streaming simulation
        with real-time surge alerts
        """)

    with col2:
        st.markdown("### 📌 Key Findings")
        st.success("""
        🕐 Peak demand hour: **18:00** (evening rush)

        📍 Highest demand zones: **Whitefield,
        Electronic City, Marathahalli**

        🌧️ Heavy rain causes **1.13x demand surge**

        📅 Weekday demand **15% higher** than weekend

        🎉 IPL match days cause **2x surge** near
        MG Road & Indiranagar

        🏢 Bengaluru Tech Summit drives **1.8x surge**
        in Electronic City & Whitefield
        """)

    st.markdown("---")
    st.markdown("### 📦 Data Sources")
    c1, c2, c3 = st.columns(3)
    c1.info("🚖 **Ride Data**\nBangalore mobility dataset\n1,035,200 ride records\n15 city zones")
    c2.info("🌤️ **Weather Data**\nOpen-Meteo API\nHourly temperature,\nprecipitation & wind")
    c3.info("🚌 **Bus Network**\nBMTC stop data\n2,975 bus stops\nReal GPS coordinates")

# ════════════════════════════════════════════════════════
# PAGE 2: LIVE PREDICTION
# ════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Demand Prediction")
    st.markdown("Predict ride demand for any zone, time, weather and event condition.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Parameters")
        zone  = st.selectbox("📍 Pickup Zone", zones)
        hour  = st.slider("🕐 Hour of Day", 0, 23, 9)
        day   = st.selectbox("📅 Day of Week",
                    ["Monday","Tuesday","Wednesday",
                     "Thursday","Friday","Saturday","Sunday"])
        month = st.selectbox("📆 Month",
                    ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"])

        is_weekend = 1 if day in ["Saturday","Sunday"] else 0
        month_num  = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"].index(month) + 1

        st.markdown("#### 🌤️ Weather Conditions")
        temperature   = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, 0.5)
        precipitation = st.slider("Precipitation (mm)", 0.0, 20.0, 0.0, 0.5)
        windspeed     = st.slider("Wind Speed (km/h)", 0.0, 30.0, 12.0, 0.5)

        st.markdown("#### 🎉 City Events")
        event = st.selectbox("Event Type",
                    ["None", "IPL Match", "Festival", "Public Holiday", "Tech Event"])

    with col2:
        st.subheader("Prediction Result")

        # Event boost on top of model prediction
        pred = predict_demand(
            zone, hour,
            ["Monday","Tuesday","Wednesday","Thursday",
             "Friday","Saturday","Sunday"].index(day)+1,
            month_num, is_weekend,
            temperature, precipitation, windspeed
        )

        # Apply event multiplier
        event_multiplier = 1.0
        event_note = ""
        if event == "IPL Match" and hour >= 18:
            if zone in ["MG Road", "Indiranagar", "Koramangala"]:
                event_multiplier = 2.0
                event_note = "⚠️ IPL match — major surge near stadium zones"
            else:
                event_multiplier = 1.3
                event_note = "⚠️ IPL match — moderate city-wide surge"
        elif event == "Festival":
            event_multiplier = 1.5
            event_note = "⚠️ Festival day — significant demand increase"
        elif event == "Public Holiday":
            event_multiplier = 1.2
            event_note = "ℹ️ Public holiday — moderate demand increase"
        elif event == "Tech Event":
            if zone in ["Whitefield", "Electronic City", "Bellandur"]:
                event_multiplier = 1.8
                event_note = "⚠️ Tech event — high surge in tech zones"
            else:
                event_multiplier = 1.1

        pred = round(pred * event_multiplier, 2)

        if pred >= 15:
            st.error(f"🔴 HIGH DEMAND\n\n## {pred} rides/hr")
            alert = "HIGH"
        elif pred >= 10:
            st.warning(f"🟡 MEDIUM DEMAND\n\n## {pred} rides/hr")
            alert = "MEDIUM"
        else:
            st.success(f"🟢 NORMAL DEMAND\n\n## {pred} rides/hr")
            alert = "NORMAL"

        if event_note:
            st.warning(event_note)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={"text": f"Predicted Demand — {zone}"},
            gauge={
                "axis": {"range": [0, 35]},
                "bar":  {"color": "darkblue"},
                "steps": [
                    {"range": [0,  10], "color": "#d4edda"},
                    {"range": [10, 15], "color": "#fff3cd"},
                    {"range": [15, 35], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 15
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Context")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Zone | {zone} |
        | Hour | {hour}:00 |
        | Day | {day} ({month}) |
        | Weather | {'🌧️ Rainy' if precipitation>2 else '☀️ Clear'} |
        | Event | {event} |
        | Event Boost | {event_multiplier}x |
        | Alert Level | {alert} |
        """)

# ════════════════════════════════════════════════════════
# PAGE 3: BATCH ANALYTICS
# ════════════════════════════════════════════════════════
elif page == "📊 Batch Analytics":
    st.title("📊 Batch Analytics")
    st.markdown("Historical demand patterns from Bangalore 2025 mobility data.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "⏰ Hourly Patterns",
        "📍 Zone Analysis",
        "🌤️ Weather Impact",
        "🗺️ Heatmap"
    ])

    with tab1:
        st.image("plots/hourly_demand.png", use_container_width=True)
        hourly = demand_df.groupby("hour")["demand"].mean()
        peak   = hourly.idxmax()
        st.info(f"**Peak demand hour: {peak}:00** with average "
                f"{hourly[peak]:.1f} rides per zone")

    with tab2:
        st.image("plots/zone_demand.png",       use_container_width=True)
        st.image("plots/weekday_vs_weekend.png", use_container_width=True)

    with tab3:
        st.image("plots/weather_impact.png", use_container_width=True)
        rain_avg   = demand_df[demand_df["precipitation_real"]>5]["demand"].mean()
        norain_avg = demand_df[demand_df["precipitation_real"]==0]["demand"].mean()
        st.info(f"**Rain surge factor: {rain_avg/norain_avg:.2f}x** — "
                f"Heavy rain: {rain_avg:.1f} rides vs No rain: {norain_avg:.1f} rides")

    with tab4:
        st.image("plots/zone_hour_heatmap.png", use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 4: STREAMING
# ════════════════════════════════════════════════════════
elif page == "📡 Streaming Simulation":
    st.title("📡 Streaming Simulation")
    st.markdown("Simulated Spark Structured Streaming — demand predictions across time windows.")
    st.markdown("---")

    high   = len(streaming_df[streaming_df["surge_alert"]=="🔴 HIGH"])
    medium = len(streaming_df[streaming_df["surge_alert"]=="🟡 MEDIUM"])
    normal = len(streaming_df[streaming_df["surge_alert"]=="🟢 NORMAL"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Windows Processed", streaming_df["window"].nunique())
    c2.metric("🔴 High Alerts",    high)
    c3.metric("🟡 Medium Alerts",  medium)
    c4.metric("🟢 Normal",         normal)

    st.markdown("---")
    st.image("plots/streaming_results.png", use_container_width=True)

    st.markdown("### 📋 Streaming Predictions Table")
    hour_filter = st.slider("Filter by Hour", 6, 22, (8, 20))
    filtered = streaming_df[
        streaming_df["hour"].between(hour_filter[0], hour_filter[1])
    ].sort_values("predicted_demand", ascending=False)

    st.dataframe(
        filtered[["hour","zone","predicted_demand",
                  "precipitation","temperature","surge_alert"]]
        .reset_index(drop=True),
        use_container_width=True
    )