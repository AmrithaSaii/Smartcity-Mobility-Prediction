# Smartcity-Mobility-Prediction

City-Scale Mobility Demand Prediction Using Big Data Analytics
Bangalore Urban Transport Analytics - Big Data Project

## About
This project predicts ride-hailing demand across 15 Bangalore zones
using Big Data Analytics. It processes 1 million+ mobility records
using Apache PySpark, trains machine learning models, and provides
a real-time demand prediction dashboard.

## Pipeline
1. Data Processing - Apache PySpark processes 1M+ ride records
2. Batch Analytics - Peak hours, zone patterns, weather and event impact
3. ML Models - Random Forest vs XGBoost, best R2 score = 0.76
4. Streaming Simulation - Spark Structured Streaming simulation
5. Dashboard - Streamlit interactive web app

## How to Run

Step 1 - Clone the repository
git clone https://github.com/YOURUSERNAME/Smartcity-Mobility-Prediction.git
cd smartcity-mobility-prediction

Step 2 - Install dependencies
pip install -r requirements.txt

Step 3 - Run the dashboard
python -m streamlit run app.py
Open browser at http://localhost:8501

## Project Structure
SMARTCITY/
    app.py                      main dashboard application
    requirements.txt            project dependencies
    data/
        demand_features.csv     processed zone-hour demand data
        merged_data.csv         merged ride and weather data
        streaming_predictions.csv   streaming simulation results
    models/
        xgb_model.pkl           trained XGBoost model
        label_encoder.pkl       zone label encoder
        features.json           model feature list
    plots/
        hourly_demand.png
        zone_demand.png
        weekday_vs_weekend.png
        weather_impact.png
        zone_hour_heatmap.png
        streaming_results.png

## Tech Stack
- Apache PySpark
- XGBoost and Scikit-learn
- Streamlit
- Plotly
- Pandas and NumPy
- Open-Meteo Weather API
- BMTC Bus Network Data
