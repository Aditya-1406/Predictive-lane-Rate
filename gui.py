import streamlit as st
import pandas as pd 
import requests
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

st.set_page_config(page_title="🚚 Lane Recommendation Dashboard", layout="wide")

# -------------------- STYLING --------------------
st.markdown("""
<style>
body, .stApp { background-color: #F5F7FA; color: #242547 !important; font-family: "Segoe UI", sans-serif; }
.section-header { padding: 15px 25px; border-radius: 15px; font-weight: 600; margin-top: 25px; box-shadow: 0px 4px 18px rgba(0,0,0,0.15); font-size: 22px; color: #242547; text-align:center;}
.metric-card { padding: 25px; margin: 10px; border-radius: 22px; text-align: center; font-weight: 600; width: 180px; height: 160px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0px 6px 18px rgba(0,0,0,0.15); color: white; font-size: 18px; }
.metric-card b { font-size: 28px; }
.metric-purple { background: linear-gradient(145deg, #6A1B9A, #AB47BC); }
.metric-green { background: linear-gradient(145deg, #43A047, #66BB6A); }
.metric-blue { background: linear-gradient(145deg, #1E88E5, #64B5F6); }
.metric-orange { background: linear-gradient(145deg, #FF8F00, #FFA726); }
.metric-pink { background: linear-gradient(145deg, #EC407A, #F48FB1); }
.metric-yellow { background: linear-gradient(145deg, #FFD93D, #FBC02D); color:#242547;}
div.stButton > button { background: linear-gradient(90deg, #FF6B6B, #FFD93D); color: #242547; font-weight: 600; border-radius: 12px; border: none; padding: 12px 30px; transition: 0.3s ease-in-out; font-size: 16px;}
div.stButton > button:hover { background: linear-gradient(90deg, #43A047, #66BB6A); color: white;}
footer { visibility: visible; padding: 10px; text-align: center; color: #242547; font-weight: 500; font-size: 14px; margin-top: 50px; }
</style>
""", unsafe_allow_html=True)

st.title("🚚 Lane Recommendation & Prediction Dashboard")

# -------------------- LOAD DATA --------------------
df_filter = pd.read_csv("lane_cost_cleaned_final.csv")
cities_df = pd.read_csv("standard_us_city_lanes_with_speed1(in).csv")
cities_df.columns = cities_df.columns.str.strip()
if 'shipment_id' not in cities_df.columns:
    cities_df.insert(0, 'shipment_id', [f"SH_{i:03}" for i in range(1, len(cities_df)+1)])

# -------------------- FEATURE PREPROCESSING --------------------
object_cols = df_filter.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df_filter, columns=object_cols, drop_first=True)
X = df_encoded.drop(columns=['total_cost_usd','lane_id_corrected','avg_speed_mph'])
y_cost = df_encoded['total_cost_usd']
y_lane = df_encoded['lane_id_corrected']
y_speed = df_encoded['avg_speed_mph']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
columns_used = X.columns
joblib.dump(columns_used, 'columns_used.pkl')

# -------------------- ALGORITHM MAPPINGS --------------------
regressor_algos = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
}

classifier_algos = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("📥 Shipment Input Features")
selected_shipment = st.sidebar.selectbox("Select Shipment ID", df_filter['shipment_id'].tolist())
axle_count = st.sidebar.selectbox("Axle Count", [3,4,5,6], index=1)

# st.sidebar.header("🧠 Choose Algorithms")
# cost_algo_name = st.sidebar.selectbox("Cost Model", list(regressor_algos.keys()), index=0)
# lane_algo_name = st.sidebar.selectbox("Lane Model", list(classifier_algos.keys()), index=0)
# speed_algo_name = st.sidebar.selectbox("Speed Model", list(regressor_algos.keys()), index=0)

# Default API keys
TOMTOM_API_KEY_DEFAULT = "VoMgKXVjYZAnjRwKmU8C9OxkykqSWeAM"
EIA_API_KEY_DEFAULT = "OQdfxEcYk9VyPNOoNjHkymVqMK0stggKOtRcHPCn"

tomtom_api_key = st.sidebar.text_input("TomTom API Key", value=TOMTOM_API_KEY_DEFAULT)
eia_api_key = st.sidebar.text_input("EIA API Key", value=EIA_API_KEY_DEFAULT)

# -------------------- CONSTANTS --------------------
weight_of_truck_tons = 12.8
fuel_efficiency_mpg = 6.19
hours_per_day = 8
lane_type_str_map = {1:'slow', 2:'normal', 3:'fast'}
weather_speed_reduction = {'Clear':0, 'Fog':2, 'Rain':3, 'Snow':4, 'Storm':5}
axle_cost_addition = {3:0,4:5.73,5:13.71,6:19.42}
coordinate_cache = {}

def traffic_speed_penalty(traffic_index):
    if traffic_index <= 3: return 0
    elif 4 <= traffic_index <= 6: return 2
    elif 7 <= traffic_index <= 8: return 4
    else: return 6

def get_coordinates(city_name, tomtom_api_key, use_api=True):
    if not use_api: return None, None
    if city_name in coordinate_cache: return coordinate_cache[city_name]
    try:
        res = requests.get(f"https://api.tomtom.com/search/2/geocode/{city_name}.json",
            params={"storeResult":"false","view":"Unified","limit":1,"key":tomtom_api_key}).json()
        if res["results"]:
            pos = res["results"][0]["position"]
            coordinate_cache[city_name] = (pos["lat"], pos["lon"])
            return pos["lat"], pos["lon"]
    except: pass
    return None, None

def get_weather_condition(city_name, tomtom_api_key, use_api=True):
    if not use_api: return "Clear"
    lat, lon = get_coordinates(city_name, tomtom_api_key, use_api)
    if lat is None: return "Clear"
    try:
        res = requests.get("https://api.open-meteo.com/v1/forecast",
                           params={"latitude":lat,"longitude":lon,"current_weather":True}).json()
        code = res["current_weather"]["weathercode"]
        if code in [71,73,75,77]: return "Snow"
        elif code in [95,96,99]: return "Storm"
        elif code in [61,63,65,80,81,82]: return "Rain"
        elif code in [45,48]: return "Fog"
        else: return "Clear"
    except: return "Clear"

def get_traffic_index(city_name, tomtom_api_key, use_api=True):
    if not use_api: return 5
    lat, lon = get_coordinates(city_name, tomtom_api_key, use_api)
    if lat is None: return 5
    try:
        res = requests.get("https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json",
            params={"point":f"{lat},{lon}","unit":"MPH","key":tomtom_api_key}).json().get("flowSegmentData",{})
        current_speed = res.get("currentSpeed")
        free_speed = res.get("freeFlowSpeed")
        if current_speed and free_speed:
            ratio = free_speed/max(current_speed,1)
            return min(10,max(1,round(ratio*5)))
    except: return 5
    return 5

def get_latest_fuel_price(eia_api_key, use_api=True):
    if not use_api: return 4.04
    try:
        url="https://api.eia.gov/v2/petroleum/pri/gnd/data/"
        params={"frequency":"weekly","data[0]":"value","sort[0][column]":"period","sort[0][direction]":"desc","length":1,"api_key":eia_api_key}
        res = requests.get(url,params=params).json()
        return float(res["response"]["data"][0]["value"])
    except: return 4.04

# -------------------- TRAIN MODELS ONCE --------------------
@st.cache_resource
def train_models():
    rf_cost = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y_cost)
    knn_lane = KNeighborsClassifier(n_neighbors=5).fit(X, y_lane)
    rf_speed = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y_speed)
    return rf_cost, knn_lane, rf_speed

rf_cost, knn_lane, rf_speed = train_models()

# -------------------- RECOMMENDATION FUNCTION --------------------
def recommend_best_lane(shipment_id, axle_count, use_api=True):
    shipment = df_filter[df_filter['shipment_id']==shipment_id]
    if shipment.empty: return None
    origin = shipment.iloc[0]['origin_city']
    destination = shipment.iloc[0]['destination_city']
    target_days = shipment.iloc[0].get('delivery_target_days',1.5)

    df = cities_df[(cities_df['origin_city']==origin)&(cities_df['dest_city']==destination)]
    if df.empty: return None

    distance = df.iloc[0]['distance_miles_est']
    required_speed = distance/(target_days*hours_per_day)
    weather = get_weather_condition(origin, tomtom_api_key, use_api)
    traffic = get_traffic_index(origin, tomtom_api_key, use_api)
    fuel_price = get_latest_fuel_price(eia_api_key, use_api)

    adjusted_speed = max(30, required_speed-(weather_speed_reduction.get(weather,0)+traffic_speed_penalty(traffic)))
    lane_type = 1 if adjusted_speed<60 else (2 if adjusted_speed<70 else 3)
    selected_lane = lane_type_str_map[lane_type]

    row = df[df['lane_type'].str.lower()==selected_lane].iloc[0]
    base_cost = row['total_cost_usd']
    fuel_cost = (distance/fuel_efficiency_mpg)*fuel_price
    total_cost = base_cost + fuel_cost + axle_cost_addition.get(axle_count or 4,0)

    suggestions=[]
    if weather in ['Rain','Snow','Storm']: suggestions.append("⚠️ Poor weather — prefer slower lanes.")
    if traffic>=8: suggestions.append("🚦 Heavy traffic — reroute or delay start.")
    if fuel_price>4.5: suggestions.append("⛽ High fuel prices — optimize load or combine deliveries.")

    shipment_row = df_filter[df_filter['shipment_id']==selected_shipment].drop(columns=['total_cost_usd','lane_id_corrected','avg_speed_mph'])
    shipment_row_encoded = pd.get_dummies(shipment_row, columns=object_cols, drop_first=True)
    shipment_row_encoded = shipment_row_encoded.reindex(columns=columns_used, fill_value=0)

    pred_cost = rf_cost.predict(shipment_row_encoded)[0]
    pred_lane = knn_lane.predict(shipment_row_encoded)[0]
    pred_speed = rf_speed.predict(shipment_row_encoded)[0]

    return {
        "lane_type": selected_lane,
        "total_cost": round(total_cost,2),
        "fuel_price": round(fuel_price,2),
        "weather": weather,
        "traffic_index": traffic,
        "distance": distance,
        "adjusted_speed": round(adjusted_speed,2),
        "required_speed": round(required_speed,2),
        "suggestions": suggestions,
        "model_pred_cost": round(pred_cost,2),
        "model_pred_lane": pred_lane,
        "model_pred_speed": round(pred_speed,2)
    }

# -------------------- DISPLAY --------------------
st.markdown("<div class='section-header' style='background: linear-gradient(90deg, #29B6F6, #BBDEFB);'>🌐 Lane Recommendation & Predictions</div>", unsafe_allow_html=True)

if st.button("🚀 Get Both Recommendations"):
    result_api = recommend_best_lane(selected_shipment, axle_count, use_api=True)
    result_offline = recommend_best_lane(selected_shipment, axle_count, use_api=False)

    cols = st.columns(2)
    for col, result, title in zip(cols, [result_api, result_offline], ["🌐 With API","⚡ Offline"]):
        if result:
            col.markdown(f"### {title}")
            col.markdown(f"<div class='metric-card metric-blue'><b>Lane</b><br>{result['lane_type'].capitalize()}</div>", unsafe_allow_html=True)
            col.markdown(f"<div class='metric-card metric-green'><b>Total Cost</b><br>${result['total_cost']}</div>", unsafe_allow_html=True)
            col.markdown(f"<div class='metric-card metric-orange'><b>Fuel Price</b><br>${result['fuel_price']}</div>", unsafe_allow_html=True)
            col.markdown(f"<div class='metric-card metric-purple'><b>Adjusted Speed</b><br>{result['adjusted_speed']} mph</div>", unsafe_allow_html=True)
            col.markdown(f"<div class='metric-card metric-pink'><b>Required Speed</b><br>{result['required_speed']} mph</div>", unsafe_allow_html=True)
            col.markdown(f"<div class='metric-card metric-yellow'><b>Weather</b><br>{result['weather']}</div>", unsafe_allow_html=True)
            col.markdown(f"<div class='metric-card metric-yellow'><b>Traffic Index</b><br>{result['traffic_index']}</div>", unsafe_allow_html=True)
            # col.markdown(f"<div class='metric-card metric-green'><b>Predicted Cost</b><br>${result['model_pred_cost']}</div>", unsafe_allow_html=True)
            # col.markdown(f"<div class='metric-card metric-blue'><b>Predicted Lane</b><br>{result['model_pred_lane']}</div>", unsafe_allow_html=True)
            # col.markdown(f"<div class='metric-card metric-orange'><b>Predicted Speed</b><br>{result['model_pred_speed']} mph</div>", unsafe_allow_html=True)
            if result['suggestions']:
                col.markdown("**Suggestions:**")
                for s in result['suggestions']:
                    col.markdown(f"- {s}")
        else:
            col.error("❌ Could not generate recommendation.")

# -------------------- FOOTER --------------------
st.markdown("<footer>© 2025 innova solutions | Lane Recommendation & Prediction Dashboard</footer>", unsafe_allow_html=True)
