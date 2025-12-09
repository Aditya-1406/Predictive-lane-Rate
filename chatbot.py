# streamlit_lane_chatbot_full.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from functools import lru_cache
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="🚛 AI Lane Chatbot (Full)", layout="wide")

st.title("🚛 AI-Powered Lane Recommendation Assistant (Chatbot)")
st.caption("Uses TomTom + Open-Meteo + EIA APIs (enter keys below). No LLMs used — just logic & models.")

# ---------------------------
# Sidebar: config + keys
# ---------------------------
with st.sidebar:
    st.header("Configuration & API keys")
    TOMTOM_API_KEY = st.text_input("TomTom API Key", value="", placeholder="Paste TomTom API key here")
    EIA_API_KEY = st.text_input("EIA API Key", value="", placeholder="Paste EIA API key here")
    st.markdown("---")
    st.header("CSV filenames (in working dir)")
    lane_csv = st.text_input("Lane CSV (detailed)", value="lane_cost_cleaned_final.csv")
    cities_csv = st.text_input("City lanes CSV", value="standard_us_city_lanes_with_speed1(in).csv")
    st.markdown("---")
    st.subheader("ML")
    retrain_models = st.button("Retrain ML models now")
    show_model_metrics = st.checkbox("Show model metrics in sidebar", value=True)
    st.markdown("---")
    st.write("Hints:")
    st.write("- Chat: `from Dallas to Denver in 2 days`")
    st.write("- Or: `recommend lane for shipment SH_471`")

# ---------------------------
# Load CSVs (cached)
# ---------------------------
@st.cache_data
def load_csvs(lane_file, cities_file):
    df_filter = pd.read_csv(lane_file)
    cities_df = pd.read_csv(cities_file)
    # normalize header whitespace
    df_filter.columns = df_filter.columns.str.strip()
    cities_df.columns = cities_df.columns.str.strip()
    # ensure shipment_id in cities_df if missing
    if 'shipment_id' not in cities_df.columns:
        cities_df.insert(0, 'shipment_id', [f"SH_{i:03}" for i in range(1, len(cities_df) + 1)])
    return df_filter, cities_df

try:
    df_filter, cities_df = load_csvs(lane_csv, cities_csv)
except FileNotFoundError as e:
    st.error(f"CSV load error: {e}. Make sure files are in working dir and names match.")
    st.stop()

# ---------------------------
# Normalize city columns (flexible)
# ---------------------------
# accomodate both formats: combined "Chicago, IL" or separate origin_state/dest_state columns
def normalize_cities_df(df_cities):
    df = df_cities.copy()
    # If separate state columns exist, create combined "City, ST"
    if 'origin_state' in df.columns and 'origin_city' in df.columns:
        df['origin_city_combined'] = df['origin_city'].astype(str).str.strip() + ", " + df['origin_state'].astype(str).str.strip()
    else:
        df['origin_city_combined'] = df['origin_city'].astype(str).str.strip()

    if 'dest_state' in df.columns and 'dest_city' in df.columns:
        df['destination_city_combined'] = df['dest_city'].astype(str).str.strip() + ", " + df['dest_state'].astype(str).str.strip()
    else:
        # some files use 'destination_city' or 'dest_city' naming — support both
        if 'destination_city' in df.columns:
            df['destination_city_combined'] = df['destination_city'].astype(str).str.strip()
        elif 'dest_city' in df.columns:
            df['destination_city_combined'] = df['dest_city'].astype(str).str.strip()
        else:
            df['destination_city_combined'] = df['destination_city'].astype(str).str.strip() if 'destination_city' in df.columns else ""

    # also keep original or fallback columns named differently
    # unify column names that later code expects:
    # origin_city_combined, destination_city_combined, distance_miles_est, lane_type, speed_limit_mph, total_cost_usd
    return df

cities_df = normalize_cities_df(cities_df)

# also ensure df_filter numeric columns exist (fillna later)
df_filter = df_filter.copy()
df_filter.columns = df_filter.columns.str.strip()

# ---------------------------
# Constants & helper maps
# ---------------------------
weight_of_truck_tons = 12.8
fuel_efficiency_mpg = 6.19
hours_per_day = 8
lane_type_str_map = {1: 'slow', 2: 'normal', 3: 'fast'}
weather_speed_reduction = {'Clear': 0, 'Fog': 2, 'Rain': 3, 'Snow': 4, 'Storm': 5}
axle_cost_addition = {3: 0, 4: 5.73, 5: 13.71, 6: 19.42}

# ---------------------------
# External API helpers
# ---------------------------
@lru_cache(maxsize=256)
def get_coordinates(city_name, tomtom_api_key):
    if not city_name:
        return None, None
    city_name = str(city_name).strip()
    url = f"https://api.tomtom.com/search/2/geocode/{city_name}.json"
    params = {"view": "Unified", "limit": 1, "key": tomtom_api_key}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        if "results" in data and len(data["results"]) > 0:
            pos = data["results"][0]["position"]
            return pos.get("lat"), pos.get("lon")
    except Exception:
        pass
    return None, None

def get_weather_condition(city_name, tomtom_api_key):
    lat, lon = get_coordinates(city_name, tomtom_api_key)
    if lat is None or lon is None:
        return "Unknown"
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "current_weather": True}, timeout=8)
        data = r.json()
        code = data.get("current_weather", {}).get("weathercode")
        if code in [71, 73, 75, 77]:
            return "Snow"
        if code in [95, 96, 99]:
            return "Storm"
        if code in [61, 63, 65, 80, 81, 82]:
            return "Rain"
        if code in [45, 48]:
            return "Fog"
        return "Clear"
    except Exception:
        return "Unknown"

def get_traffic_index(city_name, tomtom_api_key):
    lat, lon = get_coordinates(city_name, tomtom_api_key)
    if lat is None or lon is None:
        return 5
    try:
        r = requests.get("https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json",
                         params={"point": f"{lat},{lon}", "unit": "MPH", "key": tomtom_api_key}, timeout=8)
        data = r.json().get("flowSegmentData", {})
        current_speed = data.get("currentSpeed")
        free_speed = data.get("freeFlowSpeed")
        if current_speed and free_speed:
            ratio = free_speed / max(current_speed, 1)
            return min(10, max(1, round(ratio * 5)))
    except Exception:
        pass
    return 5

def get_latest_fuel_price(eia_api_key):
    if not eia_api_key:
        return 4.04
    url = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
    params = {
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 1,
        "api_key": eia_api_key
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        return float(data["response"]["data"][0]["value"])
    except Exception:
        return 4.04

def traffic_speed_penalty(traffic_index):
    if traffic_index <= 3:
        return 0
    elif 4 <= traffic_index <= 6:
        return 2
    elif 7 <= traffic_index <= 8:
        return 4
    else:
        return 6

# ---------------------------
# Flexible city matching
# ---------------------------
def find_city_match_input(user_city, cities_df):
    """Return best match string from cities_df (origin_city_combined or destination_city_combined).
       Accepts user inputs like 'Dallas' and matches 'Dallas, TX' or matches exact city if provided.
    """
    if not user_city:
        return None
    user_city = str(user_city).strip().lower()
    # Search in combined fields
    for col in ['origin_city_combined', 'destination_city_combined']:
        if col in cities_df.columns:
            series = cities_df[col].astype(str)
            # exact contains match
            mask = series.str.lower().str.contains(user_city, na=False)
            if mask.any():
                # return the first matched full name (preserve original casing)
                return series[mask].iloc[0]
    # fallback: maybe user typed "Chicago, IL" exactly and our df has 'origin_city' field
    for col in ['origin_city', 'dest_city', 'destination_city']:
        if col in cities_df.columns:
            s = cities_df[col].astype(str)
            mask = s.str.lower().str.contains(user_city, na=False)
            if mask.any():
                return s[mask].iloc[0]
    return None

# ---------------------------
# Recommendation logic (uses cities_df)
# ---------------------------
def recommend_best_lane_user(origin_input, destination_input, target_days, tomtom_api_key, eia_api_key, axle_count=4):
    # find best matching strings in cities_df
    origin_match = find_city_match_input(origin_input, cities_df)
    dest_match = find_city_match_input(destination_input, cities_df)

    if (origin_match is None) or (dest_match is None):
        return {"error": True, "message": "⚠️ Could not match the origin/destination to available city entries. Try adding state (e.g., 'Dallas, TX') or choose a different city."}

    # Filter the cities_df rows which match the matched strings either in origin_city_combined/destination_city_combined or origin_city/dest_city
    df = cities_df.copy()
    origin_col = 'origin_city_combined' if 'origin_city_combined' in df.columns else ('origin_city' if 'origin_city' in df.columns else df.columns[0])
    dest_col = 'destination_city_combined' if 'destination_city_combined' in df.columns else ('destination_city' if 'destination_city' in df.columns else df.columns[1])

    # now find rows
    df_route = df[(df[origin_col].astype(str).str.strip().str.lower() == str(origin_match).strip().lower()) &
                  (df[dest_col].astype(str).str.strip().str.lower() == str(dest_match).strip().lower())]

    if df_route.empty:
        # If exact combined match not found, try partial match by city name only (without state)
        origin_city_only = str(origin_match).split(",")[0].strip().lower()
        dest_city_only = str(dest_match).split(",")[0].strip().lower()
        df_route = df[(df[origin_col].astype(str).str.lower().str.contains(origin_city_only, na=False)) &
                      (df[dest_col].astype(str).str.lower().str.contains(dest_city_only, na=False))]

    if df_route.empty:
        return {"error": True, "message": f"⚠️ No route data for {origin_input} → {destination_input} in the cities dataset."}

    # Use the first matching row to get distance and base lane info
    row = df_route.iloc[0]
    # distance column name may be 'distance_miles_est' per your listing
    distance_col = 'distance_miles_est' if 'distance_miles_est' in df_route.columns else ('ref_distance_miles' if 'ref_distance_miles' in df_filter.columns else None)
    if distance_col and distance_col in row:
        distance = float(row[distance_col])
    else:
        # fallback to 0 (but recommendation will be conservative)
        distance = float(row.get('distance_miles_est', row.get('ref_distance_miles', 0)))

    required_speed = distance / (target_days * hours_per_day)

    # call APIs (if keys provided)
    weather = get_weather_condition(origin_match, tomtom_api_key) if tomtom_api_key else "Unknown"
    traffic_index = get_traffic_index(origin_match, tomtom_api_key) if tomtom_api_key else 5
    fuel_price = get_latest_fuel_price(eia_api_key)

    weather_penalty = weather_speed_reduction.get(weather, 0)
    traffic_penalty = traffic_speed_penalty(traffic_index)
    adjusted_speed = max(30, required_speed - (weather_penalty + traffic_penalty))

    if adjusted_speed > 80:
        return {"error": True, "message": "⚠️ Impossible to deliver under current conditions (required speed too high)."}

    # derive lane type using adjusted speed rules used earlier
    lane_type_key = 1 if adjusted_speed < 60 else (2 if adjusted_speed < 70 else 3)
    selected_lane_str = lane_type_str_map[lane_type_key]

    # try to find a matching lane row in df_route with that lane_type (cities_df might have lane_type column)
    lane_row = None
    if 'lane_type' in df_route.columns:
        matches = df_route[df_route['lane_type'].astype(str).str.lower() == selected_lane_str]
        if not matches.empty:
            lane_row = matches.iloc[0]
    # fallback: use first df_route row for cost/speed limits
    lane_row = lane_row if lane_row is not None else df_route.iloc[0]

    base_cost = float(lane_row.get('total_cost_usd', lane_row.get('ref_total_lane_cost_usd', 0)))
    lane_speed_limit = float(lane_row.get('speed_limit_mph', lane_row.get('ref_speed_limit_mph', np.nan)))

    fuel_cost = (distance / fuel_efficiency_mpg) * fuel_price
    total_cost = base_cost + fuel_cost + axle_cost_addition.get(axle_count, 0)

    suggestions = []
    if weather in ['Rain', 'Snow', 'Storm']:
        suggestions.append("⚠️ Poor weather — prefer slower lanes for safety.")
    if traffic_index >= 8:
        suggestions.append("🚦 Heavy traffic — consider off-peak travel or alternate route.")
    if fuel_price > 4.5:
        suggestions.append("⛽ High fuel prices — try load consolidation.")

    result = {
        "error": False,
        "origin_matched": origin_match,
        "destination_matched": dest_match,
        "distance_miles": round(distance, 2),
        "required_speed": round(required_speed, 2),
        "adjusted_speed": round(adjusted_speed, 2),
        "selected_lane": selected_lane_str,
        "lane_speed_limit": lane_speed_limit,
        "base_cost": round(base_cost, 2),
        "fuel_price": round(fuel_price, 2),
        "fuel_cost": round(fuel_cost, 2),
        "total_cost_estimate": round(total_cost, 2),
        "traffic_index": traffic_index,
        "weather": weather,
        "suggestions": suggestions
    }
    return result

def recommend_best_lane_by_shipment(shipment_id, tomtom_api_key, eia_api_key):
    # look inside df_filter (lane_cost_cleaned_final) for this shipment row
    df = df_filter.copy()
    if 'shipment_id' not in df.columns:
        return {"error": True, "message": "❌ shipment_id column missing from lane dataset."}
    shipment_row = df[df['shipment_id'].astype(str).str.upper() == shipment_id.upper()]
    if shipment_row.empty:
        return {"error": True, "message": f"❌ Shipment {shipment_id} not found in lane dataset."}
    row = shipment_row.iloc[0]
    origin = row.get('origin_city')
    destination = row.get('destination_city')
    target_days = row.get('delivery_target_days', 1.5)
    axle_count = None
    # try to infer axle from columns axle_3,.. axle_6_plus (if present choose 4 as default)
    if 'axle_4' in df.columns:
        axle_count = 4
    return recommend_best_lane_user(origin, destination, float(target_days), tomtom_api_key, eia_api_key, axle_count=axle_count or 4)

# ---------------------------
# ML training (cost & speed)
# ---------------------------
@st.cache_data
def train_models_from_lane_df(df_lane):
    df = df_lane.copy()
    # Use the columns you provided as features where appropriate
    # Targets:
    cost_col = 'total_cost_usd'
    speed_col = 'avg_speed_mph'
    # drop rows missing targets
    df = df.dropna(subset=[cost_col, speed_col])

    # Select numeric features known to exist in your schema
    numeric_features = []
    for c in ['ref_distance_miles', 'ref_speed_limit_mph', 'fuel_price_usd_per_gallon', 'weight_of_truck_tons',
              'road_condition_score', 'market_demand_index', 'traffic_status_encoded', 'driver_hours_per_day',
              'delivery_target_days', 'axle_3', 'axle_4', 'axle_5', 'axle_6_plus']:
        if c in df.columns:
            numeric_features.append(c)
    # If distance column in df_filter uses a different name, include it
    if 'ref_distance_miles' not in numeric_features and 'ref_distance_miles' in df.columns:
        numeric_features.append('ref_distance_miles')

    # Fallback: if no numeric features found, use available numeric columns
    if len(numeric_features) == 0:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        # remove target columns
        numeric_features = [c for c in numeric_features if c not in [cost_col, speed_col]]

    X = df[numeric_features].fillna(0).astype(float)
    y_cost = df[cost_col].astype(float)
    y_speed = df[speed_col].astype(float)

    # Scale then train simple RF regressors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_cost_train, y_cost_test = train_test_split(X_scaled, y_cost, test_size=0.2, random_state=42)
    _, _, y_speed_train, y_speed_test = train_test_split(X_scaled, y_speed, test_size=0.2, random_state=42)

    rf_cost = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_cost.fit(X_train, y_cost_train)
    cost_pred = rf_cost.predict(X_test)
    cost_r2 = r2_score(y_cost_test, cost_pred)
    cost_mae = mean_absolute_error(y_cost_test, cost_pred)

    rf_speed = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_speed.fit(X_train, y_speed_train)
    speed_pred = rf_speed.predict(X_test)
    speed_r2 = r2_score(y_speed_test, speed_pred)
    speed_mae = mean_absolute_error(y_speed_test, speed_pred)

    return {
        "rf_cost": rf_cost,
        "rf_speed": rf_speed,
        "scaler": scaler,
        "feature_cols": numeric_features,
        "metrics": {
            "cost_r2": cost_r2,
            "cost_mae": cost_mae,
            "speed_r2": speed_r2,
            "speed_mae": speed_mae
        }
    }

# train models now (or retrain if button pressed)
if retrain_models:
    models_artifacts = train_models_from_lane_df(df_filter)
else:
    # cache_data ensures this is fast on reruns
    models_artifacts = train_models_from_lane_df(df_filter)

rf_cost = models_artifacts["rf_cost"]
rf_speed = models_artifacts["rf_speed"]
scaler = models_artifacts["scaler"]
feature_cols = models_artifacts["feature_cols"]
metrics = models_artifacts["metrics"]

if show_model_metrics:
    with st.sidebar:
        st.subheader("ML Metrics (trained on lane_cost_cleaned_final)")
        st.write(f"Cost R²: {metrics['cost_r2']:.3f}  |  MAE: {metrics['cost_mae']:.2f}")
        st.write(f"Speed R²: {metrics['speed_r2']:.3f}  |  MAE: {metrics['speed_mae']:.2f}")

# ---------------------------
# Chat UI (full width)
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": (
            "Hello 👋! I'm your Transport Optimization Assistant.\n\n"
            "Try:\n• `from Dallas to Denver in 2 days`\n• `recommend lane for shipment SH_471`\n\n(Enter TomTom/EIA keys in the sidebar if you want live weather/traffic/fuel.)"
        )}
    ]

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your query (e.g., from X to Y in 2 days, or recommend lane for SH_123)...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # parse shipment id
    shipment_match = re.search(r"(sh[_-]?\d+)", user_input.lower())
    # parse routes: allow "from X to Y in N days" and also "X to Y in N days"
    route_match = re.search(r"from\s+([a-zA-Z\s,]+?)\s+to\s+([a-zA-Z\s,]+?)\s+in\s+(\d+(?:\.\d+)?)\s*days?", user_input.lower())
    if not route_match:
        route_match = re.search(r"([a-zA-Z\s,]+?)\s+to\s+([a-zA-Z\s,]+?)\s+in\s+(\d+(?:\.\d+)?)\s*days?", user_input.lower())

    response_text = "❓ I didn't understand. Try: `from CityA to CityB in N days` or `recommend lane for shipment SH_123`."

    if shipment_match:
        shipment_id = shipment_match.group(1).upper().replace('-', '_')
        with st.spinner("Fetching shipment and recommending..."):
            res = recommend_best_lane_by_shipment(shipment_id, TOMTOM_API_KEY, EIA_API_KEY)
        if res.get("error"):
            response_text = f"⚠️ {res.get('message')}"
        else:
            # Add ML predictions based on df_filter model features
            # Build feature vector from df_filter's first matching shipment row
            try:
                shipment_row = df_filter[df_filter['shipment_id'].astype(str).str.upper() == shipment_id.upper()].iloc[0]
                feat = []
                for c in feature_cols:
                    feat.append(float(shipment_row.get(c, 0)))
                feat_arr = np.array(feat).reshape(1, -1)
                feat_scaled = scaler.transform(feat_arr)
                ml_cost = rf_cost.predict(feat_scaled)[0]
                ml_speed = rf_speed.predict(feat_scaled)[0]
            except Exception:
                ml_cost = None
                ml_speed = None

            response_text = (
                f"🚚 **Shipment {shipment_id} Recommendation**\n\n"
                f"Route: **{res['origin_matched']} → {res['destination_matched']}**\n"
                f"Distance: **{res['distance_miles']} miles**\n"
                f"Required Speed: **{res['required_speed']} mph**\n"
                f"Adjusted Speed: **{res['adjusted_speed']} mph**\n"
                f"Recommended Lane: **{res['selected_lane'].capitalize()}**\n"
                f"Lane Speed Limit: **{res['lane_speed_limit']} mph**\n"
                f"Fuel Price (used): **${res['fuel_price']}/gal**\n"
                f"Rule-based Total Cost Estimate: **${res['total_cost_estimate']}**\n"
            )
            # if ml_cost is not None:
            #     response_text += f"\n\n**ML Predicted Total Cost:** ${ml_cost:.2f}"
            # if ml_speed is not None:
            #     response_text += f"\n**ML Predicted Avg Speed:** {ml_speed:.2f} mph"

            if res.get('suggestions'):
                response_text += "\n\n**Suggestions:**\n" + "\n".join([f"- {s}" for s in res['suggestions']])
            else:
                response_text += "\n\n✅ No additional safety suggestions."

    elif route_match:
        origin_raw, dest_raw, days_str = route_match.groups()
        origin_raw = origin_raw.strip()
        dest_raw = dest_raw.strip()
        days = float(days_str)
        with st.spinner(f"Calculating recommendation for {origin_raw} → {dest_raw}..."):
            res = recommend_best_lane_user(origin_raw, dest_raw, days, TOMTOM_API_KEY, EIA_API_KEY)
        if res.get("error"):
            response_text = f"⚠️ {res.get('message')}"
        else:
            # Build a lightweight feature vector from route row if possible (use distance & some defaults)
            try:
                # try to find matching cities_df row used earlier
                origin_match = res['origin_matched']
                dest_match = res['destination_matched']
                df_row = cities_df[
                    (cities_df.get('origin_city_combined', cities_df.get('origin_city')).astype(str).str.lower() == str(origin_match).strip().lower()) &
                    (cities_df.get('destination_city_combined', cities_df.get('destination_city')).astype(str).str.lower() == str(dest_match).strip().lower())
                ]
                if df_row.empty:
                    df_row = cities_df.iloc[[0]]
                # Build features using available feature_cols: pull matching names from df_filter if possible else use defaults
                feat = []
                for c in feature_cols:
                    # prefer df_filter columns aggregated by matching route (try drop-in)
                    # default to 0 if not available
                    val = 0.0
                    # look in df_row first for distance or speed_limit etc
                    if c in df_row.columns:
                        val = float(df_row.iloc[0].get(c, 0) or 0)
                    # fallback: try df_filter averages for that column
                    elif c in df_filter.columns:
                        val = float(df_filter[c].median(skipna=True) if not df_filter[c].isna().all() else 0)
                    feat.append(val)
                feat_arr = np.array(feat).reshape(1, -1)
                feat_scaled = scaler.transform(feat_arr)
                ml_cost = rf_cost.predict(feat_scaled)[0]
                ml_speed = rf_speed.predict(feat_scaled)[0]
            except Exception:
                ml_cost = None
                ml_speed = None

            response_text = (
                f"🚚 **Route Recommendation**\n\n"
                f"Route: **{res['origin_matched']} → {res['destination_matched']}**\n"
                f"Distance: **{res['distance_miles']} miles**\n"
                f"Required Speed: **{res['required_speed']} mph**\n"
                f"Adjusted Speed: **{res['adjusted_speed']} mph**\n"
                f"Recommended Lane: **{res['selected_lane'].capitalize()}**\n"
                f"Lane Speed Limit: **{res['lane_speed_limit']} mph**\n"
                f"Fuel Price (used): **${res['fuel_price']}/gal**\n"
                f"Rule-based Total Cost Estimate: **${res['total_cost_estimate']}**\n"
            )
            # if ml_cost is not None:
            #     response_text += f"\n\n**ML Predicted Total Cost:** ${ml_cost:.2f}"
            # if ml_speed is not None:
            #     response_text += f"\n**ML Predicted Avg Speed:** {ml_speed:.2f} mph"

            if res.get('suggestions'):
                response_text += "\n\n**Suggestions:**\n" + "\n".join([f"- {s}" for s in res['suggestions']])
            else:
                response_text += "\n\n✅ No additional safety suggestions."

    else:
        # fallback small intents
        if "help" in user_input.lower():
            response_text = "Try: `from CityA to CityB in N days` or `recommend lane for shipment SH_123`."
        else:
            response_text = "❓ I didn't understand. Try: `from CityA to CityB in N days` or `recommend lane for shipment SH_123`."

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)

# Footer
st.markdown("---")
st.write("Built for demo: uses your datasets and in-code APIs (TomTom/Open-Meteo/EIA).")

