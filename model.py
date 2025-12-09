import pandas as pd 
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier ,GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor , DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

df_filter = pd.read_csv("lane_cost_cleaned_final.csv")

object_cols = df_filter.select_dtypes(include='object').columns
print(object_cols)

df_encoded = pd.get_dummies(df_filter,columns=object_cols,drop_first=True)

X = df_encoded.drop(columns=['total_cost_usd','lane_id_corrected','avg_speed_mph'])
y_cost = df_encoded['total_cost_usd']
y_lane = df_encoded['lane_id_corrected'] 
y_speed = df_encoded['avg_speed_mph']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# After creating X and before training
columns_used = X.columns



# Save it using joblib (just once)
import joblib
joblib.dump(columns_used, 'columns_used.pkl')

X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(
    X, y_cost, test_size=0.2, random_state=42
)

# Classification split
X_train_lane, X_test_lane, y_train_lane, y_test_lane = train_test_split(
    X, y_lane, test_size=0.2, random_state=42
)

#for speed 
X_train_speed, X_test_speed, y_train_speed, y_test_speed = train_test_split(
    X, y_speed, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_cost, y_train_cost)
y_pred_reg = rf.predict(X_test_cost)

r2 = r2_score(y_test_cost, y_pred_reg)
mae = mean_absolute_error(y_test_cost, y_pred_reg)

print("\n🌲 Random Forest Regressor Results:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_lane, y_train_lane)
y_pred_class = knn.predict(X_test_lane)

acc = accuracy_score(y_test_lane, y_pred_class)
f1 = f1_score(y_test_lane, y_pred_class, average='weighted')

print("\n🤖 KNN Classifier Results:")
print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")

rf_speed = RandomForestRegressor(n_estimators=200, random_state=42)
rf_speed.fit(X_train_speed, y_train_speed)

y_pred_speed = rf_speed.predict(X_test_speed)
from sklearn.metrics import r2_score, mean_absolute_error
print("Avg Speed R2:", r2_score(y_test_speed, y_pred_speed))
print("Avg Speed MAE:", mean_absolute_error(y_test_speed, y_pred_speed))



# -------------------------------
# 🧩 Load cities dataset
# -------------------------------
cities_df = pd.read_csv("standard_us_city_lanes_with_speed1(in).csv")
cities_df.columns = cities_df.columns.str.strip()

# ✅ Ensure shipment_id column exists
if 'shipment_id' not in cities_df.columns:
    cities_df.insert(0, 'shipment_id', [f"SH_{i:03}" for i in range(1, len(cities_df) + 1)])

# -------------------------------
# ⚙️ Constants
# -------------------------------
weight_of_truck_tons = 12.8
fuel_efficiency_mpg = 6.19
hours_per_day = 8

# -------------------------------
# 🧩 Mappings
# -------------------------------
lane_type_map = {'slow': 1, 'normal': 2, 'fast': 3}
lane_type_str_map = {1: 'slow', 2: 'normal', 3: 'fast'}
weather_speed_reduction = {'Clear': 0, 'Fog': 2, 'Rain': 3, 'Snow': 4, 'Storm': 5}
axle_cost_addition = {3: 0, 4: 5.73, 5: 13.71, 6: 19.42}

coordinate_cache = {}

# -------------------------------
# 🚗 Traffic penalty
# -------------------------------
def traffic_speed_penalty(traffic_index):
    if traffic_index <= 3:
        return 0
    elif 4 <= traffic_index <= 6:
        return 2
    elif 7 <= traffic_index <= 8:
        return 4
    else:
        return 6

# -------------------------------
# 🌐 TomTom Geocoding
# -------------------------------
def get_coordinates(city_name, tomtom_api_key):
    if city_name in coordinate_cache:
        return coordinate_cache[city_name]
    url = f"https://api.tomtom.com/search/2/geocode/{city_name}.json"
    params = {"storeResult": "false", "view": "Unified", "limit": 1, "key": tomtom_api_key}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            pos = data["results"][0]["position"]
            coordinate_cache[city_name] = (pos["lat"], pos["lon"])
            return pos["lat"], pos["lon"]
    except:
        pass
    print(f"⚠️ Could not fetch coordinates for {city_name}")
    return None, None

# -------------------------------
# 🌤️ Open-Meteo Weather
# -------------------------------
def get_weather_condition(city_name, tomtom_api_key):
    lat, lon = get_coordinates(city_name, tomtom_api_key)
    if lat is None or lon is None:
        return "Unknown"
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        code = data["current_weather"]["weathercode"]
        if code in [71, 73, 75, 77]:
            return "Snow"
        elif code in [95, 96, 99]:
            return "Storm"
        elif code in [61, 63, 65, 80, 81, 82]:
            return "Rain"
        elif code in [45, 48]:
            return "Fog"
        else:
            return "Clear"
    except:
        return "Unknown"

# -------------------------------
# 🚦 TomTom Traffic Flow
# -------------------------------
def get_traffic_index(city_name, tomtom_api_key):
    lat, lon = get_coordinates(city_name, tomtom_api_key)
    if lat is None or lon is None:
        return 5
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
    params = {"point": f"{lat},{lon}", "unit": "MPH", "key": tomtom_api_key}
    try:
        response = requests.get(url, params=params)
        data = response.json().get("flowSegmentData", {})
        current_speed = data.get("currentSpeed")
        free_speed = data.get("freeFlowSpeed")
        if current_speed and free_speed:
            ratio = free_speed / max(current_speed, 1)
            return min(10, max(1, round(ratio * 5)))
    except:
        pass
    return 5

# -------------------------------
# ⛽ Dynamic Fuel Price (EIA)
# -------------------------------
def get_latest_fuel_price(eia_api_key):
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
        response = requests.get(url, params=params)
        data = response.json()
        return float(data["response"]["data"][0]["value"])
    except:
        print("⚠️ Could not fetch live fuel data, using default $4.04")
        return 4.04

# -------------------------------
# 🧠 Real-Time Recommendation
# -------------------------------
def recommend_best_lane_by_shipment(shipment_id, tomtom_api_key, eia_api_key):
    shipment = df_filter[df_filter['shipment_id'] == shipment_id]
    if shipment.empty:
        print("❌ Invalid Shipment ID.")
        return None

    origin = shipment.iloc[0]['origin_city']
    destination = shipment.iloc[0]['destination_city']
    target_days = shipment.iloc[0].get('delivery_target_days', 1.5)  # default if not in dataset

    print(f"\n🔍 Shipment {shipment_id} details fetched from dataset:")
    print(f"  Origin: {origin}")
    print(f"  Destination: {destination}")
    print(f"  Target Days: {target_days}\n")

    return recommend_best_lane_user(origin, destination, target_days, tomtom_api_key, eia_api_key)

# -------------------------------
# 🚛 Core Function (Reused)
# -------------------------------
def recommend_best_lane_user(origin, destination, target_days, tomtom_api_key, eia_api_key, axle_count=None):
    df = cities_df[(cities_df['origin_city'] == origin) & (cities_df['dest_city'] == destination)]
    if df.empty:
        print("⚠️ No data for this route.")
        return None

    distance = df.iloc[0]['distance_miles_est']
    required_speed = distance / (target_days * hours_per_day)

    weather = get_weather_condition(origin, tomtom_api_key)
    traffic = get_traffic_index(origin, tomtom_api_key)
    fuel_price = get_latest_fuel_price(eia_api_key)

    weather_penalty = weather_speed_reduction.get(weather, 0)
    traffic_penalty = traffic_speed_penalty(traffic)
    adjusted_speed = max(30, required_speed - (weather_penalty + traffic_penalty))

    if adjusted_speed > 80:
        print("⚠️ Impossible to deliver under current conditions.")
        return None

    lane_type = 1 if adjusted_speed < 60 else (2 if adjusted_speed < 70 else 3)
    selected_lane = lane_type_str_map[lane_type]

    row = df[df['lane_type'].str.lower() == selected_lane].iloc[0]
    base_cost = row['total_cost_usd']
    lane_speed_limit = row['speed_limit_mph']

    fuel_cost = (distance / fuel_efficiency_mpg) * fuel_price
    total_cost = base_cost + fuel_cost + axle_cost_addition.get(axle_count or 4, 0)

    # Smart suggestions
    suggestions = []
    if weather in ['Rain', 'Snow', 'Storm']:
        suggestions.append("⚠️ Poor weather — prefer slower lanes for safety.")
    if traffic >= 8:
        suggestions.append("🚦 Heavy traffic — reroute or delay start to off-peak hours.")
    if fuel_price > 4.5:
        suggestions.append("⛽ High fuel prices — try optimizing load or combining deliveries.")

    print(f"🚚 Real-Time Lane Recommendation")
    print(f"-------------------------------------")
    print(f"Route: {origin} → {destination}")
    print(f"Weather: {weather}")
    print(f"Traffic Index: {traffic}/10")
    print(f"Required Speed: {round(required_speed,2)} mph")
    print(f"Adjusted Speed: {round(adjusted_speed,2)} mph")
    print(f"Selected Lane: {selected_lane.capitalize()}")
    print(f"Fuel Price: ${fuel_price}/gal")
    print(f"Fuel Cost: ${round(fuel_cost,2)}")
    print(f"Total Cost: ${round(total_cost,2)}")

    if suggestions:
        print("\n🧭 Smart Suggestions:")
        for s in suggestions:
            print("  -", s)
    else:
        print("\n✅ Conditions are optimal for travel.")

    return {
        "lane_type": selected_lane,
        "total_cost": round(total_cost, 2),
        "fuel_price": round(fuel_price, 2),
        "weather": weather,
        "traffic_index": traffic,
        "suggestions": suggestions
    }

# -------------------------------
# 🧾 Example: Fetch by Shipment ID
# -------------------------------
TOMTOM_API_KEY = "VoMgKXVjYZAnjRwKmU8C9OxkykqSWeAM"
EIA_API_KEY = "OQdfxEcYk9VyPNOoNjHkymVqMK0stggKOtRcHPCn"

best = recommend_best_lane_by_shipment("SH_471", TOMTOM_API_KEY, EIA_API_KEY)
