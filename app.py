from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from config import Config
from predict import WeatherPredictor
from data_collector import WeatherDataCollector
import pandas as pd
import os

# ============================================================
# APP SETUP
# ============================================================
app = Flask(__name__)
CORS(app)

# Initialize configuration
Config.init_app()

# Initialize ML & data components
predictor = WeatherPredictor()
collector = WeatherDataCollector()

# ============================================================
# LOAD SAMPLE DATA (SAFE)
# ============================================================
sample_data = None
has_data = False

try:
    if os.path.exists(Config.DATASET_FILE):
        sample_data = pd.read_csv(Config.DATASET_FILE)
        has_data = len(sample_data) > 0
        print(f"✅ Loaded {len(sample_data)} sample records")
    else:
        print("⚠️ Dataset file not found")
except Exception as e:
    print(f"⚠️ Failed to load sample data: {e}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_aqi_category(aqi):
    categories = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }
    return categories.get(aqi, "Unknown")


def get_sample_city_data(city):
    """Fallback data if live API fails"""
    if not has_data:
        return None

    city_data = sample_data[sample_data["city"] == city]
    if city_data.empty:
        city_data = sample_data

    latest = city_data.iloc[-1]

    return {
        "city": city,
        "timestamp": str(latest["timestamp"]),
        "current": {
            "temperature": float(latest["temperature"]),
            "feels_like": float(latest["feels_like"]),
            "humidity": int(latest["humidity"]),
            "pressure": int(latest["pressure"]),
            "wind_speed": float(latest["wind_speed"]),
            "clouds": int(latest["clouds"]),
            "weather": str(latest["weather_main"]),
            "description": str(latest["weather_description"]),
            "aqi": int(latest["aqi"]),
            "aqi_category": get_aqi_category(int(latest["aqi"])),
            "pm2_5": float(latest["pm2_5"]),
            "pm10": float(latest["pm10"])
        }
    }

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    try:
        from dashboard import create_dashboard
        create_dashboard()
        return render_template("dashboard.html")
    except Exception as e:
        return f"Dashboard error: {e}", 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        city = data.get("city", Config.DEFAULT_CITY)

        # Try real-time prediction
        try:
            predictions = predictor.predict_for_city(city)
            if predictions and "error" not in predictions:
                return jsonify({"success": True, "data": predictions})
        except:
            pass

        # Fallback to sample data
        fallback = get_sample_city_data(city)
        if fallback:
            fallback["health_advice"] = predictor.get_health_advice(
                fallback["current"]["aqi"],
                fallback["current"]["pm2_5"]
            )
            return jsonify({"success": True, "data": fallback})

        return jsonify({
            "success": False,
            "error": "No data available. Run generate_sample_data.py."
        }), 404

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/weather/<city>")
def get_weather(city):
    try:
        data = collector.fetch_weather_data(city)
        if not data:
            return jsonify({"success": False, "error": "City not found"}), 404
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/cities")
def get_cities():
    return jsonify({"success": True, "cities": Config.CITIES})


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json()
        message = payload.get("message", "").lower()
        city = payload.get("city", Config.DEFAULT_CITY)

        try:
            predictions = predictor.predict_for_city(city)
            if "error" in predictions:
                raise Exception
        except:
            predictions = get_sample_city_data(city)
            if not predictions:
                return jsonify({
                    "success": False,
                    "reply": "No data available. Run generate_sample_data.py."
                })

            predictions["health_advice"] = predictor.get_health_advice(
                predictions["current"]["aqi"],
                predictions["current"]["pm2_5"]
            )

        current = predictions["current"]

        if any(k in message for k in ["weather", "temperature", "forecast"]):
            reply = (
                f"🌍 Weather in {city}\n\n"
                f"🌡️ Temp: {current['temperature']}°C (Feels {current['feels_like']}°C)\n"
                f"☁️ {current['weather']} - {current['description']}\n"
                f"💧 Humidity: {current['humidity']}%\n"
                f"💨 Wind: {current['wind_speed']} m/s\n\n"
                f"🏭 AQI: {current['aqi_category']} ({current['aqi']})\n\n"
                f"💡 {predictions.get('health_advice', '')}"
            )
            return jsonify({"success": True, "reply": reply, "data": predictions})

        if any(k in message for k in ["aqi", "air", "pollution"]):
            reply = (
                f"💨 Air Quality in {city}\n\n"
                f"AQI: {current['aqi_category']} ({current['aqi']})\n"
                f"PM2.5: {current['pm2_5']} µg/m³\n"
                f"PM10: {current['pm10']} µg/m³\n\n"
                f"💡 {predictions.get('health_advice', '')}"
            )
            return jsonify({"success": True, "reply": reply, "data": predictions})

        return jsonify({
            "success": True,
            "reply": "Ask me about 🌦️ weather, 🌡️ temperature, or 💨 AQI for any city!"
        })

    except Exception as e:
        return jsonify({"success": False, "reply": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "data_available": has_data
    })

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 WEATHER ML FLASK SERVER STARTED")
    print("=" * 60)
    print("📍 App: http://127.0.0.1:5000")
    print("📊 Dashboard: http://127.0.0.1:5000/dashboard")
    print(f"📁 Data: {'Available' if has_data else 'Not available'}")
    print("=" * 60 + "\n")

    # 🔒 CRITICAL FIX: disable auto reloader
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
