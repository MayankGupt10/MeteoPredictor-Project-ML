import joblib
import numpy as np
from datetime import datetime
from config import Config
from data_collector import WeatherDataCollector

class WeatherPredictor:
    """Make predictions using trained models"""
    
    def __init__(self):
        self.load_models()
        self.collector = WeatherDataCollector()
        
    def load_models(self):
        """Load trained models"""
        try:
            self.temp_model = joblib.load(Config.TEMP_MODEL_FILE)
            self.weather_clf = joblib.load(Config.WEATHER_MODEL_FILE)
            self.humidity_model = joblib.load(Config.HUMIDITY_MODEL_FILE)
            self.scaler = joblib.load(Config.SCALER_FILE)
            print("✅ Models loaded successfully!")
        except FileNotFoundError:
            print("❌ Models not found! Train models first using train_model.py")
            self.temp_model = None
            self.weather_clf = None
            self.humidity_model = None
            self.scaler = None
    
    def get_aqi_category(self, aqi):
        """Convert AQI number to category"""
        categories = {
            1: "Good",
            2: "Fair", 
            3: "Moderate",
            4: "Poor",
            5: "Very Poor"
        }
        return categories.get(aqi, "Unknown")
    
    def get_health_advice(self, aqi, pm25):
        """Provide health advice based on AQI"""
        if aqi >= 4 or (pm25 and pm25 > 55):
            return "⚠️ Air quality is poor. Avoid outdoor activities. Wear a mask if going outside."
        elif aqi == 3:
            return "⚡ Moderate air quality. Sensitive groups should limit outdoor activities."
        else:
            return "✅ Air quality is good. Safe for outdoor activities."
    
    def predict_for_city(self, city_name):
        """Make predictions for a city using real-time data"""
        print(f"\n🌍 Fetching real-time data for {city_name}...")
        
        # Get live data
        live_data = self.collector.fetch_weather_data(city_name)
        
        if not live_data:
            return {"error": f"Could not fetch data for {city_name}"}
        
        # Extract features
        now = datetime.now()
        features_temp = np.array([[
            live_data['feels_like'],
            live_data['temp_min'],
            live_data['temp_max'],
            live_data['pressure'],
            live_data['humidity'],
            live_data['wind_speed'],
            live_data['clouds'],
            live_data['pm2_5'] or 0,
            live_data['pm10'] or 0,
            now.hour,
            now.month
        ]])
        
        features_weather = np.array([[
            live_data['temperature'],
            live_data['humidity'],
            live_data['pressure'],
            live_data['wind_speed'],
            live_data['clouds'],
            live_data['pm2_5'] or 0,
            live_data['aqi'],
            now.hour,
            now.month
        ]])
        
        features_humidity = np.array([[
            live_data['temperature'],
            live_data['pressure'],
            live_data['wind_speed'],
            live_data['clouds'],
            live_data['pm2_5'] or 0,
            now.hour,
            now.month
        ]])
        
        # Make predictions
        predictions = {
            'city': city_name,
            'timestamp': str(live_data['timestamp']),
            'current': {
                'temperature': live_data['temperature'],
                'feels_like': live_data['feels_like'],
                'humidity': live_data['humidity'],
                'pressure': live_data['pressure'],
                'wind_speed': live_data['wind_speed'],
                'clouds': live_data['clouds'],
                'weather': live_data['weather_main'],
                'description': live_data['weather_description'],
                'aqi': live_data['aqi'],
                'aqi_category': self.get_aqi_category(live_data['aqi']),
                'pm2_5': live_data['pm2_5'],
                'pm10': live_data['pm10']
            }
        }
        
        # ML Predictions
        if self.temp_model and self.scaler:
            features_temp_scaled = self.scaler.transform(features_temp)
            pred_temp = self.temp_model.predict(features_temp_scaled)[0]
            predictions['ml_predictions'] = {
                'predicted_temperature': round(pred_temp, 2),
                'temp_difference': round(pred_temp - live_data['temperature'], 2)
            }
            
            if self.humidity_model:
                pred_humidity = self.humidity_model.predict(features_humidity)[0]
                predictions['ml_predictions']['predicted_humidity'] = round(pred_humidity, 2)
            
            if self.weather_clf:
                pred_weather_code = self.weather_clf.predict(features_weather)[0]
                weather_classes = ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Snow', 'Thunderstorm']
                if pred_weather_code < len(weather_classes):
                    predictions['ml_predictions']['predicted_weather'] = weather_classes[pred_weather_code]
        
        # Health advice
        predictions['health_advice'] = self.get_health_advice(
            live_data['aqi'], 
            live_data['pm2_5']
        )
        
        return predictions
    
    def display_predictions(self, predictions):
        """Display predictions in a readable format"""
        if 'error' in predictions:
            print(f"\n❌ {predictions['error']}")
            return
        
        print("\n" + "=" * 70)
        print(f"📍 WEATHER REPORT: {predictions['city']}")
        print("=" * 70)
        
        current = predictions['current']
        print(f"\n🌡️  CURRENT CONDITIONS:")
        print(f"   Temperature: {current['temperature']}°C (Feels like: {current['feels_like']}°C)")
        print(f"   Weather: {current['weather']} - {current['description']}")
        print(f"   Humidity: {current['humidity']}%")
        print(f"   Pressure: {current['pressure']} hPa")
        print(f"   Wind Speed: {current['wind_speed']} m/s")
        print(f"   Cloud Cover: {current['clouds']}%")
        
        print(f"\n💨 AIR QUALITY:")
        print(f"   AQI: {current['aqi_category']} (Level {current['aqi']})")
        print(f"   PM2.5: {current['pm2_5']} µg/m³")
        print(f"   PM10: {current['pm10']} µg/m³")
        
        if 'ml_predictions' in predictions:
            ml_pred = predictions['ml_predictions']
            print(f"\n🤖 ML PREDICTIONS:")
            print(f"   Predicted Temperature: {ml_pred['predicted_temperature']}°C")
            if 'predicted_humidity' in ml_pred:
                print(f"   Predicted Humidity: {ml_pred['predicted_humidity']}%")
            if 'predicted_weather' in ml_pred:
                print(f"   Predicted Weather: {ml_pred['predicted_weather']}")
        
        print(f"\n💡 HEALTH ADVICE:")
        print(f"   {predictions['health_advice']}")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    predictor = WeatherPredictor()
    
    # Test prediction
    city = input("\nEnter city name (or press Enter for Delhi): ").strip() or "Delhi"
    predictions = predictor.predict_for_city(city)
    predictor.display_predictions(predictions)