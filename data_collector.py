import requests
import pandas as pd
from datetime import datetime
import os
from config import Config

class WeatherDataCollector:
    """Collects real-time weather and AQI data from OpenWeather API"""
    
    def __init__(self):
        self.api_key = Config.OPENWEATHER_API_KEY
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY not found in .env file!")
        
    def get_coordinates(self, city_name):
        """Get latitude and longitude for a city"""
        try:
            url = f"{Config.GEO_API_URL}?q={city_name}&limit=1&appid={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                return data[0]['lat'], data[0]['lon']
            return None, None
        except Exception as e:
            print(f"Error getting coordinates for {city_name}: {e}")
            return None, None
    
    def fetch_weather_data(self, city_name):
        """Fetch real-time weather data"""
        try:
            lat, lon = self.get_coordinates(city_name)
            if not lat or not lon:
                return None
            
            # Weather data
            weather_url = f"{Config.WEATHER_API_URL}?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            weather_response = requests.get(weather_url, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Air pollution data
            aqi_url = f"{Config.AIR_POLLUTION_API_URL}?lat={lat}&lon={lon}&appid={self.api_key}"
            aqi_response = requests.get(aqi_url, timeout=10)
            aqi_response.raise_for_status()
            aqi_data = aqi_response.json()
            
            return self._parse_data(city_name, weather_data, aqi_data)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {city_name}: {e}")
            return None
    
    def _parse_data(self, city, weather_data, aqi_data):
        """Parse API response into structured format"""
        try:
            main = weather_data.get('main', {})
            wind = weather_data.get('wind', {})
            clouds = weather_data.get('clouds', {})
            weather = weather_data.get('weather', [{}])[0]
            
            aqi_list = aqi_data.get('list', [{}])[0]
            aqi_main = aqi_list.get('main', {})
            components = aqi_list.get('components', {})
            
            parsed = {
                'timestamp': datetime.now(),
                'city': city,
                'temperature': main.get('temp'),
                'feels_like': main.get('feels_like'),
                'temp_min': main.get('temp_min'),
                'temp_max': main.get('temp_max'),
                'pressure': main.get('pressure'),
                'humidity': main.get('humidity'),
                'wind_speed': wind.get('speed'),
                'wind_deg': wind.get('deg'),
                'clouds': clouds.get('all'),
                'weather_main': weather.get('main'),
                'weather_description': weather.get('description'),
                'aqi': aqi_main.get('aqi'),
                'pm2_5': components.get('pm2_5'),
                'pm10': components.get('pm10'),
                'co': components.get('co'),
                'no2': components.get('no2'),
                'o3': components.get('o3'),
                'so2': components.get('so2')
            }
            
            return parsed
            
        except Exception as e:
            print(f"Error parsing data: {e}")
            return None
    
    def collect_and_save(self, cities=None):
        """Collect data for multiple cities and save to CSV"""
        if cities is None:
            cities = Config.CITIES
        
        all_data = []
        
        for city in cities:
            print(f"Fetching data for {city}...")
            data = self.fetch_weather_data(city)
            if data:
                all_data.append(data)
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Append to existing file or create new
            if os.path.exists(Config.DATASET_FILE):
                existing_df = pd.read_csv(Config.DATASET_FILE)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_csv(Config.DATASET_FILE, index=False)
            print(f"\n✅ Data saved! Total records: {len(df)}")
            return df
        
        return None

if __name__ == "__main__":
    Config.init_app()
    collector = WeatherDataCollector()
    
    # Collect data
    print("Starting data collection...\n")
    df = collector.collect_and_save()
    
    if df is not None:
        print("\n📊 Data Summary:")
        print(df.tail())
        print(f"\nDataset shape: {df.shape}")
    else:
        print("❌ No data collected!")