"""
Generate Sample Weather Data for Testing
Creates realistic sample data if you don't want to wait for real API data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config
import random

def generate_sample_data(num_records=500):
    """
    Generate realistic sample weather data for training
    
    Args:
        num_records: Number of sample records to generate
    """
    Config.init_app()
    
    print("=" * 60)
    print("🎲 GENERATING SAMPLE WEATHER DATA")
    print("=" * 60)
    
    cities = Config.CITIES
    weather_conditions = ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Mist', 'Haze']
    weather_descriptions = {
        'Clear': ['clear sky', 'sunny'],
        'Clouds': ['few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds'],
        'Rain': ['light rain', 'moderate rain', 'heavy rain'],
        'Drizzle': ['light drizzle', 'drizzle'],
        'Mist': ['mist', 'fog'],
        'Haze': ['haze', 'smoke']
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(num_records):
        city = random.choice(cities)
        timestamp = start_date + timedelta(hours=i)
        
        # Base temperature varies by city
        base_temp = {
            'Delhi': 25, 'Mumbai': 28, 'Bangalore': 23,
            'Chennai': 30, 'Kolkata': 27, 'Hyderabad': 26,
            'Pune': 24, 'Ahmedabad': 28
        }.get(city, 25)
        
        # Add seasonal and daily variation
        hour = timestamp.hour
        temp_variation = np.sin((hour - 6) * np.pi / 12) * 5  # Daily cycle
        temperature = base_temp + temp_variation + np.random.normal(0, 2)
        
        # Weather conditions affect other parameters
        weather_main = random.choice(weather_conditions)
        description = random.choice(weather_descriptions[weather_main])
        
        # Humidity (higher for rain, lower for clear)
        if weather_main == 'Rain' or weather_main == 'Drizzle':
            humidity = np.random.randint(70, 95)
            clouds = np.random.randint(80, 100)
        elif weather_main == 'Clear':
            humidity = np.random.randint(30, 60)
            clouds = np.random.randint(0, 20)
        else:
            humidity = np.random.randint(50, 80)
            clouds = np.random.randint(40, 80)
        
        # AQI (varies by city - Delhi/Kolkata worse, Bangalore better)
        if city in ['Delhi', 'Kolkata']:
            aqi = np.random.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.2])
        elif city in ['Bangalore', 'Pune']:
            aqi = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        else:
            aqi = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        
        # PM2.5 based on AQI
        pm2_5_ranges = {1: (0, 30), 2: (30, 60), 3: (60, 90), 4: (90, 120), 5: (120, 250)}
        pm2_5 = np.random.uniform(*pm2_5_ranges[aqi])
        pm10 = pm2_5 * 1.5 + np.random.normal(0, 10)
        
        record = {
            'timestamp': timestamp,
            'city': city,
            'temperature': round(temperature, 2),
            'feels_like': round(temperature + np.random.uniform(-2, 2), 2),
            'temp_min': round(temperature - np.random.uniform(1, 3), 2),
            'temp_max': round(temperature + np.random.uniform(1, 3), 2),
            'pressure': int(1013 + np.random.normal(0, 10)),
            'humidity': humidity,
            'wind_speed': round(np.random.uniform(0.5, 8), 2),
            'wind_deg': np.random.randint(0, 360),
            'clouds': clouds,
            'weather_main': weather_main,
            'weather_description': description,
            'aqi': aqi,
            'pm2_5': round(pm2_5, 2),
            'pm10': round(max(0, pm10), 2),
            'co': round(np.random.uniform(200, 1000), 2),
            'no2': round(np.random.uniform(10, 50), 2),
            'o3': round(np.random.uniform(20, 100), 2),
            'so2': round(np.random.uniform(5, 30), 2)
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(Config.DATASET_FILE, index=False)
    
    print(f"\n✅ Generated {num_records} sample records!")
    print(f"📁 Saved to: {Config.DATASET_FILE}")
    print(f"\n📊 Data Summary:")
    print(f"   Cities: {df['city'].nunique()}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
    print(f"   Weather conditions: {df['weather_main'].unique()}")
    print(f"\n🚀 You can now train your ML models!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    # Generate 500 sample records
    df = generate_sample_data(num_records=500)
    
    # Show first few records
    print("\n📋 Sample Data Preview:")
    print(df.head(10))