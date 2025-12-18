import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
from config import Config
import warnings
warnings.filterwarnings('ignore')

class WeatherModelTrainer:
    """Train ML models for weather prediction"""
    
    def __init__(self):
        Config.init_app()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and prepare dataset"""
        try:
            df = pd.read_csv(Config.DATASET_FILE)
            print(f"✅ Loaded {len(df)} records")
            
            # Handle missing values
            df = df.fillna(df.mean(numeric_only=True))
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['month'] = df['timestamp'].dt.month
            
            return df
            
        except FileNotFoundError:
            print("❌ Dataset not found! Run data_collector.py first.")
            return None
    
    def train_temperature_model(self, df):
        """Train temperature prediction model (Regression)"""
        print("\n🔥 Training Temperature Prediction Model...")
        
        features = ['feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 
                   'wind_speed', 'clouds', 'pm2_5', 'pm10', 'hour', 'month']
        
        X = df[features].values
        y = df['temperature'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=150, max_depth=10, random_state=42)
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n{name}:")
            print(f"  MAE: {mae:.3f}°C")
            print(f"  RMSE: {rmse:.3f}°C")
            print(f"  R² Score: {r2:.3f}")
            
            if mae < best_score:
                best_score = mae
                best_model = model
        
        # Save best model
        joblib.dump(best_model, Config.TEMP_MODEL_FILE)
        joblib.dump(self.scaler, Config.SCALER_FILE)
        print(f"\n✅ Temperature model saved! (MAE: {best_score:.3f}°C)")
        
        return best_model
    
    def train_weather_classifier(self, df):
        """Train weather condition classifier"""
        print("\n☁️ Training Weather Condition Classifier...")
        
        features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'clouds', 
                   'pm2_5', 'aqi', 'hour', 'month']
        
        X = df[features].values
        y = df['weather_main'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n  Accuracy: {accuracy:.3f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Save model
        joblib.dump(clf, Config.WEATHER_MODEL_FILE)
        print(f"\n✅ Weather classifier saved! (Accuracy: {accuracy:.3f})")
        
        return clf
    
    def train_humidity_model(self, df):
        """Train humidity prediction model"""
        print("\n💧 Training Humidity Prediction Model...")
        
        features = ['temperature', 'pressure', 'wind_speed', 'clouds', 
                   'pm2_5', 'hour', 'month']
        
        X = df[features].values
        y = df['humidity'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n  MAE: {mae:.3f}%")
        print(f"  RMSE: {rmse:.3f}%")
        
        # Save model
        joblib.dump(model, Config.HUMIDITY_MODEL_FILE)
        print(f"\n✅ Humidity model saved! (MAE: {mae:.3f}%)")
        
        return model
    
    def train_all_models(self):
        """Train all ML models"""
        print("=" * 60)
        print("🚀 STARTING ML MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Train models
        self.train_temperature_model(df)
        self.train_weather_classifier(df)
        self.train_humidity_model(df)
        
        print("\n" + "=" * 60)
        print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 60)

if __name__ == "__main__":
    trainer = WeatherModelTrainer()
    trainer.train_all_models()