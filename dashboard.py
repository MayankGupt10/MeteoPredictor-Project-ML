"""
Futuristic Analytics Dashboard with ML Training Visualization
"""

import pandas as pd
import json
from datetime import datetime
from config import Config

def create_dashboard():
    """Create futuristic analytics dashboard"""
    
    try:
        df = pd.read_csv(Config.DATASET_FILE)
        print(f"✅ Loaded {len(df)} records")
    except:
        print("❌ No data found! Run generate_sample_data.py first")
        return None
    
    # Calculate advanced statistics
    stats = {
        'total_records': len(df),
        'cities': df['city'].nunique(),
        'avg_temp': round(df['temperature'].mean(), 2),
        'max_temp': round(df['temperature'].max(), 2),
        'min_temp': round(df['temperature'].min(), 2),
        'avg_humidity': round(df['humidity'].mean(), 2),
        'avg_aqi': round(df['aqi'].mean(), 2),
        'total_cities': list(df['city'].unique())
    }
    
    # Prepare comprehensive data for visualizations
    city_stats = df.groupby('city').agg({
        'temperature': ['mean', 'min', 'max'],
        'humidity': 'mean',
        'aqi': 'mean',
        'pm2_5': 'mean',
        'pm10': 'mean',
        'wind_speed': 'mean',
        'pressure': 'mean'
    }).round(2)
    
    weather_dist = df['weather_main'].value_counts().to_dict()
    aqi_dist = df['aqi'].value_counts().sort_index().to_dict()
    
    # City-wise data
    temp_by_city = df.groupby('city')['temperature'].mean().sort_values(ascending=False).to_dict()
    aqi_by_city = df.groupby('city')['aqi'].mean().sort_values(ascending=False).to_dict()
    humidity_by_city = df.groupby('city')['humidity'].mean().sort_values(ascending=False).to_dict()
    pm25_by_city = df.groupby('city')['pm2_5'].mean().sort_values(ascending=False).to_dict()
    
    # Time series data (simulate hourly data)
    hourly_temp = [round(df['temperature'].mean() + (i-12)*0.5, 2) for i in range(24)]
    hourly_humidity = [round(df['humidity'].mean() + (i-12)*-0.3, 2) for i in range(24)]
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WeatherML - Advanced Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Rajdhani', sans-serif;
            background: #000000;
            color: #00ff88;
            overflow-x: hidden;
        }}
        
        /* Animated Grid Background */
        .grid-bg {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                linear-gradient(90deg, rgba(0, 255, 136, 0.03) 1px, transparent 1px),
                linear-gradient(rgba(0, 255, 136, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: gridScroll 20s linear infinite;
            z-index: 0;
        }}
        
        @keyframes gridScroll {{
            0% {{ transform: translate(0, 0); }}
            100% {{ transform: translate(50px, 50px); }}
        }}
        
        /* Glowing Orbs */
        .orb {{
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            animation: float 20s ease-in-out infinite;
            z-index: 0;
        }}
        
        .orb1 {{
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, #00ff88, transparent);
            top: -250px;
            left: -250px;
        }}
        
        .orb2 {{
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, #00ffff, transparent);
            bottom: -200px;
            right: -200px;
            animation-delay: -10s;
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translate(0, 0) scale(1); }}
            50% {{ transform: translate(100px, 100px) scale(1.2); }}
        }}
        
        .container {{
            position: relative;
            z-index: 1;
            max-width: 1900px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        /* Top Bar */
        .top-bar {{
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 2px solid rgba(0, 255, 136, 0.3);
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
        }}
        
        .logo {{
            font-family: 'Orbitron', sans-serif;
            font-size: 2em;
            font-weight: 900;
            letter-spacing: 3px;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.8);
        }}
        
        .nav-btn {{
            padding: 12px 30px;
            background: linear-gradient(135deg, #00ff88, #00ffff);
            color: #000;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            text-decoration: none;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s ease;
        }}
        
        .nav-btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.8);
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(20px);
            border: 2px solid rgba(0, 255, 136, 0.3);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0, 255, 136, 0.1) 0%, transparent 70%);
            animation: rotate 15s linear infinite;
        }}
        
        @keyframes rotate {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        
        .stat-card:hover {{
            border-color: #00ff88;
            box-shadow: 0 0 40px rgba(0, 255, 136, 0.4);
            transform: translateY(-5px);
        }}
        
        .stat-icon {{
            font-size: 2.5em;
            margin-bottom: 15px;
            filter: drop-shadow(0 0 15px rgba(0, 255, 136, 0.8));
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: 700;
            font-family: 'Orbitron', sans-serif;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.8);
            margin: 10px 0;
            position: relative;
            z-index: 1;
        }}
        
        .stat-label {{
            color: rgba(0, 255, 136, 0.6);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Charts Grid */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 25px;
            margin-bottom: 25px;
        }}
        
        .chart-card {{
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(20px);
            border: 2px solid rgba(0, 255, 136, 0.3);
            border-radius: 20px;
            padding: 30px;
            transition: all 0.4s ease;
        }}
        
        .chart-card:hover {{
            border-color: #00ff88;
            box-shadow: 0 0 50px rgba(0, 255, 136, 0.3);
        }}
        
        .chart-card.full {{
            grid-column: 1 / -1;
        }}
        
        .chart-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(0, 255, 136, 0.2);
        }}
        
        .chart-icon {{
            font-size: 2em;
            filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.8));
        }}
        
        .chart-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.4em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        
        .chart-container {{
            position: relative;
            height: 350px;
        }}
        
        .chart-container.large {{
            height: 450px;
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(0, 255, 136, 0.05);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(135deg, #00ff88, #00ffff);
            border-radius: 10px;
        }}
        
        @media (max-width: 1400px) {{
            .stats-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}
        
        @media (max-width: 1024px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="grid-bg"></div>
    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    
    <div class="container">
        <div class="top-bar">
            <div class="logo">⚡ ANALYTICS DASHBOARD</div>
            <a href="/" class="nav-btn">← Back</a>
        </div>
        
        <!-- Stats Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-value">{stats['total_records']}</div>
                <div class="stat-label">Total Records</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">🏙️</div>
                <div class="stat-value">{stats['cities']}</div>
                <div class="stat-label">Cities</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">🌡️</div>
                <div class="stat-value">{stats['avg_temp']}°</div>
                <div class="stat-label">Avg Temp</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">💧</div>
                <div class="stat-value">{stats['avg_humidity']}%</div>
                <div class="stat-label">Avg Humidity</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">💨</div>
                <div class="stat-value">{stats['avg_aqi']}</div>
                <div class="stat-label">Avg AQI</div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-icon">🌡️</span>
                    <span class="chart-title">Temperature Analysis</span>
                </div>
                <div class="chart-container">
                    <canvas id="tempChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-icon">💨</span>
                    <span class="chart-title">AQI Distribution</span>
                </div>
                <div class="chart-container">
                    <canvas id="aqiChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-icon">📈</span>
                    <span class="chart-title">Humidity Trends</span>
                </div>
                <div class="chart-container">
                    <canvas id="humidityChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-icon">☁️</span>
                    <span class="chart-title">Weather Distribution</span>
                </div>
                <div class="chart-container">
                    <canvas id="weatherChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card full">
                <div class="chart-header">
                    <span class="chart-icon">🔬</span>
                    <span class="chart-title">Multi-Metric Comparison</span>
                </div>
                <div class="chart-container large">
                    <canvas id="comparisonChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card full">
                <div class="chart-header">
                    <span class="chart-icon">📊</span>
                    <span class="chart-title">Hourly Forecast Simulation</span>
                </div>
                <div class="chart-container large">
                    <canvas id="hourlyChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-icon">🏭</span>
                    <span class="chart-title">PM2.5 Pollution</span>
                </div>
                <div class="chart-container">
                    <canvas id="pm25Chart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-icon">🎯</span>
                    <span class="chart-title">City Comparison</span>
                </div>
                <div class="chart-container">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        Chart.defaults.color = '#00ff88';
        Chart.defaults.borderColor = 'rgba(0, 255, 136, 0.2)';
        Chart.defaults.font.family = "'Rajdhani', sans-serif";
        
        // Temperature Chart
        new Chart(document.getElementById('tempChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(list(temp_by_city.keys()))},
                datasets: [{{
                    label: 'Temperature (°C)',
                    data: {json.dumps(list(temp_by_city.values()))},
                    backgroundColor: 'rgba(0, 255, 136, 0.6)',
                    borderColor: '#00ff88',
                    borderWidth: 2,
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        grid: {{ color: 'rgba(0, 255, 136, 0.1)' }},
                        ticks: {{ color: '#00ff88' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#00ff88' }}
                    }}
                }}
            }}
        }});
        
        // AQI Chart
        new Chart(document.getElementById('aqiChart'), {{
            type: 'polarArea',
            data: {{
                labels: {json.dumps(list(aqi_by_city.keys()))},
                datasets: [{{
                    label: 'AQI',
                    data: {json.dumps(list(aqi_by_city.values()))},
                    backgroundColor: [
                        'rgba(0, 255, 136, 0.6)',
                        'rgba(0, 255, 255, 0.6)',
                        'rgba(255, 170, 0, 0.6)',
                        'rgba(255, 0, 68, 0.6)',
                        'rgba(136, 0, 255, 0.6)'
                    ],
                    borderColor: '#00ff88',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right',
                        labels: {{ color: '#00ff88', padding: 15 }}
                    }}
                }},
                scales: {{
                    r: {{
                        ticks: {{ color: '#00ff88', backdropColor: 'transparent' }},
                        grid: {{ color: 'rgba(0, 255, 136, 0.2)' }}
                    }}
                }}
            }}
        }});
        
        // Humidity Chart
        new Chart(document.getElementById('humidityChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(list(humidity_by_city.keys()))},
                datasets: [{{
                    label: 'Humidity (%)',
                    data: {json.dumps(list(humidity_by_city.values()))},
                    backgroundColor: 'rgba(0, 255, 255, 0.2)',
                    borderColor: '#00ffff',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 6,
                    pointBackgroundColor: '#00ffff',
                    pointBorderColor: '#000',
                    pointBorderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#00ff88' }} }} }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        grid: {{ color: 'rgba(0, 255, 136, 0.1)' }},
                        ticks: {{ color: '#00ff88' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#00ff88' }}
                    }}
                }}
            }}
        }});
        
        // Weather Distribution
        new Chart(document.getElementById('weatherChart'), {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(list(weather_dist.keys()))},
                datasets: [{{
                    data: {json.dumps(list(weather_dist.values()))},
                    backgroundColor: [
                        'rgba(0, 255, 136, 0.8)',
                        'rgba(0, 255, 255, 0.8)',
                        'rgba(255, 170, 0, 0.8)',
                        'rgba(255, 0, 68, 0.8)',
                        'rgba(136, 0, 255, 0.8)'
                    ],
                    borderColor: '#000',
                    borderWidth: 3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{ color: '#00ff88', padding: 15 }}
                    }}
                }}
            }}
        }});
        
        // Multi-Metric Comparison
        const cities = {json.dumps(list(temp_by_city.keys()))};
        new Chart(document.getElementById('comparisonChart'), {{
            type: 'line',
            data: {{
                labels: cities,
                datasets: [
                    {{
                        label: 'Temperature',
                        data: {json.dumps(list(temp_by_city.values()))},
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Humidity',
                        data: {json.dumps(list(humidity_by_city.values()))},
                        borderColor: '#00ffff',
                        backgroundColor: 'rgba(0, 255, 255, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                plugins: {{ legend: {{ labels: {{ color: '#00ff88', padding: 20 }} }} }},
                scales: {{
                    y: {{
                        type: 'linear',
                        position: 'left',
                        grid: {{ color: 'rgba(0, 255, 136, 0.1)' }},
                        ticks: {{ color: '#00ff88' }},
                        title: {{ display: true, text: 'Temperature (°C)', color: '#00ff88' }}
                    }},
                    y1: {{
                        type: 'linear',
                        position: 'right',
                        grid: {{ display: false }},
                        ticks: {{ color: '#00ffff' }},
                        title: {{ display: true, text: 'Humidity (%)', color: '#00ffff' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#00ff88' }}
                    }}
                }}
            }}
        }});
        
        // Hourly Forecast
        new Chart(document.getElementById('hourlyChart'), {{
            type: 'line',
            data: {{
                labels: Array.from({{length: 24}}, (_, i) => i + ':00'),
                datasets: [
                    {{
                        label: 'Temperature',
                        data: {json.dumps(hourly_temp)},
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.2)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }},
                    {{
                        label: 'Humidity',
                        data: {json.dumps(hourly_humidity)},
                        borderColor: '#00ffff',
                        backgroundColor: 'rgba(0, 255, 255, 0.2)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#00ff88', padding: 20 }} }} }},
                scales: {{
                    y: {{
                        grid: {{ color: 'rgba(0, 255, 136, 0.1)' }},
                        ticks: {{ color: '#00ff88' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#00ff88' }}
                    }}
                }}
            }}
        }});
        
        // PM2.5 Chart
        new Chart(document.getElementById('pm25Chart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(list(pm25_by_city.keys()))},
                datasets: [{{
                    label: 'PM2.5',
                    data: {json.dumps(list(pm25_by_city.values()))},
                    backgroundColor: 'rgba(255, 0, 68, 0.7)',
                    borderColor: '#ff0044',
                    borderWidth: 2,
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{ color: 'rgba(0, 255, 136, 0.1)' }},
                        ticks: {{ color: '#00ff88' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#00ff88' }}
                    }}
                }}
            }}
        }});
        
        // Radar Chart
        new Chart(document.getElementById('radarChart'), {{
            type: 'radar',
            data: {{
                labels: ['Temperature', 'Humidity', 'AQI', 'PM2.5', 'Wind'],
                datasets: {json.dumps([{
                    'label': city,
                    'data': [
                        float(temp_by_city.get(city, 0)),
                        float(humidity_by_city.get(city, 0)),
                        float(aqi_by_city.get(city, 0)) * 20,
                        float(pm25_by_city.get(city, 0)),
                        20
                    ]
                } for city in list(temp_by_city.keys())[:3]])}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#00ff88' }} }} }},
                scales: {{
                    r: {{
                        ticks: {{ color: '#00ff88', backdropColor: 'transparent' }},
                        grid: {{ color: 'rgba(0, 255, 136, 0.2)' }},
                        pointLabels: {{ color: '#00ff88' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    dashboard_file = 'templates/dashboard.html'
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Futuristic dashboard created: {dashboard_file}")
    return dashboard_file

if __name__ == "__main__":
    Config.init_app()
    create_dashboard()
    print("\n🚀 Dashboard ready!")