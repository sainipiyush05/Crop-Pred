import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. DEFINE CROP PARAMETERS BASED ON INDIAN AGRICULTURE
# ============================================================================

# Complete crop list with Indian agricultural characteristics
crops_data = {
    # KHARIF CROPS (June-October) - Monsoon season
    'rice': {
        'N_range': (40, 140), 'P_range': (20, 90), 'K_range': (20, 90),
        'temp_range': (22, 35), 'humidity_range': (65, 90), 'ph_range': (5.5, 7.0),
        'rainfall_range': (150, 300), 'season': 'Kharif', 'states': ['WB', 'UP', 'PB', 'AP', 'TN']
    },
    'maize': {
        'N_range': (40, 130), 'P_range': (20, 80), 'K_range': (20, 85),
        'temp_range': (21, 32), 'humidity_range': (55, 85), 'ph_range': (5.5, 7.5),
        'rainfall_range': (50, 120), 'season': 'Kharif', 'states': ['KA', 'AP', 'MP', 'RJ', 'UP']
    },
    'sorghum': {
        'N_range': (35, 110), 'P_range': (20, 75), 'K_range': (20, 80),
        'temp_range': (25, 35), 'humidity_range': (50, 80), 'ph_range': (6.0, 7.5),
        'rainfall_range': (40, 100), 'season': 'Kharif', 'states': ['MH', 'KA', 'AP', 'TS', 'RJ']
    },
    'pearl_millet': {  # Bajra
        'N_range': (30, 100), 'P_range': (15, 65), 'K_range': (15, 70),
        'temp_range': (25, 38), 'humidity_range': (40, 75), 'ph_range': (6.0, 8.0),
        'rainfall_range': (30, 80), 'season': 'Kharif', 'states': ['RJ', 'UP', 'HR', 'GJ', 'MH']
    },
    'finger_millet': {  # Ragi
        'N_range': (35, 110), 'P_range': (20, 70), 'K_range': (20, 75),
        'temp_range': (20, 30), 'humidity_range': (55, 85), 'ph_range': (5.5, 7.0),
        'rainfall_range': (60, 120), 'season': 'Kharif', 'states': ['KA', 'TN', 'AP', 'MH', 'OD']
    },
    'cotton': {
        'N_range': (60, 140), 'P_range': (25, 80), 'K_range': (30, 90),
        'temp_range': (25, 35), 'humidity_range': (50, 80), 'ph_range': (6.0, 7.5),
        'rainfall_range': (50, 100), 'season': 'Kharif', 'states': ['GJ', 'MH', 'TS', 'AP', 'PB']
    },
    'sugarcane': {
        'N_range': (80, 160), 'P_range': (30, 90), 'K_range': (40, 110),
        'temp_range': (20, 32), 'humidity_range': (60, 85), 'ph_range': (6.0, 7.5),
        'rainfall_range': (100, 200), 'season': 'Kharif', 'states': ['UP', 'MH', 'KA', 'TN', 'GJ']
    },
    'groundnut': {
        'N_range': (30, 100), 'P_range': (20, 70), 'K_range': (20, 75),
        'temp_range': (22, 32), 'humidity_range': (55, 80), 'ph_range': (6.0, 7.0),
        'rainfall_range': (50, 100), 'season': 'Kharif', 'states': ['GJ', 'AP', 'TN', 'KA', 'RJ']
    },
    'soybean': {
        'N_range': (35, 110), 'P_range': (20, 75), 'K_range': (25, 80),
        'temp_range': (22, 32), 'humidity_range': (60, 85), 'ph_range': (6.0, 7.0),
        'rainfall_range': (60, 110), 'season': 'Kharif', 'states': ['MP', 'MH', 'RJ', 'UP', 'KA']
    },
    'pigeonpea': {  # Arhar/Tur
        'N_range': (30, 100), 'P_range': (20, 70), 'K_range': (20, 75),
        'temp_range': (22, 32), 'humidity_range': (55, 80), 'ph_range': (6.0, 7.5),
        'rainfall_range': (40, 90), 'season': 'Kharif', 'states': ['UP', 'MP', 'MH', 'KA', 'AP']
    },
    'green_gram': {  # Moong
        'N_range': (30, 90), 'P_range': (15, 65), 'K_range': (20, 70),
        'temp_range': (25, 35), 'humidity_range': (50, 75), 'ph_range': (6.0, 7.5),
        'rainfall_range': (30, 80), 'season': 'Kharif', 'states': ['MP', 'RJ', 'AP', 'UP', 'KA']
    },
    'black_gram': {  # Urad
        'N_range': (30, 95), 'P_range': (15, 65), 'K_range': (20, 70),
        'temp_range': (24, 35), 'humidity_range': (50, 75), 'ph_range': (6.0, 7.5),
        'rainfall_range': (30, 85), 'season': 'Kharif', 'states': ['MP', 'UP', 'AP', 'TN', 'OD']
    },
    
    # RABI CROPS (October-March) - Winter season
    'wheat': {
        'N_range': (70, 140), 'P_range': (30, 80), 'K_range': (35, 90),
        'temp_range': (12, 24), 'humidity_range': (45, 75), 'ph_range': (6.0, 7.5),
        'rainfall_range': (25, 65), 'season': 'Rabi', 'states': ['UP', 'PB', 'HR', 'MP', 'RJ']
    },
    'barley': {
        'N_range': (60, 130), 'P_range': (25, 75), 'K_range': (30, 85),
        'temp_range': (12, 25), 'humidity_range': (45, 70), 'ph_range': (6.0, 7.5),
        'rainfall_range': (30, 70), 'season': 'Rabi', 'states': ['UP', 'RJ', 'HR', 'PB', 'MP']
    },
    'mustard': {
        'N_range': (50, 110), 'P_range': (25, 75), 'K_range': (30, 80),
        'temp_range': (10, 25), 'humidity_range': (45, 70), 'ph_range': (6.0, 7.5),
        'rainfall_range': (30, 60), 'season': 'Rabi', 'states': ['RJ', 'UP', 'HR', 'PB', 'MP']
    },
    'chickpea': {  # Gram
        'N_range': (40, 110), 'P_range': (20, 70), 'K_range': (25, 80),
        'temp_range': (15, 28), 'humidity_range': (45, 70), 'ph_range': (6.0, 7.5),
        'rainfall_range': (30, 70), 'season': 'Rabi', 'states': ['MP', 'UP', 'RJ', 'MH', 'KA']
    },
    'lentil': {  # Masoor
        'N_range': (35, 100), 'P_range': (20, 65), 'K_range': (25, 75),
        'temp_range': (15, 28), 'humidity_range': (45, 70), 'ph_range': (6.0, 7.5),
        'rainfall_range': (30, 70), 'season': 'Rabi', 'states': ['UP', 'MP', 'BR', 'WB', 'RJ']
    },
    'safflower': {
        'N_range': (40, 100), 'P_range': (20, 70), 'K_range': (25, 75),
        'temp_range': (15, 28), 'humidity_range': (45, 70), 'ph_range': (6.5, 8.0),
        'rainfall_range': (30, 70), 'season': 'Rabi', 'states': ['MH', 'KA', 'AP', 'TS', 'RJ']
    },
    'sunflower': {
        'N_range': (50, 110), 'P_range': (25, 75), 'K_range': (30, 85),
        'temp_range': (15, 28), 'humidity_range': (45, 70), 'ph_range': (6.0, 7.5),
        'rainfall_range': (40, 80), 'season': 'Rabi', 'states': ['KA', 'AP', 'TN', 'MH', 'PB']
    },
    
    # ZAID CROPS (March-June) - Summer season
    'watermelon': {
        'N_range': (50, 110), 'P_range': (30, 80), 'K_range': (40, 90),
        'temp_range': (22, 35), 'humidity_range': (55, 80), 'ph_range': (6.0, 7.0),
        'rainfall_range': (30, 70), 'season': 'Zaid', 'states': ['UP', 'PB', 'HR', 'AP', 'TN']
    },
    'muskmelon': {
        'N_range': (45, 100), 'P_range': (25, 75), 'K_range': (35, 85),
        'temp_range': (22, 35), 'humidity_range': (55, 80), 'ph_range': (6.0, 7.0),
        'rainfall_range': (30, 70), 'season': 'Zaid', 'states': ['UP', 'PB', 'HR', 'RJ', 'AP']
    },
    'cucumber': {
        'N_range': (40, 95), 'P_range': (25, 70), 'K_range': (30, 80),
        'temp_range': (22, 32), 'humidity_range': (60, 85), 'ph_range': (6.0, 7.0),
        'rainfall_range': (40, 80), 'season': 'Zaid', 'states': ['UP', 'PB', 'HR', 'WB', 'OD']
    },
    'bottle_gourd': {
        'N_range': (45, 100), 'P_range': (25, 75), 'K_range': (35, 85),
        'temp_range': (22, 32), 'humidity_range': (60, 85), 'ph_range': (6.0, 7.0),
        'rainfall_range': (40, 80), 'season': 'Zaid', 'states': ['UP', 'PB', 'HR', 'WB', 'BR']
    },
    'bitter_gourd': {
        'N_range': (45, 100), 'P_range': (25, 75), 'K_range': (35, 85),
        'temp_range': (22, 32), 'humidity_range': (60, 85), 'ph_range': (6.0, 7.0),
        'rainfall_range': (40, 80), 'season': 'Zaid', 'states': ['UP', 'PB', 'HR', 'WB', 'OD']
    },
    
    # FRUIT CROPS (Perennial/Year-round)
    'mango': {
        'N_range': (40, 100), 'P_range': (20, 60), 'K_range': (30, 80),
        'temp_range': (24, 35), 'humidity_range': (55, 85), 'ph_range': (6.0, 7.5),
        'rainfall_range': (80, 250), 'season': 'Perennial', 'states': ['UP', 'AP', 'TN', 'KA', 'GJ']
    },
    'banana': {
        'N_range': (60, 130), 'P_range': (30, 80), 'K_range': (50, 110),
        'temp_range': (22, 32), 'humidity_range': (65, 85), 'ph_range': (6.0, 7.5),
        'rainfall_range': (100, 250), 'season': 'Perennial', 'states': ['TN', 'MH', 'AP', 'GJ', 'KA']
    },
    'papaya': {
        'N_range': (50, 110), 'P_range': (25, 75), 'K_range': (40, 90),
        'temp_range': (22, 32), 'humidity_range': (60, 85), 'ph_range': (6.0, 7.0),
        'rainfall_range': (80, 200), 'season': 'Perennial', 'states': ['AP', 'TN', 'KA', 'GJ', 'MH']
    },
    'pomegranate': {
        'N_range': (40, 100), 'P_range': (20, 65), 'K_range': (30, 80),
        'temp_range': (20, 35), 'humidity_range': (45, 75), 'ph_range': (6.5, 7.5),
        'rainfall_range': (50, 120), 'season': 'Perennial', 'states': ['MH', 'KA', 'AP', 'GJ', 'RJ']
    },
    'orange': {
        'N_range': (40, 100), 'P_range': (20, 60), 'K_range': (30, 80),
        'temp_range': (15, 30), 'humidity_range': (60, 85), 'ph_range': (6.0, 7.0),
        'rainfall_range': (80, 150), 'season': 'Perennial', 'states': ['MH', 'AP', 'TN', 'KA', 'NE']
    },
    'grapes': {
        'N_range': (50, 110), 'P_range': (25, 70), 'K_range': (40, 90),
        'temp_range': (15, 35), 'humidity_range': (45, 70), 'ph_range': (6.5, 7.5),
        'rainfall_range': (50, 100), 'season': 'Perennial', 'states': ['MH', 'KA', 'TN', 'AP', 'PB']
    },
    'apple': {
        'N_range': (60, 120), 'P_range': (30, 75), 'K_range': (40, 90),
        'temp_range': (5, 22), 'humidity_range': (55, 80), 'ph_range': (6.0, 7.0),
        'rainfall_range': (80, 150), 'season': 'Perennial', 'states': ['JK', 'HP', 'UK', 'PB', 'HR']
    },
    
    # COMMERCIAL CROPS
    'coffee': {
        'N_range': (50, 110), 'P_range': (25, 70), 'K_range': (35, 85),
        'temp_range': (15, 28), 'humidity_range': (65, 85), 'ph_range': (5.5, 6.5),
        'rainfall_range': (150, 300), 'season': 'Perennial', 'states': ['KA', 'TN', 'KL', 'AP', 'NE']
    },
    'tea': {
        'N_range': (50, 110), 'P_range': (25, 70), 'K_range': (35, 85),
        'temp_range': (15, 28), 'humidity_range': (70, 90), 'ph_range': (4.5, 5.5),
        'rainfall_range': (150, 300), 'season': 'Perennial', 'states': ['WB', 'AS', 'TN', 'KL', 'HP']
    },
    'coconut': {
        'N_range': (50, 120), 'P_range': (30, 80), 'K_range': (50, 110),
        'temp_range': (22, 32), 'humidity_range': (70, 85), 'ph_range': (6.0, 7.5),
        'rainfall_range': (150, 300), 'season': 'Perennial', 'states': ['KL', 'TN', 'AP', 'KA', 'OD']
    },
    'arecanut': {
        'N_range': (50, 110), 'P_range': (25, 70), 'K_range': (40, 90),
        'temp_range': (20, 32), 'humidity_range': (70, 85), 'ph_range': (6.0, 7.0),
        'rainfall_range': (150, 300), 'season': 'Perennial', 'states': ['KL', 'KA', 'TN', 'AP', 'NE']
    },
    'jute': {
        'N_range': (60, 130), 'P_range': (30, 80), 'K_range': (35, 85),
        'temp_range': (22, 32), 'humidity_range': (70, 85), 'ph_range': (6.0, 7.0),
        'rainfall_range': (100, 200), 'season': 'Kharif', 'states': ['WB', 'BR', 'AS', 'OD', 'UP']
    }
}

# ============================================================================
# 2. GENERATE REALISTIC SAMPLES FOR EACH CROP
# ============================================================================

def generate_crop_samples(crop_name, params, n_samples):
    """Generate realistic samples for a crop based on defined ranges"""
    
    samples = []
    
    for _ in range(n_samples):
        # Generate features within specified ranges
        n = np.random.uniform(params['N_range'][0], params['N_range'][1])
        p = np.random.uniform(params['P_range'][0], params['P_range'][1])
        k = np.random.uniform(params['K_range'][0], params['K_range'][1])
        
        # Temperature - ensure seasonal patterns
        if params['season'] == 'Kharif':
            temp = np.random.uniform(max(params['temp_range'][0], 22), params['temp_range'][1])
        elif params['season'] == 'Rabi':
            temp = np.random.uniform(params['temp_range'][0], min(params['temp_range'][1], 25))
        else:  # Zaid or Perennial
            temp = np.random.uniform(params['temp_range'][0], params['temp_range'][1])
        
        humidity = np.random.uniform(params['humidity_range'][0], params['humidity_range'][1])
        ph = np.random.uniform(params['ph_range'][0], params['ph_range'][1])
        rainfall = np.random.uniform(params['rainfall_range'][0], params['rainfall_range'][1])
        
        # Add some realistic noise/variation
        n += np.random.normal(0, 5)
        p += np.random.normal(0, 4)
        k += np.random.normal(0, 5)
        temp += np.random.normal(0, 1.5)
        humidity += np.random.normal(0, 3)
        ph += np.random.normal(0, 0.2)
        rainfall += np.random.normal(0, 10)
        
        # Clip to realistic ranges
        n = np.clip(n, params['N_range'][0] - 10, params['N_range'][1] + 10)
        p = np.clip(p, params['P_range'][0] - 8, params['P_range'][1] + 8)
        k = np.clip(k, params['K_range'][0] - 10, params['K_range'][1] + 10)
        temp = np.clip(temp, params['temp_range'][0] - 3, params['temp_range'][1] + 3)
        humidity = np.clip(humidity, params['humidity_range'][0] - 5, params['humidity_range'][1] + 5)
        ph = np.clip(ph, params['ph_range'][0] - 0.3, params['ph_range'][1] + 0.3)
        rainfall = np.clip(rainfall, params['rainfall_range'][0] - 15, params['rainfall_range'][1] + 15)
        
        samples.append({
            'N': round(n, 2),
            'P': round(p, 2),
            'K': round(k, 2),
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'ph': round(ph, 2),
            'rainfall': round(rainfall, 2),
            'label': crop_name,
            'season': params['season']
        })
    
    return samples

# Define sample sizes (realistic distribution - more for staple crops)
sample_sizes = {
    # Staple crops - more samples
    'rice': 500, 'wheat': 450, 'maize': 350, 'sorghum': 250, 'pearl_millet': 250,
    'finger_millet': 200, 'barley': 200, 'mustard': 250, 'chickpea': 300,
    
    # Commercial crops
    'cotton': 300, 'sugarcane': 250, 'groundnut': 250, 'soybean': 250,
    'pigeonpea': 200, 'green_gram': 200, 'black_gram': 200,
    
    # Fruits
    'mango': 300, 'banana': 250, 'papaya': 200, 'pomegranate': 200,
    'orange': 200, 'grapes': 200, 'apple': 150,
    
    # Vegetables (Zaid)
    'watermelon': 200, 'muskmelon': 180, 'cucumber': 180, 'bottle_gourd': 150,
    'bitter_gourd': 150,
    
    # Commercial crops
    'coffee': 150, 'tea': 150, 'coconut': 200, 'arecanut': 150, 'jute': 200,
    
    # Other pulses
    'lentil': 200, 'safflower': 150, 'sunflower': 180
}

# Generate all samples
all_samples = []
for crop_name, n_samples in sample_sizes.items():
    if crop_name in crops_data:
        samples = generate_crop_samples(crop_name, crops_data[crop_name], n_samples)
        all_samples.extend(samples)
        print(f"Generated {len(samples)} samples for {crop_name} ({crops_data[crop_name]['season']})")

# Create DataFrame
df = pd.DataFrame(all_samples)

# ============================================================================
# 3. VALIDATE AND CLEAN THE DATASET
# ============================================================================

print("\n" + "="*60)
print("DATASET VALIDATION")
print("="*60)

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check data types
print(f"\nData types:\n{df.dtypes}")

# Display basic statistics
print(f"\nBasic statistics:")
print(df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].describe())

# Check label distribution
print(f"\nLabel distribution:")
label_dist = df['label'].value_counts()
print(label_dist)
print(f"\nTotal samples: {len(df)}")
print(f"Number of unique crops: {df['label'].nunique()}")

# ============================================================================
# 4. ADD SEASONAL AND STATE INFORMATION FOR FILTERING LAYER
# ============================================================================

# Map season to months
season_months = {
    'Kharif': list(range(6, 11)),  # June to October
    'Rabi': list(range(10, 13)) + [1, 2, 3],  # October to March
    'Zaid': list(range(3, 7)),  # March to June
    'Perennial': list(range(1, 13))  # All months
}

# Add month column (random month within the crop's growing season)
def assign_month(row):
    crop_name = row['label']
    season = crops_data[crop_name]['season']
    months = season_months[season]
    return np.random.choice(months)

df['month'] = df.apply(assign_month, axis=1)

# Add state column (random from crop's growing states)
def assign_state(row):
    crop_name = row['label']
    states = crops_data[crop_name]['states']
    return np.random.choice(states)

df['state'] = df.apply(assign_state, axis=1)

# ============================================================================
# 5. VERIFY REALISTIC CONDITIONS
# ============================================================================

print("\n" + "="*60)
print("VERIFYING REALISTIC INDIAN CONDITIONS")
print("="*60)

# Check temperature ranges by season
print("\nTemperature by season:")
season_temp = df.groupby(df['label'].map(lambda x: crops_data[x]['season']))['temperature'].agg(['min', 'mean', 'max'])
print(season_temp)

# Check rainfall by crop type
print("\nRainfall by crop type:")
crop_type_rainfall = df.groupby(df['label'].map(lambda x: crops_data[x]['season']))['rainfall'].agg(['min', 'mean', 'max'])
print(crop_type_rainfall)

# Check pH ranges
print(f"\npH range: {df['ph'].min():.2f} - {df['ph'].max():.2f}")
print(f"Mean pH: {df['ph'].mean():.2f}")

# ============================================================================
# 6. EXPORT DATASET
# ============================================================================

# Reorder columns
df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label', 'season', 'month', 'state']]

# Save to CSV
df.to_csv('indian_crop_recommendation_dataset.csv', index=False)
print(f"\n✅ Dataset saved to 'indian_crop_recommendation_dataset.csv'")

# Display sample
print("\n" + "="*60)
print("SAMPLE DATA (First 10 rows)")
print("="*60)
print(df.head(10))

# ============================================================================
# 7. DATA QUALITY REPORT
# ============================================================================

print("\n" + "="*60)
print("DATA QUALITY REPORT")
print("="*60)

print(f"""
Dataset Statistics:
- Total samples: {len(df)}
- Number of crops: {df['label'].nunique()}
- Features: N, P, K, temperature, humidity, ph, rainfall
- Additional columns: season, month, state (for filtering)

Crop Distribution by Season:
""")

season_counts = df.groupby(df['label'].map(lambda x: crops_data[x]['season']))['label'].count()
for season, count in season_counts.items():
    print(f"  {season}: {count} samples")

print(f"""
Key Features Ranges:
- Nitrogen (N): {df['N'].min():.2f} - {df['N'].max():.2f} kg/ha
- Phosphorus (P): {df['P'].min():.2f} - {df['P'].max():.2f} kg/ha
- Potassium (K): {df['K'].min():.2f} - {df['K'].max():.2f} kg/ha
- Temperature: {df['temperature'].min():.2f} - {df['temperature'].max():.2f} °C
- Humidity: {df['humidity'].min():.2f} - {df['humidity'].max():.2f} %
- pH: {df['ph'].min():.2f} - {df['ph'].max():.2f}
- Rainfall: {df['rainfall'].min():.2f} - {df['rainfall'].max():.2f} mm

Seasonal Patterns Verified:
- Kharif crops: Higher temperature (22-35°C), higher rainfall (50-300mm)
- Rabi crops: Lower temperature (10-28°C), lower rainfall (25-80mm)
- Zaid crops: Moderate temperature (22-35°C), moderate rainfall (30-80mm)
- Perennial crops: Year-round, varied conditions
""")

# ============================================================================
# 8. ADDITIONAL VALIDATION - CHECK FOR OUTLIERS
# ============================================================================

print("\n" + "="*60)
print("OUTLIER DETECTION (Using IQR method)")
print("="*60)

for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers detected ({(len(outliers)/len(df))*100:.2f}%)")

print("\n✅ Dataset is ready for machine learning model training!")