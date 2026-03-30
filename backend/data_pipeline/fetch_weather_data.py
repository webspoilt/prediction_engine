import os
import time
import requests
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path("e:/IDEAS/ipl prediction engine")
MATCHES_CSV = BASE_DIR / "dataset/archive/matches.csv"
ENHANCED_DIR = BASE_DIR / "dataset/enhanced"
WEATHER_CSV = ENHANCED_DIR / "weather_dataset.csv"

# Global cache to reduce repetitive API calls to Geocoder
GEOCODE_CACHE = {}

def get_coordinates(city_name):
    """Fetch Lat/Lon for a city using Open-Meteo Geocoding API."""
    if pd.isna(city_name) or not str(city_name).strip():
        return None, None
        
    city_name = str(city_name).strip()
    
    if city_name in GEOCODE_CACHE:
        return GEOCODE_CACHE[city_name]

    # Special handling for known tricky names in IPL data
    search_name = city_name
    if city_name == "Bengaluru": search_name = "Bangalore"
    if city_name == "Dharamsala": search_name = "Dharamshala"
    if city_name == "Chandigarh": search_name = "Chandigarh"

    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={search_name}&count=1&format=json"
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            lat = data['results'][0]['latitude']
            lon = data['results'][0]['longitude']
            GEOCODE_CACHE[city_name] = (lat, lon)
            time.sleep(0.1) # Be gentle on rate limits
            return lat, lon
        else:
            print(f"Warning: Could not geocode city '{city_name}'")
            GEOCODE_CACHE[city_name] = (None, None)
            return None, None
    except Exception as e:
        print(f"Error geocoding '{city_name}': {e}")
        return None, None

def fetch_match_weather(lat, lon, date_str):
    """Fetch hourly weather for a specific lat/lon and date, and aggregate it."""
    # We want temperature, relative humidity, and dew point.
    url = f"https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m",
        "timezone": "auto" # Returns local time for the coords
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"Error fetching weather for {lat},{lon} on {date_str}: {response.status_code} - {response.text}")
            return None
            
        data = response.json()
        
        if 'hourly' not in data:
            return None
            
        hourly = data['hourly']
        
        # Most IPL matches are played between 16:00 and 23:00 local time
        # We slice indices 16 to 23 (inclusive)
        if len(hourly['temperature_2m']) >= 24:
            temps = hourly['temperature_2m'][16:24]
            hums = hourly['relative_humidity_2m'][16:24]
            dews = hourly['dew_point_2m'][16:24]
            
            # Remove Nones if any
            temps = [t for t in temps if t is not None]
            hums = [h for h in hums if h is not None]
            dews = [d for d in dews if d is not None]
            
            if not temps: return None
            
            return {
                "temp_mean": round(sum(temps)/len(temps), 2),
                "humidity_mean": round(sum(hums)/len(hums), 2),
                "dew_point_mean": round(sum(dews)/len(dews), 2)
            }
        return None
        
    except Exception as e:
        print(f"Exception fetching weather for {date_str}: {e}")
        return None

def main():
    print(f"Loading matches from {MATCHES_CSV}...")
    df = pd.read_csv(MATCHES_CSV)
    
    # We need to map each match based on date and city. If city is missing, use venue
    df['search_city'] = df['city'].fillna(df['venue'])
    
    # Extract unique date-city combinations to minimize API calls
    unique_contexts = df[['date', 'search_city']].drop_duplicates().dropna()
    
    print(f"Found {len(unique_contexts)} unique date-city combinations.")
    
    weather_records = []
    
    # Create directory if doesn't exist
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each combination
    total = len(unique_contexts)
    for idx, row in enumerate(unique_contexts.itertuples(index=False), 1):
        date_str = str(row.date)[:10] # Handle any YYYY-MM-DD HH:MM types
        city_name = row.search_city
        
        # Open-Meteo expects dates as YYYY-MM-DD
        try:
            # Simple check if date is valid format, if it's like DD-MM-YYYY we need to convert
            # In matches.csv some are YYYY-MM-DD
            dt = pd.to_datetime(date_str)
            api_date_str = dt.strftime('%Y-%m-%d')
        except:
            print(f"Could not parse date: {date_str}")
            continue
            
        lat, lon = get_coordinates(city_name)
        if lat is None or lon is None:
            continue
            
        print(f"[{idx}/{total}] Fetching weather for {city_name} on {api_date_str}...")
        
        weather_data = fetch_match_weather(lat, lon, api_date_str)
        if weather_data:
            record = {
                "date": date_str,  # Keep original date string for merging back later
                "search_city": city_name,
                "api_date": api_date_str,
                "lat": lat,
                "lon": lon,
                **weather_data
            }
            weather_records.append(record)
            
        # Very brief pause to respect rate limits
        time.sleep(0.3)
        
        # Save every 100 rows just in case it crashes
        if idx % 100 == 0:
            pd.DataFrame(weather_records).to_csv(WEATHER_CSV, index=False)
            print(f"Checkpoint saved to {WEATHER_CSV}")
            
    # Final save
    result_df = pd.DataFrame(weather_records)
    result_df.to_csv(WEATHER_CSV, index=False)
    print(f"\nDone! Successfully fetched weather for {len(result_df)} matches out of {total} combinations.")
    print(f"Saved complete weather dataset to {WEATHER_CSV}")

if __name__ == "__main__":
    main()
