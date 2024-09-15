import streamlit as st
from indian_beaches import indian_beaches
import requests
import requests_cache
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
import datetime
from math import radians, sin, cos, sqrt, atan2
import urllib3
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pytz


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize the geocoder
geolocator = Nominatim(user_agent="coastal_app")

# Setup caching for requests
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)

# Load the model, label encoder, and scaler
loaded_model = pickle.load(open('beach_model.sav', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define features
features = [
    'wave_height (m)', 'wind_wave_height (m)', 'swell_wave_height (m)', 'ocean_current_velocity (km/h)',
    'temperature_2m (°C)', 'relative_humidity_2m (%)', 'apparent_temperature (°C)',
    'precipitation_probability (%)', 'pressure_msl (hPa)', 'cloud_cover (%)', 'visibility (m)',
    'wind_speed_10m (km/h)', 'wind_gusts_10m (km/h)'
]

def fetch_weather_data(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation_probability",
            "pressure_msl",
            "cloud_cover",
            "visibility",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m"
        ],
        "timezone": "GMT"
    }
    response = cache_session.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_marine_data(latitude, longitude):
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "wave_height",
            "wave_direction",
            "wind_wave_height",
            "swell_wave_height",
            "ocean_current_velocity"
        ]
    }
    response = cache_session.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    hourly = data['hourly']
    hourly_data = {
        "Time": pd.to_datetime(hourly['time']),
        "Wave Height": hourly['wave_height'],
        "Wave Direction": hourly['wave_direction'],
        "Wind Wave Height": hourly['wind_wave_height'],
        "Swell Wave Height": hourly['swell_wave_height'],
        "Ocean Current Velocity": hourly['ocean_current_velocity']
    }
    return pd.DataFrame(data=hourly_data)

def geocode_place(place_name):
    try:
        location = geolocator.geocode(place_name, exactly_one=True, timeout=10)
        if location:
            return location.latitude, location.longitude, location.address
        else:
            return None, None, None
    except GeocoderServiceError as e:
        st.error(f"Geocoding service error: {e}")
        return None, None, None

def fetch_ocean_alerts():
    url = "https://tsunami.incois.gov.in/itews/DSSProducts/OPR/past90days.json"
    response = requests.get(url, verify=False)
    data = response.json()
    return data['datasets']

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def predict_suitability_ml(new_data):
    new_data_scaled = scaler.transform(new_data)
    predicted_class = loaded_model.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_class)
    return predicted_label[0]

def determine_suitability(wind_speed, wave_height, wind_gusts, visibility, alerts, activity_type, current_weather, current_marine):
    new_data = pd.DataFrame([[
        wave_height,
        current_marine['Wind Wave Height'],
        current_marine['Swell Wave Height'],
        current_marine['Ocean Current Velocity'],
        current_weather['temperature_2m'],
        current_weather['relative_humidity_2m'],
        current_weather['apparent_temperature'],
        current_weather['precipitation_probability'],
        current_weather['pressure_msl'],
        current_weather['cloud_cover'],
        visibility,
        wind_speed,
        wind_gusts
    ]], columns=features)
    
    ml_prediction = predict_suitability_ml(new_data)
    suitability = ml_prediction
    
    if suitability == "Suitable":
        color = "green"
    elif suitability == "Caution":
        color = "yellow"
    else:
        color = "red"

    if alerts:
        suitability = "Not Suitable"
        color = "red"
    
    if activity_type == "swimming" and wave_height > 1.2:
        suitability = "Not Suitable"
        color = "red"
    elif activity_type == "boating" and (wind_speed > 12 or visibility < 1000):
        suitability = "Not Suitable"
        color = "red"
    elif activity_type == "fishing" and wind_speed > 18:
        suitability = "Not Suitable"
        color = "red"
    
    
    return suitability, color

def predict_future_suitability(hourly_dataframe, marine_dataframe):
    hourly_dataframe['Date'] = hourly_dataframe['Time'].dt.date
    marine_dataframe['Date'] = marine_dataframe['Time'].dt.date

    daily_weather = hourly_dataframe.groupby('Date').agg({
        'wind_speed_10m': 'mean',
        'temperature_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'apparent_temperature': 'mean',
        'precipitation_probability': 'max',
        'pressure_msl': 'mean',
        'cloud_cover': 'mean',
        'wind_gusts_10m': 'max',
        'visibility': 'min'
    }).reset_index()

    daily_marine = marine_dataframe.groupby('Date').agg({
        'Wave Height': 'mean',
        'Wind Wave Height': 'mean',
        'Swell Wave Height': 'mean',
        'Ocean Current Velocity': 'mean'
    }).reset_index()

    daily_data = pd.merge(daily_weather, daily_marine, on='Date')

    suitability_forecast = []
    for _, row in daily_data.iterrows():
        wind_speed = row['wind_speed_10m']
        wave_height = row['Wave Height']
        wind_gusts = row['wind_gusts_10m']
        visibility = row['visibility']
        
        current_weather = {
            'temperature_2m': row['temperature_2m'],
            'relative_humidity_2m': row['relative_humidity_2m'],
            'apparent_temperature': row['apparent_temperature'],
            'precipitation_probability': row['precipitation_probability'],
            'pressure_msl': row['pressure_msl'],
            'cloud_cover': row['cloud_cover']
        }
        
        current_marine = {
            'Wind Wave Height': row['Wind Wave Height'],
            'Swell Wave Height': row['Swell Wave Height'],
            'Ocean Current Velocity': row['Ocean Current Velocity']
        }
        
        suitability, color = determine_suitability(wind_speed, wave_height, wind_gusts, visibility, [], "swimming", current_weather, current_marine)
        suitability_forecast.append({
            'Date': row['Date'],
            'Suitability': suitability,
            'Color': color,
            'Wind Speed': wind_speed,
            'Wave Height': wave_height,
            'Temperature': row['temperature_2m'],
            'Precipitation Probability': row['precipitation_probability']
        })
    
    return pd.DataFrame(suitability_forecast)


def create_map(location_data, suitability, weather_data):
    fig = go.Figure(go.Scattermapbox(
        lat=[location_data['latitude']],
        lon=[location_data['longitude']],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=16,
            color=suitability[1],
            opacity=0.7
        ),
        text=[f"Suitability: {suitability[0]}<br>Temperature: {weather_data['temperature']}°C<br>Wind Speed: {weather_data['wind_speed']} m/s"],
        hoverinfo='text'
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=go.layout.mapbox.Center(
                lat=location_data['latitude'],
                lon=location_data['longitude']
            ),
            zoom=10
        ),
        showlegend=False
    )

    return fig


def add_marine_warning_overlay(fig, warning_zones):
    for zone in warning_zones:
        fig.add_trace(go.Scattermapbox(
            lat=zone['coordinates']['lat'],
            lon=zone['coordinates']['lon'],
            mode='lines',
            line=dict(width=2, color='red'),
            name=zone['warning_type']
        ))
    return fig

st.title("Coastal Tourism Suitability App")

if 'location_data' not in st.session_state:
    st.session_state['location_data'] = None
if 'weather_data' not in st.session_state:
    st.session_state['weather_data'] = None
if 'marine_data' not in st.session_state:
    st.session_state['marine_data'] = None
if 'show_details' not in st.session_state:
    st.session_state.show_details = False
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False

if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False



st.sidebar.header("Search Location")
search_options = indian_beaches + ["Other (type custom location)"]
selected_option = st.sidebar.selectbox("Select or type a location", search_options)

if selected_option == "Other (type custom location)":
    place_name = st.sidebar.text_input("Enter custom location")
else:
    place_name = selected_option

activity_type = st.sidebar.selectbox("Select activity", ["swimming", "boating", "fishing"])

if st.sidebar.button("Fetch Weather Data"):
    if place_name and place_name != "Other (type custom location)":
        with st.spinner("Geocoding the place name..."):
            latitude, longitude, address = geocode_place(place_name)
        
        if latitude is None or longitude is None:
            st.error("Could not find the location. Please try a different place name.")
        else:
            st.session_state['location_data'] = {
                "latitude": latitude,
                "longitude": longitude,
                "address": address
            }
            
            with st.spinner("Fetching weather data..."):
                st.session_state['weather_data'] = fetch_weather_data(latitude, longitude)
                st.session_state['marine_data'] = fetch_marine_data(latitude, longitude)
    else:
        st.error("Please select a beach or enter a custom location.")

if (st.session_state['location_data'] and 
    st.session_state['weather_data'] is not None and 
    st.session_state['marine_data'] is not None and 
    not st.session_state['marine_data'].empty):
    
    location_data = st.session_state['location_data']
    weather_data = st.session_state['weather_data']
    marine_data = st.session_state['marine_data']
    
    st.subheader("Location Information")
    st.write(f"**Address:** {location_data['address']}")
    st.write(f"**Coordinates:** {location_data['latitude']}°N, {location_data['longitude']}°E")

    current_hour = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    #india_tz = pytz.timezone('Asia/Kolkata')
    #current_hour = datetime.now(india_tz).replace(minute=0, second=0, microsecond=0)
        
    hourly_dataframe = pd.DataFrame(data=weather_data['hourly'])
    hourly_dataframe['Time'] = pd.to_datetime(hourly_dataframe['time'])
    current_weather = hourly_dataframe[hourly_dataframe['Time'] == current_hour]
    
    marine_dataframe = pd.DataFrame(data=marine_data)
    marine_dataframe['Time'] = pd.to_datetime(marine_dataframe['Time'])
    current_marine = marine_dataframe[marine_dataframe['Time'] == current_hour]
    
    if not current_weather.empty and not current_marine.empty:
        current_weather = current_weather.iloc[0]
        current_marine = current_marine.iloc[0]
        
        wind_speed = current_weather.get('wind_speed_10m')
        wave_height = current_marine.get('Wave Height')
        wind_gusts = current_weather.get('wind_gusts_10m')
        visibility = current_weather.get('visibility')
        
        wind_speed = float(wind_speed) if wind_speed is not None else 0
        wave_height = float(wave_height) if wave_height is not None else 0
        wind_gusts = float(wind_gusts) if wind_gusts is not None else 0
        visibility = float(visibility) if visibility is not None else 0
        
        ocean_alerts = fetch_ocean_alerts()
        nearby_alerts = []
        if ocean_alerts:
            for alert in ocean_alerts:
                distance = haversine_distance(
                    location_data['latitude'], location_data['longitude'],
                    float(alert['LATITUDE']), float(alert['LONGITUDE'])
                )
                if distance <= 1000:
                    nearby_alerts.append((alert, distance))
            
            nearby_alerts.sort(key=lambda x: x[1])
        
        suitability, color = determine_suitability(wind_speed, wave_height, wind_gusts, visibility, nearby_alerts, activity_type, current_weather, current_marine)


        st.subheader("Geospatial Map Visualization")
        fig = create_map(location_data, (suitability, color), {'temperature': current_weather['temperature_2m'], 'wind_speed': wind_speed})
        
        
        st.plotly_chart(fig)

        st.subheader("Nearby Ocean Alerts")
        if nearby_alerts:
            for alert, distance in nearby_alerts:
                with st.expander(f"{alert['REGIONNAME']} - Magnitude {alert['MAGNITUDE']} (Distance: {distance:.2f} km)"):
                    st.write(f"Event ID: {alert['EVID']}")
                    st.write(f"Origin Time: {alert['ORIGINTIME']}")
                    st.write(f"Longitude: {alert['LONGITUDE']}")
                    st.write(f"Latitude: {alert['LATITUDE']}")
                    st.write(f"Depth: {alert['DEPTH']} km")
                    st.write(f"[More details]({alert['detail']})")
        else:
            st.write("No recent ocean alerts within 1000 km of the searched location.")
        
        st.subheader(f"Tourism Suitability for {activity_type.capitalize()}")
        st.markdown(f"<h3 style='color: {color};'>{suitability}</h3>", unsafe_allow_html=True)
        
        if nearby_alerts:
            st.warning("Caution: There are marine alerts in your vicinity. Please check the 'Nearby Ocean Alerts' section for more details.")
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Current Weather Data ({current_hour.strftime('%I %p')})")
            st.write(f"Temperature (2m): {current_weather['temperature_2m']}°C")
            st.write(f"Wind Speed (10m): {wind_speed} m/s")
            st.write(f"Wind Gusts (10m): {wind_gusts} m/s")

        with col2:
            st.subheader(f"Current Marine Data ({current_hour.strftime('%I %p')})")
            st.write(f"Wave Height: {wave_height} m")
            st.write(f"Ocean Current Velocity: {current_marine['Ocean Current Velocity']} m/s")
            st.write(f"Visibility: {visibility} m")
        
        if st.button("Toggle Details"):
           st.session_state.show_details = not st.session_state.show_details

           button_text = "Hide Details" if st.session_state.show_details else "Show Details"
           st.write(f"Details are currently {'shown' if st.session_state.show_details else 'hidden'}")


        if st.session_state.show_details:
            st.subheader("Detailed Weather Data")
            st.write(f"Relative Humidity (2m): {current_weather['relative_humidity_2m']}%")
            st.write(f"Apparent Temperature: {current_weather['apparent_temperature']}°C")
            st.write(f"Precipitation Probability: {current_weather['precipitation_probability']}%")
            st.write(f"Pressure (MSL): {current_weather['pressure_msl']} hPa")
            st.write(f"Cloud Cover: {current_weather['cloud_cover']}%")
            st.write(f"Wind Direction (10m): {current_weather['wind_direction_10m']}°")

            st.subheader("Detailed Marine Data")
            st.write(f"Wave Direction: {current_marine['Wave Direction']}°")
            st.write(f"Wind Wave Height: {current_marine['Wind Wave Height']} m")
            st.write(f"Swell Wave Height: {current_marine['Swell Wave Height']} m")
        
        st.subheader("5-Day Suitability Prediction")
        suitability_forecast = predict_future_suitability(hourly_dataframe, marine_dataframe)
        st.dataframe(suitability_forecast)
        
        # Visualize 5-day suitability forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(suitability_forecast['Date'], suitability_forecast['Wind Speed'], label='Wind Speed (m/s)')
        ax.plot(suitability_forecast['Date'], suitability_forecast['Wave Height'], label='Wave Height (m)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Add color-coded background for suitability
        for i in range(len(suitability_forecast) - 1):
            ax.axvspan(suitability_forecast['Date'][i], suitability_forecast['Date'][i+1], 
                       facecolor=suitability_forecast['Color'][i], alpha=0.3)
        
        st.pyplot(fig)
       
   
