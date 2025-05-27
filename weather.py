"""
Weather module for LangGraph 101 project.

This module provides weather information functionality.
"""

import os
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class WeatherError(Exception):
    """Exception raised for weather API errors."""
    pass


def get_weather(location: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Get current weather for a location.

    Args:
        location: City name or location
        api_key: OpenWeatherMap API key (optional, will use env var if not provided)

    Returns:
        Dictionary with weather information

    Raises:
        WeatherError: If there's an error fetching the weather
    """
    # Get API key from environment if not provided
    if not api_key:
        load_dotenv(encoding='utf-16-le')
        api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise WeatherError("API key not found. Set OPENWEATHER_API_KEY in .env file.")

    # Base URL for OpenWeatherMap API
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    # Parameters for the API request
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric",  # Use metric units (Celsius)
        "lang": "pt_br"     # Portuguese language
    }

    try:
        # Make the API request
        response = requests.get(base_url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Extract relevant information
            weather_info = {
                "location": f"{data['name']}, {data.get('sys', {}).get('country', '')}",
                "temperature": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "description": data.get("weather", [{}])[0].get("description", ""),
                "icon": data.get("weather", [{}])[0].get("icon", ""),
                "timestamp": data.get("dt")
            }

            return weather_info
        else:
            # Handle API errors
            error_data = response.json()
            error_message = error_data.get("message", "Unknown error")
            raise WeatherError(f"API Error: {error_message} (Code: {response.status_code})")

    except requests.RequestException as e:
        raise WeatherError(f"Request failed: {str(e)}")
    except ValueError as e:
        raise WeatherError(f"Invalid response: {str(e)}")
    except Exception as e:
        raise WeatherError(f"Unexpected error: {str(e)}")


def format_weather_response(weather_data: Dict[str, Any]) -> str:
    """Format weather data into a human-readable string.

    Args:
        weather_data: Dictionary with weather information

    Returns:
        Formatted weather information as a string
    """
    return f"""
🌡️ Clima atual para {weather_data['location']}:

🌡️ Temperatura: {weather_data['temperature']}°C
🤔 Sensação térmica: {weather_data['feels_like']}°C
💧 Umidade: {weather_data['humidity']}%
💨 Velocidade do vento: {weather_data['wind_speed']} m/s
🔍 Condições: {weather_data['description']}
""".strip()


def get_mock_weather(location: str) -> Dict[str, Any]:
    """Get mock weather data for testing without an API key.

    Args:
        location: City name or location

    Returns:
        Dictionary with mock weather information
    """
    return {
        "location": f"{location}, BR",
        "temperature": 25.5,
        "feels_like": 26.2,
        "humidity": 65,
        "pressure": 1012,
        "wind_speed": 3.5,
        "description": "céu limpo",
        "icon": "01d",
        "timestamp": 1621345678
    }
