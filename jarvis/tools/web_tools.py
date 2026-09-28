"""Small web lookups that need no API keys: clock, weather, quick facts."""

from __future__ import annotations

import re
import time
from typing import Any

from . import ToolError, registry

_UA = {"User-Agent": "Mozilla/5.0 (compatible; JARVIS/1.0)"}

# UK, Irish, and Canadian postcodes, so an unsupported lookup gets a useful
# message instead of "no such place".
_POSTCODE = re.compile(
    r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$"  # UK: SW1A 1AA, M1 1AE
    r"|^\d[A-Z]\d[A-Z]\s*\d[A-Z]\d$"  # Ireland: D02 AF30
    r"|^[A-Z]\d[A-Z]\s*\d[A-Z]\d$",  # Canada: K4P 1A1
    re.IGNORECASE,
)


def _get(url: str, timeout: float = 10.0, **params: Any) -> Any:
    import requests

    try:
        resp = requests.get(url, headers=_UA, params=params or None, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ToolError(f"request to {url} failed: {exc}") from exc
    return resp


def _get_json(url: str, timeout: float = 10.0, **params: Any) -> Any:
    """Fetch JSON, refusing anything that is really an HTML page.

    Several free endpoints answer a plain request with their website when they
    are unhappy, and a lenient parser will happily hand the model a page of CSS
    as if it were a result. Checking the content type turns that into a clear
    error instead of a confident wrong answer.
    """
    resp = _get(url, timeout=timeout, **params)
    ctype = (resp.headers.get("Content-Type") or "").lower()
    body = resp.text.lstrip()[:200].lower()
    if "json" not in ctype and not body.startswith(("{", "[")):
        raise ToolError(
            f"{url} returned a web page instead of data (content-type "
            f"{ctype or 'unknown'}); the service is probably rate limiting"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise ToolError(f"{url} returned malformed JSON: {exc}") from exc


@registry.add(
    "what_time_is_it",
    "Get the current local date and time.",
    {"type": "object", "properties": {}},
    tags=("web",),
)
def what_time_is_it() -> dict[str, Any]:
    now = time.localtime()
    return {
        "time": time.strftime("%I:%M %p", now).lstrip("0"),
        "date": time.strftime("%A, %d %B %Y", now),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S", now),
        "timezone": time.strftime("%Z"),
    }


# WMO weather interpretation codes, as Open-Meteo reports them. Only the ones
# worth naming in a spoken sentence are listed; the rest fall back to a generic
# description so an unknown code never reaches the user as a bare number.
_WMO = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def _conditions(code: Any) -> str:
    try:
        return _WMO.get(int(code), "mixed conditions")
    except (TypeError, ValueError):
        return "mixed conditions"


@registry.add(
    "get_weather",
    "Current conditions and a day's forecast for a city or town, by name.",
    {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or town name, e.g. 'Seattle' or 'Berlin'. "
                "Postcodes are not supported; use the town name.",
            },
        },
        "required": ["location"],
    },
    tags=("web",),
)
def get_weather(location: str) -> dict[str, Any]:
    """Weather via Open-Meteo.

    This used to scrape wttr.in, which now answers every request, including
    ``?format=3``, with its HTML landing page. The old parser did not notice
    and returned a "summary" made of CSS, so the model reported the data as
    missing. Open-Meteo is a real JSON API and needs no key.
    """
    query = (location or "").strip()
    if not query:
        raise ToolError("a location is required, e.g. 'Seattle'")

    # Geocoding first: Open-Meteo forecasts by coordinates, not by place name.
    geo = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        timeout=15.0,
        name=query,
        count=1,
        language="en",
        format="json",
    )
    results = geo.get("results") or []
    if not results:
        # The geocoder is name-only, so a postcode fails here in a way that
        # looks like the place does not exist. Say what is actually wrong.
        if _POSTCODE.match(query):
            raise ToolError(
                f"the weather service cannot look up postcodes like {query!r}; "
                "ask for the town or city name instead"
            )
        raise ToolError(f"could not find a place called {query!r}")
    place = results[0]

    data = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        timeout=15.0,
        latitude=place["latitude"],
        longitude=place["longitude"],
        current="temperature_2m,apparent_temperature,relative_humidity_2m,"
        "weather_code,wind_speed_10m",
        daily="temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        forecast_days=1,
        timezone="auto",
    )

    now = data.get("current") or {}
    day = data.get("daily") or {}

    def num(seq: Any, index: int = 0) -> float | None:
        try:
            return seq[index]
        except (IndexError, TypeError, KeyError):
            return None

    temp = now.get("temperature_2m")
    feels = now.get("apparent_temperature")
    lo = num(day.get("temperature_2m_min"))
    hi = num(day.get("temperature_2m_max"))
    rain = num(day.get("precipitation_probability_max"))
    desc = _conditions(now.get("weather_code"))

    bits = [f"{desc}, {temp:g}C" if isinstance(temp, (int, float)) else desc]
    if isinstance(feels, (int, float)) and isinstance(temp, (int, float)):
        bits.append(f"feels like {feels:g}C")
    if lo is not None and hi is not None:
        bits.append(f"today {lo:g} to {hi:g}C")
    if rain is not None:
        bits.append(f"{rain:g}% chance of rain")
    if now.get("relative_humidity_2m") is not None:
        bits.append(f"humidity {now['relative_humidity_2m']:g}%")
    if now.get("wind_speed_10m") is not None:
        bits.append(f"wind {now['wind_speed_10m']:g} km/h")

    return {
        "location": ", ".join(
            p for p in (place.get("name"), place.get("admin1"), place.get("country_code")) if p
        ),
        "conditions": desc,
        "temperature_c": temp,
        "feels_like_c": feels,
        "humidity_pct": now.get("relative_humidity_2m"),
        "wind_kph": now.get("wind_speed_10m"),
        "today_min_c": lo,
        "today_max_c": hi,
        "rain_chance_pct": rain,
        "summary": ", ".join(bits),
    }


@registry.add(
    "quick_lookup",
    "Look up a fact, definition, or current headline. Returns a short summary, "
    "not a full web page. Good for 'who is', 'what is', 'how tall is'.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The thing to look up."}
        },
        "required": ["query"],
    },
    tags=("web",),
)
def quick_lookup(query: str) -> dict[str, Any]:
    data = _get_json("https://api.duckduckgo.com/", query=query, format="json")
    abstract = (data.get("AbstractText") or "").strip()
    answer = (data.get("Answer") or "").strip()
    related = [
        t.get("Text") for t in data.get("RelatedTopics", [])[:6] if t.get("Text")
    ]
    if not any([abstract, answer, related]):
        raise ToolError(
            f"nothing found for {query!r}. "
            "This lookup only covers encyclopedic facts; use run_powershell with "
            "Invoke-WebRequest if you need a real web search."
        )
    return {
        "query": query,
        "answer": answer or None,
        "abstract": abstract[:900] or None,
        "related": related,
    }


@registry.add(
    "convert_units",
    "Convert between common units of length, mass, temperature, volume, and speed.",
    {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string", "description": "e.g. 'celsius', 'kg', 'miles'"},
            "to_unit": {"type": "string", "description": "e.g. 'fahrenheit', 'lbs', 'km'"},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
    tags=("web",),
)
def convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    units: dict[str, dict[str, float]] = {
        "length": {
            "m": 1.0, "meter": 1.0, "meters": 1.0, "km": 1000.0, "kilometer": 1000.0,
            "cm": 0.01, "mm": 0.001, "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
            "yd": 0.9144, "yard": 0.9144, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
            "in": 0.0254, "inch": 0.0254, "inches": 0.0254, "nmi": 1852.0,
        },
        "mass": {
            "kg": 1.0, "kilogram": 1.0, "g": 0.001, "gram": 0.001, "t": 1000.0,
            "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
            "oz": 0.0283495, "ounce": 0.0283495, "stone": 6.35029,
        },
        "volume": {
            "l": 1.0, "liter": 1.0, "litre": 1.0, "ml": 0.001, "gal": 3.78541,
            "gallon": 3.78541, "pt": 0.473176, "pint": 0.473176, "cup": 0.236588,
            "floz": 0.0295735, "tbsp": 0.0147868, "tsp": 0.00492892,
        },
        "speed": {
            "mps": 1.0, "kph": 0.277778, "kmh": 0.277778, "mph": 0.44704,
            "knot": 0.514444, "knots": 0.514444,
        },
    }
    source, target = from_unit.strip().lower(), to_unit.strip().lower()
    value = float(value)

    for table in units.values():
        if source in table and target in table:
            result = value * table[source] / table[target]
            return {
                "input": f"{value} {from_unit}",
                "result": round(result, 6),
                "output": f"{round(result, 4)} {to_unit}",
            }

    if {source, target} <= {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}:
        celsius = {"c": value, "celsius": value, "f": (value - 32) * 5 / 9,
                   "fahrenheit": (value - 32) * 5 / 9, "k": value - 273.15,
                   "kelvin": value - 273.15}[source]
        result = {
            "c": celsius, "celsius": celsius, "f": celsius * 9 / 5 + 32,
            "fahrenheit": celsius * 9 / 5 + 32, "k": celsius + 273.15,
            "kelvin": celsius + 273.15,
        }[target]
        return {
            "input": f"{value} {from_unit}",
            "result": round(result, 4),
            "output": f"{round(result, 4)} {to_unit}",
        }

    raise ToolError(
        f"cannot convert {from_unit!r} to {to_unit!r}. "
        "Supported groups: length, mass, volume, speed, and celsius/fahrenheit/kelvin."
    )
