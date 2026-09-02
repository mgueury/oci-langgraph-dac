"""Tools shared by the weather clothing agent examples."""

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.tools import tool
from pydantic import BaseModel
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


HR_FAQ = {
    "annual leave": "Employees receive 25 annual-leave days per calendar year.",
    "request deadline": "Submit leave requests at least five working days in advance when possible.",
    "approval": "Holiday bookings require manager approval before the leave is final.",
    "sick leave": "Report sick leave to your manager before the start of your working day.",
}
HOLIDAY_FILE = Path(__file__).with_name("holiday.json")
DEFAULT_TIMEZONE = "Europe/Brussels"
ANNUAL_HOLIDAY_ALLOWANCE = 25


class HolidayPeriod(BaseModel):
    """A normalized, inclusive holiday period returned by the date resolver."""

    start_date: date | None = None
    end_date: date | None = None
    interpretation: str


tools_model: Any | None = None
pending_holiday: HolidayPeriod | None = None


# -- get_current_weather -----------------------------------------------------
@tool
def get_current_weather(city: str) -> str:
    """Get current weather in a city. Use City,CountryCode when helpful."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return "OPENWEATHER_API_KEY is not configured."
    url = "https://api.openweathermap.org/data/2.5/weather?" + urlencode(
        {"q": city, "appid": key, "units": "metric"}
    )
    try:
        with urlopen(url, timeout=10) as response:
            data = json.load(response)
        if int(data.get("cod", 200)) != 200:
            return f"Weather lookup failed: {data.get('message', 'unknown location')}"
        weather, main = data["weather"][0], data["main"]
        return json.dumps({
            "location": f"{data['name']}, {data.get('sys', {}).get('country', '')}".rstrip(", "),
            "temperature_c": main["temp"],
            "feels_like_c": main["feels_like"],
            "condition": weather.get("description", weather["main"]),
            "precipitation": "rain" if data.get("rain") else "snow" if data.get("snow") else "none reported",
            "wind_m_s": data.get("wind", {}).get("speed", 0),
            "humidity_percent": main["humidity"],
        })
    except HTTPError as error:
        return f"Weather lookup failed (HTTP {error.code})."
    except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        return "Weather lookup failed. Try a more specific city or try again later."


# -- get_wikipedia_page ------------------------------------------------------
@tool
def get_wikipedia_page(title: str) -> str:
    """Get the plain-text content of an English Wikipedia page by its title."""
    url = "https://en.wikipedia.org/w/api.php?" + urlencode({
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exchars": 12000,
        "titles": title,
        "format": "json",
    })
    request = Request(
        url,
        headers={"User-Agent": "oci-langgraph-dac/1.0 (https://github.com/mgueury/oci-langgraph-dac)"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            pages = json.load(response)["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            return f"No English Wikipedia page was found for '{title}'."
        return json.dumps({"title": page["title"], "extract": page.get("extract", "")})
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError):
        return "Wikipedia could not be reached. Please try again later or use a different title."


# -- search_hr_faq -----------------------------------------------------------
@tool
def search_hr_faq(question: str) -> str:
    """Answer an HR question by matching it against the complete configured HR FAQ."""
    if not tools_model:
        return "The shared tools model is not configured."
    prompt = (
        "Answer the user's HR question using only the FAQ below. If it is not covered, "
        "say exactly: 'No matching HR FAQ was found. Ask HR for clarification.' Do not "
        "invent policy.\n\nHR FAQ:\n"
        f"{json.dumps(HR_FAQ, indent=2)}"
    )
    try:
        response = tools_model.invoke([("system", prompt), ("human", question)])
        return str(getattr(response, "content", response))
    except Exception:
        return "The HR FAQ could not be consulted. Please try again."


# -- tools_model_init --------------------------------------------------------
def tools_model_init(model: Any) -> None:
    """Initialize every LLM-backed shared tool from one chat model instance."""
    global tools_model
    tools_model = model


# -- configured_timezone_name ------------------------------------------------
def configured_timezone_name() -> str:
    return os.getenv("TIMEZONE", DEFAULT_TIMEZONE)


# -- current_local_date ------------------------------------------------------
def current_local_date() -> date:
    timezone_name = configured_timezone_name()
    try:
        return datetime.now(ZoneInfo(timezone_name)).date()
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"TIMEZONE must be a valid IANA timezone, not '{timezone_name}'.") from error


# -- resolve_holiday_period --------------------------------------------------
def resolve_holiday_period(request: str, today: date | None = None) -> HolidayPeriod:
    """Use the configured LLM to normalize a holiday request to validated ISO dates."""
    if not tools_model:
        raise ValueError("The shared tools model is not configured.")
    today = today or current_local_date()
    prompt = (
        "Normalize the holiday request into an inclusive date range. Today is "
        f"{today.isoformat()} in {configured_timezone_name()}. Interpret flexible English dates such "
        "as tomorrow, next Friday, next week, and explicit date ranges. Return exact "
        "ISO-8601 dates. If the request is ambiguous or not a date period, leave both "
        "dates empty and explain why in interpretation."
    )
    try:
        structured_model = tools_model.with_structured_output(HolidayPeriod, method="function_calling")
        result = structured_model.invoke([("system", prompt), ("human", request)])        
        period = result if isinstance(result, HolidayPeriod) else HolidayPeriod.model_validate(result)
    except Exception as error:
        raise ValueError("I could not understand that holiday period. Please use clearer dates.") from error

    if not period.start_date or not period.end_date:
        raise ValueError(f"I could not determine exact dates: {period.interpretation}")
    if period.end_date < period.start_date:
        raise ValueError("The end date is before the start date. Please try again.")
    if period.start_date < today:
        raise ValueError("Holiday dates cannot be in the past. Please try again.")
    return period


# -- holiday_days_in_year ----------------------------------------------------
def holiday_days_in_year(start: date, end: date, year: int) -> int:
    """Count weekdays in an inclusive holiday period that fall within one calendar year."""
    first = max(start, date(year, 1, 1))
    last = min(end, date(year, 12, 31))
    return sum(
        day.weekday() < 5
        for offset in range((last - first).days + 1)
        for day in [first + timedelta(days=offset)]
    ) if first <= last else 0


# -- load_holidays -----------------------------------------------------------
def load_holidays() -> list[dict[str, Any]]:
    """Load every booking, accepting the former single-booking file format."""
    if not HOLIDAY_FILE.exists():
        return []
    data = json.loads(HOLIDAY_FILE.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("holidays"), list):
        holidays = data["holidays"]
    elif isinstance(data, dict) and "start_date" in data and "end_date" in data:
        holidays = [data]
    else:
        raise ValueError("holiday.json does not contain holiday bookings")
    if not all(isinstance(holiday, dict) for holiday in holidays):
        raise ValueError("holiday.json contains an invalid holiday booking")
    return holidays


# -- propose_holiday_booking -------------------------------------------------
@tool
def propose_holiday_booking(request: str) -> str:
    """Propose a holiday from a natural-language request; confirmation is required to book it."""
    global pending_holiday
    try:
        pending_holiday = resolve_holiday_period(request)
    except ValueError as error:
        return f"Holiday proposal failed: {error}"
    return (
        f"Proposed holiday from {pending_holiday.start_date} to {pending_holiday.end_date}. "
        f"Interpretation: {pending_holiday.interpretation}. Reply 'confirm' to book it."
    )


# -- confirm_holiday_booking -------------------------------------------------
@tool
def confirm_holiday_booking() -> str:
    """Confirm the in-memory holiday proposal and write it to holiday.json."""
    global pending_holiday
    if not pending_holiday:
        return "No holiday proposal is waiting for confirmation. Ask to book a holiday first."

    holiday = {
        "status": "booked_pending_approval",
        "start_date": pending_holiday.start_date.isoformat(),
        "end_date": pending_holiday.end_date.isoformat(),
        "holiday_days": holiday_days_in_year(
            pending_holiday.start_date, pending_holiday.end_date, pending_holiday.start_date.year
        ),
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        holidays = load_holidays()
        holidays.append(holiday)
        HOLIDAY_FILE.write_text(json.dumps({"holidays": holidays}, indent=2) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "Holiday confirmation failed because holiday.json could not be written."
    pending_holiday = None
    return f"Holiday booked from {holiday['start_date']} to {holiday['end_date']} (pending approval)."


# -- get_current_holiday -----------------------------------------------------
@tool
def get_current_holiday() -> str:
    """Read all currently stored holiday bookings from holiday.json."""
    try:
        holidays = load_holidays()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "The current holiday could not be read."
    if not holidays:
        return "No holiday is currently booked."
    return json.dumps({"holidays": holidays}, indent=2)


# -- get_holiday_balance -----------------------------------------------------
@tool
def get_holiday_balance() -> str:
    """Show the remaining balance from the 25 weekday annual-holiday allowance."""
    today = current_local_date()
    booked_days = 0
    try:
        booked_days = sum(
            holiday_days_in_year(
                date.fromisoformat(holiday["start_date"]),
                date.fromisoformat(holiday["end_date"]),
                today.year,
            )
            for holiday in load_holidays()
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "Holiday balance is unavailable because holiday.json is invalid."
    return json.dumps({
        "year": today.year,
        "annual_allowance_days": ANNUAL_HOLIDAY_ALLOWANCE,
        "booked_weekdays": booked_days,
        "remaining_days": ANNUAL_HOLIDAY_ALLOWANCE - booked_days,
    })
