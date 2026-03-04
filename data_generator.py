"""
data_generator.py
Generates synthetic railway dataset for RailSmart Planner.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ROUTES = [
    {"route_id": "R01", "source": "Mumbai",    "destination": "Delhi",     "distance_km": 1385},
    {"route_id": "R02", "source": "Chennai",   "destination": "Bangalore", "distance_km": 346},
    {"route_id": "R03", "source": "Kolkata",   "destination": "Patna",     "distance_km": 531},
    {"route_id": "R04", "source": "Delhi",     "destination": "Jaipur",    "distance_km": 281},
    {"route_id": "R05", "source": "Hyderabad", "destination": "Pune",      "distance_km": 558},
    {"route_id": "R06", "source": "Ahmedabad", "destination": "Surat",     "distance_km": 263},
    {"route_id": "R07", "source": "Delhi",     "destination": "Amritsar",  "distance_km": 449},
    {"route_id": "R08", "source": "Mumbai",    "destination": "Goa",       "distance_km": 593},
]

TRAIN_TYPES = ["Rajdhani", "Shatabdi", "Express", "Superfast", "Intercity"]
PEAK_HOURS  = [6, 7, 8, 9, 17, 18, 19, 20]

# Each station has a realistic number of platforms
STATION_PLATFORMS = {
    "Mumbai":    6,   # CSMT — 6 major platforms
    "Delhi":     8,   # New Delhi Railway Station
    "Chennai":   5,   # Chennai Central
    "Kolkata":   4,   # Howrah
    "Hyderabad": 4,   # Kachiguda / Secunderabad
    "Ahmedabad": 5,
    "Bangalore": 4,   # KSR Bengaluru
    "Patna":     3,
    "Jaipur":    4,
    "Surat":     3,
    "Amritsar":  3,
    "Pune":      4,
    "Goa":       2,   # Madgaon
}


def generate_train_schedule(n_days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Generate n_days of train schedule + passenger data."""
    rng = np.random.default_rng(seed)
    records = []
    start = datetime(2025, 1, 1)

    for day_offset in range(n_days):
        date = start + timedelta(days=day_offset)
        is_weekend = date.weekday() >= 5
        is_holiday = day_offset in {0, 25, 51, 76, 88}   # synthetic holidays

        for i, route in enumerate(ROUTES):
            n_trains = rng.integers(2, 6)
            for t in range(n_trains):
                train_type  = rng.choice(TRAIN_TYPES)
                coaches     = int(rng.integers(8, 20))
                capacity    = coaches * 72
                hour        = int(rng.choice(range(24)))
                minute      = int(rng.integers(0, 60))
                is_peak     = hour in PEAK_HOURS

                # Demand factors
                base_occ = 0.55
                if is_peak:    base_occ += 0.18
                if is_weekend: base_occ += 0.12
                if is_holiday: base_occ += 0.22
                base_occ = min(base_occ, 0.99)

                occupancy = int(rng.binomial(capacity, base_occ))
                occupancy = min(occupancy, capacity)
                delay_min = int(rng.choice([0]*6 + list(range(5, 60)), 1)[0])

                # Platform is specific to the SOURCE station
                src_station  = route["source"]
                max_platform = STATION_PLATFORMS.get(src_station, 4)
                platform     = int(rng.integers(1, max_platform + 1))
                platform_label = f"{src_station} P{platform}"

                train_id = f"TRN-{(day_offset * 40 + i * 5 + t + 1):04d}"

                records.append({
                    "train_id":        train_id,
                    "route_id":        route["route_id"],
                    "source":          src_station,
                    "destination":     route["destination"],
                    "route":           f"{src_station} → {route['destination']}",
                    "distance_km":     route["distance_km"],
                    "train_type":      train_type,
                    "date":            date.date(),
                    "departure_hour":  hour,
                    "departure_time":  f"{hour:02d}:{minute:02d}",
                    "coaches":         coaches,
                    "capacity":        capacity,
                    "occupancy":       occupancy,
                    "occupancy_pct":   round(occupancy / capacity * 100, 1),
                    "platform":        platform,
                    "platform_label":  platform_label,
                    "station":         src_station,
                    "delay_min":       delay_min,
                    "is_weekend":      int(is_weekend),
                    "is_holiday":      int(is_holiday),
                    "is_peak_hour":    int(is_peak),
                })

    df = pd.DataFrame(records)
    return df


def generate_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-day totals for time-series forecasting."""
    daily = df.groupby("date").agg(
        total_passengers=("occupancy", "sum"),
        total_capacity=("capacity", "sum"),
        avg_occupancy_pct=("occupancy_pct", "mean"),
        total_trains=("train_id", "count"),
        delayed_trains=("delay_min", lambda x: (x > 0).sum()),
        avg_delay=("delay_min", "mean"),
        is_weekend=("is_weekend", "max"),
        is_holiday=("is_holiday", "max"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day_of_month"] = daily["date"].dt.day
    return daily


if __name__ == "__main__":
    df = generate_train_schedule()
    df.to_csv("railway_data.csv", index=False)
    print(f"Generated {len(df):,} records across {df['date'].nunique()} days.")
    print(df[["train_id","route","platform","platform_label","station"]].head(10))