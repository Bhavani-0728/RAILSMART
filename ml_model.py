"""
ml_model.py
Demand forecasting using Random Forest Regressor.
Returns predictions + feature importances + recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model      import LinearRegression
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing     import LabelEncoder


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def build_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Lag features
    df["lag_1"]  = df["total_passengers"].shift(1)
    df["lag_7"]  = df["total_passengers"].shift(7)
    df["lag_14"] = df["total_passengers"].shift(14)

    # Rolling averages
    df["rolling_7d_mean"]  = df["total_passengers"].shift(1).rolling(7).mean()
    df["rolling_14d_mean"] = df["total_passengers"].shift(1).rolling(14).mean()

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "day_of_week", "month", "day_of_month",
    "is_weekend", "is_holiday",
    "lag_1", "lag_7", "lag_14",
    "rolling_7d_mean", "rolling_14d_mean",
    "total_trains",
]

TARGET = "total_passengers"


# ── TRAIN MODEL ───────────────────────────────────────────────────────────────

def train_model(daily_df: pd.DataFrame):
    df = build_features(daily_df)
    X  = df[FEATURE_COLS]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=150, max_depth=8,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae":      round(mean_absolute_error(y_test, y_pred), 0),
        "mape":     round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2),
        "r2":       round(r2_score(y_test, y_pred), 4),
        "accuracy": round((1 - mean_absolute_percentage_error(y_test, y_pred)) * 100, 1),
    }

    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False).reset_index()
    importances.columns = ["feature", "importance"]

    return model, metrics, importances, df, X_test, y_test, y_pred


# ── FORECAST FUTURE ───────────────────────────────────────────────────────────

def forecast_next_n_days(model, daily_df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    df = build_features(daily_df)
    last_known = df.copy()

    future_records = []
    last_date = pd.to_datetime(df["date"].max())

    # Use last known values as rolling buffer
    passenger_history = list(df["total_passengers"].values)

    for i in range(1, n + 1):
        future_date = last_date + pd.Timedelta(days=i)
        dow  = future_date.dayofweek
        mon  = future_date.month
        dom  = future_date.day
        iswk = int(dow >= 5)
        ishol = 0  # simplified

        lag1  = passenger_history[-1]
        lag7  = passenger_history[-7]  if len(passenger_history) >= 7  else passenger_history[0]
        lag14 = passenger_history[-14] if len(passenger_history) >= 14 else passenger_history[0]
        roll7  = np.mean(passenger_history[-7:])
        roll14 = np.mean(passenger_history[-14:])
        n_trains = int(df["total_trains"].mean())

        row = pd.DataFrame([{
            "day_of_week": dow, "month": mon, "day_of_month": dom,
            "is_weekend": iswk, "is_holiday": ishol,
            "lag_1": lag1, "lag_7": lag7, "lag_14": lag14,
            "rolling_7d_mean": roll7, "rolling_14d_mean": roll14,
            "total_trains": n_trains,
        }])

        pred = int(model.predict(row)[0])
        passenger_history.append(pred)

        # Confidence interval: ±8%
        future_records.append({
            "date":      future_date.date(),
            "predicted": pred,
            "lower":     int(pred * 0.92),
            "upper":     int(pred * 1.08),
            "is_weekend": iswk,
            "is_holiday": ishol,
        })

    return pd.DataFrame(future_records)


# ── RECOMMENDATIONS ENGINE ────────────────────────────────────────────────────

def generate_recommendations(df: pd.DataFrame, forecast_df: pd.DataFrame) -> list:
    recs = []

    # 1. Overcrowded routes
    overcrowded = (
        df.groupby("route")["occupancy_pct"].mean()
        .sort_values(ascending=False)
    )
    top_route = overcrowded.index[0]
    top_pct   = round(overcrowded.iloc[0], 1)
    if top_pct >= 80:
        recs.append({
            "priority": "🔴 High",
            "color":    "#EF4444",
            "title":    f"Add Extra Coaches — {top_route}",
            "detail":   f"Average occupancy of {top_pct}% detected. Recommend adding 2–4 coaches on peak-hour departures.",
            "tags":     ["Overcrowding", "Coach Allocation", "Immediate"],
        })

    # 2. Underutilised routes
    underutil = overcrowded[overcrowded < 55]
    if len(underutil):
        recs.append({
            "priority": "🟢 Low",
            "color":    "#10B981",
            "title":    f"Reduce Coaches — {underutil.index[0]}",
            "detail":   f"Avg occupancy {round(underutil.iloc[0],1)}%. Removing 4–6 coaches frees rolling stock and cuts costs.",
            "tags":     ["Cost Saving", "Resource Reuse"],
        })

    # 3. Peak hour surge
    peak_occ = df[df["is_peak_hour"] == 1]["occupancy_pct"].mean()
    offpeak_occ = df[df["is_peak_hour"] == 0]["occupancy_pct"].mean()
    if peak_occ - offpeak_occ > 15:
        recs.append({
            "priority": "🔴 High",
            "color":    "#EF4444",
            "title":    "Increase Frequency During Peak Hours",
            "detail":   f"Peak hour demand is {round(peak_occ-offpeak_occ,1)}% higher than off-peak. Add 1–2 services per corridor during 07:00–09:00 and 17:00–20:00.",
            "tags":     ["Peak Planning", "Scheduling", "Frequency"],
        })

    # 4. Weekend demand
    wk_occ  = df[df["is_weekend"] == 1]["occupancy_pct"].mean()
    wkd_occ = df[df["is_weekend"] == 0]["occupancy_pct"].mean()
    if wk_occ > wkd_occ + 8:
        recs.append({
            "priority": "🟡 Medium",
            "color":    "#F59E0B",
            "title":    "Weekend Capacity Expansion Required",
            "detail":   f"Weekend occupancy averages {round(wk_occ,1)}% vs {round(wkd_occ,1)}% on weekdays. Activate reserve trainsets on Fri–Sun.",
            "tags":     ["Weekend", "Reserve Fleet", "Demand Surge"],
        })

    # 5. High-delay routes
    delay_by_route = df.groupby("route")["delay_min"].mean().sort_values(ascending=False)
    worst_delay_route = delay_by_route.index[0]
    worst_delay_val   = round(delay_by_route.iloc[0], 1)
    if worst_delay_val > 10:
        recs.append({
            "priority": "🟡 Medium",
            "color":    "#F97316",
            "title":    f"Address Chronic Delays — {worst_delay_route}",
            "detail":   f"Average delay of {worst_delay_val} min. Recommend pre-positioning rolling stock and tightening turnaround windows.",
            "tags":     ["Delay Reduction", "Operational", "On-time Performance"],
        })

    # 6. Forecast surge
    peak_forecast = forecast_df.nlargest(3, "predicted")
    surge_date = peak_forecast.iloc[0]["date"]
    surge_val  = peak_forecast.iloc[0]["predicted"]
    recs.append({
        "priority": "🔴 High",
        "color":    "#EF4444",
        "title":    f"Pre-plan Surge Capacity — {surge_date}",
        "detail":   f"ML model predicts {surge_val:,} passengers — highest in 30-day window. Activate 4–6 reserve trainsets 2 weeks in advance.",
        "tags":     ["Forecast", "Holiday Buffer", "Pre-planning"],
    })

    # 7. Platform overload
    platform_load = df.groupby("platform")["train_id"].count().sort_values(ascending=False)
    busiest_plat  = platform_load.index[0]
    quietest_plat = platform_load.index[-1]
    if platform_load.iloc[0] > platform_load.iloc[-1] * 2:
        recs.append({
            "priority": "🟡 Medium",
            "color":    "#F59E0B",
            "title":    f"Rebalance Platform Allocation (P{busiest_plat} → P{quietest_plat})",
            "detail":   f"Platform {busiest_plat} handles {platform_load.iloc[0]} trains vs {platform_load.iloc[-1]} on Platform {quietest_plat}. Redistribute 3–4 intercity services.",
            "tags":     ["Platform Planning", "Load Balancing"],
        })

    return recs