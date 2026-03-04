# 🚂 RailSmart Planner

A data-driven **Smart Railway Resource Planning System** that uses historical and operational data to optimize railway resources. Built with Python, Pandas, NumPy, scikit-learn, and Streamlit.

---

## 🌟 Features

- **📊 Demand Visualization**: Visualize passenger demand across routes and time periods
- **📈 Demand Forecasting**: 30-day forecast with confidence intervals using Random Forest ML model
- **🚄 Train Fleet Explorer**: Search, filter, and analyze train occupancy and delays
- **🏛️ Platform Management**: Station-specific platform usage, delay tracking, and load distribution
- **📋 Resource Recommendations**: Data-driven suggestions for coach allocation, scheduling, and capacity planning
- **🔍 Interactive Filters**: Filter by route, train type, occupancy range, and date range
- **⬇️ CSV Export**: Download filtered train schedule data

---

## 📁 Project Structure

```
RailSmart/
│
├── app.py                  # Main Streamlit dashboard (5 tabs)
├── data_generator.py       # Synthetic railway dataset generator
├── ml_model.py             # Random Forest demand forecasting + recommendations
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Quick Start

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone or Download the Project

```bash
# Option 1: Clone with Git
git clone <repository-url>
cd RailSmart

# Option 2: Download and extract the ZIP file
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🖥️ How to Use

1. **Overview Tab** — View KPI cards, weekly demand, hourly traffic, route-wise demand vs capacity, and high-occupancy alerts
2. **Demand Forecast Tab** — See 30-day ML forecast with confidence band, model validation, and feature importances
3. **Train Fleet Tab** — Search by Train ID or route, view occupancy distribution, delay heatmap, and full schedule table
4. **Platforms Tab** — Select a station to view platform-wise train count, delay, occupancy, and utilization
5. **Resource Recommendations Tab** — View auto-generated, data-driven resource allocation suggestions based on active filters

### Sidebar Filters

| Filter | Effect |
|---|---|
| 🛤️ Route | Filter all tabs to a specific route |
| 🚄 Train Type | Filter by Rajdhani, Shatabdi, Express, etc. |
| 📊 Occupancy % Range | Filter trains by occupancy percentage |
| 📅 Date Range | Restrict analysis to a date window |

---

## 🛠️ Technical Details

### Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn (Random Forest Regressor) |
| Dashboard | Streamlit |
| Visualizations | Plotly |

### Key Components

1. **`data_generator.py`** — Generates 90 days × 8 routes of synthetic railway data using NumPy with a fixed seed for reproducibility. Each train record includes Train ID, Route, Date, Departure Time, Coaches, Capacity, Occupancy, Platform (station-specific), Delay, and demand indicators.

2. **`ml_model.py`** — Trains a Random Forest Regressor on daily aggregated passenger data. Uses lag features (1/7/14 day), rolling averages, and calendar features. Generates a 30-day forecast with ±8% confidence interval. Also contains the recommendation engine that analyses occupancy, delays, peak hours, and weekends to produce resource suggestions.

3. **`app.py`** — Streamlit dashboard with 5 tabs. Uses `st.session_state` for tab persistence. All charts use Plotly with a consistent dark theme. Recommendations are generated live from the sidebar-filtered dataset.

### ML Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Trees | 150 |
| Max Depth | 8 |
| Train / Test Split | 80% / 20% (time-ordered, no shuffle) |
| Features | lag_1, lag_7, lag_14, rolling_7d_mean, rolling_14d_mean, day_of_week, month, is_weekend, is_holiday, total_trains |
| Target | Daily total passengers |
| Accuracy | ~90–93% (1 − MAPE) |

---

## 📦 Dataset

All data is **100% synthetic** — generated deterministically using NumPy's random generator with seed `42`. No real or confidential railway data is used.

### Dataset Fields

| Field | Description |
|---|---|
| `train_id` | Unique train identifier (e.g. TRN-0012) |
| `route` | Source → Destination (e.g. Mumbai → Delhi) |
| `date` | Departure date |
| `departure_time` | Scheduled departure time (HH:MM) |
| `train_type` | Rajdhani / Shatabdi / Express / Superfast / Intercity |
| `coaches` | Number of coaches (8–19) |
| `capacity` | Total seat capacity (coaches × 72) |
| `occupancy` | Actual passengers |
| `occupancy_pct` | Occupancy percentage |
| `platform_label` | Station-specific platform (e.g. Mumbai P3) |
| `delay_min` | Departure delay in minutes |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_holiday` | 1 if synthetic holiday |
| `is_peak_hour` | 1 if departure between 06–09 or 17–20 |

---

## ⚠️ Troubleshooting

### Common Issues

1. **"Module not found" error**
   - Run `pip install -r requirements.txt` from the project folder
   - Make sure you are in the `RailSmart/` directory

2. **App shows stale data after code changes**
   - Run `streamlit cache clear` then restart:
     ```bash
     streamlit run app.py
     ```

3. **Streamlit not launching in browser**
   - Open `http://localhost:8501` manually
   - Check if another process is using that port

4. **Charts appear empty after filtering**
   - Charts in Train Fleet tab use the full sidebar-filtered dataset, not the search box
   - Only the Full Train Schedule table responds to the search input

5. **Platforms tab redirecting to Overview**
   - Fixed via `st.session_state` — ensure you are using the latest `app.py`

---

## 🎯 Evaluation Alignment

| Criterion | Implementation |
|---|---|
| Problem Understanding | Covers all 4 goals: demand prediction, coach allocation, platform usage, scheduling |
| Innovation | Live ML forecast + automated recommendation engine based on active filters |
| Quality of Insights | 5 analytical tabs + conditional alerts + color-coded tables |
| Technical Approach | Clean 3-file architecture, cached ML pipeline, session state tab management |
| Ease of Use | Sidebar filters, search, CSV export, station selector |
| Presentation | Custom dark theme, Plotly charts, KPI metric cards |

---

## 🚀 Future Enhancements

- 🌐 **Real Dataset Integration** — Connect to Indian Railways open data APIs
- 📅 **Recurring Pattern Detection** — Auto-flag festival season demand spikes
- 📱 **Mobile PWA** — Deploy as Progressive Web App
- 🔔 **Alerts System** — Email notifications when occupancy exceeds threshold
- 🗺️ **Route Map View** — Geographic visualization of train corridors
- 📥 **CSV Upload** — Let planners upload their own schedule data

---

## 📞 Support

If you encounter any issues or have questions:

- 📧 Email: bhavanibhavya77@gmail.com
- 📱 Phone: +91 90631997036

---


*RailSmart Planner v1.0 · Built for Smart Railway Resource Planning Hackathon · Python · Pandas · NumPy · scikit-learn · Streamlit · Plotly*