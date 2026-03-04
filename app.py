"""
app.py  —  RailSmart Planner
Smart Railway Resource Planning System
Tech Stack: Python · Pandas · NumPy · scikit-learn · Streamlit · Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator import generate_train_schedule, generate_daily_summary
from ml_model       import train_model, forecast_next_n_days, generate_recommendations

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RailSmart Planner",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&family=Space+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* Dark card style */
  .metric-card {
    background: #161D2E;
    border: 1px solid #1E2A3D;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color .2s;
  }
  .metric-card:hover { border-color: #00D4FF; }
  .metric-label { color: #64748B; font-size: 11px; font-weight: 700;
                  letter-spacing: 1px; text-transform: uppercase; }
  .metric-value { color: #E2E8F0; font-size: 30px; font-weight: 800;
                  letter-spacing: -1px; margin: 4px 0; }
  .metric-sub   { color: #64748B; font-size: 12px; }
  .metric-trend-up   { color: #10B981; font-size: 12px; }
  .metric-trend-warn { color: #F97316; font-size: 12px; }

  /* Recommendation card */
  .rec-card {
    background: #111827;
    border-left: 4px solid;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .rec-title  { color: #E2E8F0; font-weight: 700; font-size: 14px; margin-bottom: 5px; }
  .rec-detail { color: #94A3B8; font-size: 12px; line-height: 1.6; }
  .rec-tag {
    display: inline-block;
    background: #1E2A3D; color: #64748B;
    border-radius: 6px; padding: 2px 8px;
    font-size: 11px; margin: 4px 4px 0 0;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0F1623; }
  .stSelectbox label, .stSlider label, .stMultiSelect label { color: #94A3B8 !important; font-size: 12px !important; }

  /* Tab styling */
  .stTabs [data-baseweb="tab"] {
    font-weight: 600; color: #64748B;
    border-bottom: 2px solid transparent;
  }
  .stTabs [aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
  }
  div[data-testid="metric-container"] {
    background: #161D2E;
    border: 1px solid #1E2A3D;
    border-radius: 12px;
    padding: 14px;
  }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY TEMPLATE ───────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(family="DM Sans", color="#94A3B8", size=11),
    xaxis=dict(gridcolor="#1E2A3D", zeroline=False),
    yaxis=dict(gridcolor="#1E2A3D", zeroline=False),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)
COLORS = ["#00D4FF","#7C3AED","#10B981","#F59E0B","#EF4444","#F97316","#EC4899","#6366F1"]


# ── DATA LOADING (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df      = generate_train_schedule(n_days=90)
    daily   = generate_daily_summary(df)
    model, metrics, importances, feat_df, X_test, y_test, y_pred = train_model(daily)
    forecast_df = forecast_next_n_days(model, daily, n=30)
    return df, daily, model, metrics, importances, feat_df, forecast_df, y_test, y_pred


# ── HEADER ────────────────────────────────────────────────────────────────────
col_logo, col_title, col_badge = st.columns([0.05, 0.8, 0.15])
with col_logo:  st.markdown("# 🚂")
with col_title:
    st.markdown("### **RailSmart** <span style='color:#00D4FF'>Planner</span> — Smart Railway Resource Planning System",
                unsafe_allow_html=True)
with col_badge:
    st.markdown("<div style='padding-top:12px'><span style='background:#00D4FF22;color:#00D4FF;"
                "border:1px solid #00D4FF44;padding:4px 12px;border-radius:8px;"
                "font-size:11px;font-weight:700'>● LIVE SYNTHETIC DATA</span></div>",
                unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1E2A3D;margin:6px 0 18px'>", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("🚄  Generating synthetic data & training ML model..."):
    df, daily, model, metrics, importances, feat_df, forecast_df, y_test, y_pred = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Filters")
    st.markdown("---")

    all_routes = ["All"] + sorted(df["route"].unique().tolist())
    sel_route  = st.selectbox("🛤️ Route", all_routes)

    all_types  = ["All"] + sorted(df["train_type"].unique().tolist())
    sel_type   = st.selectbox("🚄 Train Type", all_types)

    st.markdown("**📊 Occupancy % Range**")
    occ_col1, occ_col2 = st.columns(2)
    with occ_col1:
        occ_min = st.number_input("Min %", min_value=0, max_value=100, value=0, step=5)
    with occ_col2:
        occ_max = st.number_input("Max %", min_value=0, max_value=100, value=100, step=5)
    occ_range = (int(occ_min), int(occ_max))

    st.markdown("---")
    st.markdown("**📅 Date Range**")
    date_min = pd.to_datetime(df["date"].min())
    date_max = pd.to_datetime(df["date"].max())
    sel_dates = st.date_input("Select range", [date_min, date_max],
                               min_value=date_min, max_value=date_max)

    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.caption("RailSmart Planner uses a **Random Forest** model trained on 90 days of "
               "synthetic data to forecast demand and recommend resource allocation.")
    st.caption(f"**Model Accuracy:** {metrics['accuracy']}% | **R²:** {metrics['r2']}")
    st.markdown("---")
    st.caption("🔬 Synthetic data only — no real railway data used.")

# ── FILTER DATA ───────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_route != "All":
    fdf = fdf[fdf["route"] == sel_route]
if sel_type != "All":
    fdf = fdf[fdf["train_type"] == sel_type]
fdf = fdf[(fdf["occupancy_pct"] >= occ_range[0]) & (fdf["occupancy_pct"] <= occ_range[1])]
if len(sel_dates) == 2:
    fdf = fdf[(pd.to_datetime(fdf["date"]) >= pd.to_datetime(sel_dates[0])) &
              (pd.to_datetime(fdf["date"]) <= pd.to_datetime(sel_dates[1]))]

# ── TABS ──────────────────────────────────────────────────────────────────────
# ── MANUAL TAB SYSTEM (preserves state across sidebar interactions) ──────────
TAB_NAMES = ["📊 Overview", "📈 Demand Forecast", "🚄 Train Fleet", "🏛️ Platforms", "📋 Resource Recommendations"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Overview"

# Inject tab bar CSS — mimics original Streamlit tab look
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 4px !important;
    transition: color 0.15s !important;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    color: #E2E8F0 !important;
    background: transparent !important;
    border-bottom: 2px solid #334155 !important;
}
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid #00D4FF !important;
    border-radius: 0 !important;
    color: #00D4FF !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 4px !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

cols = st.columns(len(TAB_NAMES))
for i, (col, name) in enumerate(zip(cols, TAB_NAMES)):
    with col:
        active = st.session_state.active_tab == name
        if st.button(name, key=f"tab_btn_{i}",
                     use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.active_tab = name
            st.rerun()

active_tab = st.session_state.active_tab
st.markdown("<hr style='border-color:#1E2A3D;margin:0 0 20px'>", unsafe_allow_html=True)




# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if active_tab == "📊 Overview":
    # KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("🚄 Total Trains",   f"{len(fdf):,}", delta=f"{fdf['train_id'].nunique():,} unique IDs")
    with k2:
        st.metric("👥 Total Passengers", f"{fdf['occupancy'].sum():,}", delta="synthetic est.")
    with k3:
        avg_occ = round(fdf["occupancy_pct"].mean(), 1)
        st.metric("📊 Avg Occupancy",  f"{avg_occ}%",
                  delta=f"{'⚠️ High' if avg_occ > 80 else '✅ Normal'}")
    with k4:
        critical = len(fdf[fdf["occupancy_pct"] >= 90])
        st.metric("🔴 Critical Trains", critical, delta="≥90% capacity", delta_color="inverse")
    with k5:
        delayed = len(fdf[fdf["delay_min"] > 0])
        st.metric("⏱️ Delayed Trains",  delayed,
                  delta=f"avg {round(fdf[fdf['delay_min']>0]['delay_min'].mean(),1)} min")
    with k6:
        st.metric("🛤️ Active Routes",   fdf["route"].nunique(), delta="corridors")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Weekly demand + Hourly pattern
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📅 Weekly Passenger Demand")
        weekly = (
            fdf.assign(week=pd.to_datetime(fdf["date"]).dt.isocalendar().week)
            .groupby("week")
            .agg(passengers=("occupancy","sum"), capacity=("capacity","sum"))
            .reset_index()
        )
        weekly["utilization"] = (weekly["passengers"] / weekly["capacity"] * 100).round(1)
        fig = go.Figure()
        fig.add_bar(x=weekly["week"], y=weekly["passengers"],
                    name="Passengers", marker_color=COLORS[0], marker_line_width=0)
        fig.add_scatter(x=weekly["week"], y=weekly["capacity"],
                        name="Capacity", mode="lines", line=dict(color=COLORS[4], dash="dash"))
        fig.update_layout(**PLOT_LAYOUT, height=280, title="Weekly Demand vs Capacity")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### ⏰ Hourly Traffic Pattern")
        hourly = (
            fdf.groupby("departure_hour")
            .agg(passengers=("occupancy","sum"), trains=("train_id","count"))
            .reset_index()
        )
        fig2 = go.Figure()
        fig2.add_scatter(x=hourly["departure_hour"], y=hourly["passengers"],
                         fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
                         line=dict(color=COLORS[0], width=2),
                         name="Passengers", mode="lines")
        fig2.add_bar(x=hourly["departure_hour"], y=hourly["trains"],
                     name="Train Count", marker_color=COLORS[1], opacity=0.5,
                     yaxis="y2")
        fig2.update_layout(**PLOT_LAYOUT, height=280,
                           title="Hourly Demand & Train Frequency",
                           yaxis2=dict(overlaying="y", side="right",
                                       gridcolor="#1E2A3D", title="Trains"))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Route demand + Type distribution
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 🛤️ Route-wise Demand vs Capacity")
        route_agg = (
            fdf.groupby("route")
            .agg(demand=("occupancy","sum"), capacity=("capacity","sum"))
            .reset_index()
            .sort_values("demand", ascending=True)
        )
        route_agg["gap"] = route_agg["capacity"] - route_agg["demand"]
        fig3 = go.Figure()
        fig3.add_bar(y=route_agg["route"], x=route_agg["demand"],
                     name="Demand", orientation="h", marker_color=COLORS[0])
        fig3.add_bar(y=route_agg["route"], x=route_agg["gap"],
                     name="Spare Capacity", orientation="h",
                     marker_color=COLORS[2], opacity=0.4)
        fig3.update_layout(**PLOT_LAYOUT, barmode="stack",
                           height=320, title="Route Demand vs Spare Capacity")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("#### 🚄 Fleet Composition")
        type_agg = fdf.groupby("train_type")["occupancy"].sum().reset_index()
        fig4 = px.pie(type_agg, names="train_type", values="occupancy",
                      color_discrete_sequence=COLORS, hole=0.5)
        fig4.update_layout(**PLOT_LAYOUT, height=320,
                           title="Passenger Share by Train Type")
        fig4.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig4, use_container_width=True)

    # Critical trains table
    st.markdown("#### 🔴 High-Occupancy Alerts")
    critical_trains = (
        fdf[fdf["occupancy_pct"] >= 85]
        [["train_id","route","train_type","departure_time","coaches",
          "occupancy_pct","delay_min","platform","date"]]
        .sort_values("occupancy_pct", ascending=False)
        .head(10)
        .rename(columns={
            "train_id":"Train ID","route":"Route","train_type":"Type",
            "departure_time":"Departure","coaches":"Coaches",
            "occupancy_pct":"Occupancy %","delay_min":"Delay (min)",
            "platform":"Platform","date":"Date",
        })
    )
    st.dataframe(
        critical_trains.style
            .background_gradient(subset=["Occupancy %"], cmap="Reds")
            .background_gradient(subset=["Delay (min)"], cmap="Oranges"),
        use_container_width=True, height=300,
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — DEMAND FORECAST
# ═══════════════════════════════════════════════════════════════════════
if active_tab == "📈 Demand Forecast":
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("🎯 Model Accuracy",  f"{metrics['accuracy']}%", delta="Random Forest")
    with m2: st.metric("📐 R² Score",         metrics["r2"], delta="on test set")
    with m3: st.metric("📉 MAE",              f"{int(metrics['mae']):,} pax", delta="mean abs. error")
    with m4: st.metric("📅 Forecast Horizon", "30 days", delta="next month")

    st.markdown("<br>", unsafe_allow_html=True)

    # 30-day forecast
    st.markdown("#### 📈 30-Day Demand Forecast with Confidence Interval")
    fig_fore = go.Figure()
    fig_fore.add_scatter(
        x=forecast_df["date"].astype(str), y=forecast_df["upper"],
        mode="lines", line=dict(width=0), name="Upper Bound", showlegend=False
    )
    fig_fore.add_scatter(
        x=forecast_df["date"].astype(str), y=forecast_df["lower"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(124,58,237,0.12)", name="Confidence Band"
    )
    fig_fore.add_scatter(
        x=forecast_df["date"].astype(str), y=forecast_df["predicted"],
        mode="lines+markers", line=dict(color=COLORS[1], width=2.5, dash="dot"),
        marker=dict(size=4), name="Forecast"
    )
    # Add actual (last 14 days from training)
    recent = daily.tail(14)
    fig_fore.add_scatter(
        x=recent["date"].astype(str), y=recent["total_passengers"],
        mode="lines+markers", line=dict(color=COLORS[0], width=2),
        marker=dict(size=5), name="Actual (Historical)"
    )
    fig_fore.update_layout(**PLOT_LAYOUT, height=350)
    st.plotly_chart(fig_fore, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        # Actual vs predicted (test set)
        st.markdown("#### 🔬 Model Validation — Actual vs Predicted")
        val_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        fig_val = go.Figure()
        fig_val.add_scatter(x=list(range(len(val_df))), y=val_df["Actual"],
                            mode="lines", name="Actual", line=dict(color=COLORS[0]))
        fig_val.add_scatter(x=list(range(len(val_df))), y=val_df["Predicted"],
                            mode="lines", name="Predicted",
                            line=dict(color=COLORS[4], dash="dash"))
        fig_val.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig_val, use_container_width=True)

    with col_b:
        # Feature importance
        st.markdown("#### 🧠 Feature Importances (ML Model)")
        fig_imp = px.bar(
            importances.head(8),
            x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=["#1E2A3D","#00D4FF"],
        )
        fig_imp.update_layout(**PLOT_LAYOUT, height=280, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    # Peak predictions table
    st.markdown("#### 🚨 Predicted High-Demand Days")
    top_forecast = forecast_df.sort_values("predicted", ascending=False).head(10).copy()
    top_forecast["surge_risk"] = top_forecast["predicted"].apply(
        lambda x: "🔴 Critical" if x > forecast_df["predicted"].quantile(0.9)
        else "🟡 High" if x > forecast_df["predicted"].quantile(0.75)
        else "🟢 Normal"
    )
    st.dataframe(
        top_forecast[["date","predicted","lower","upper","is_weekend","surge_risk"]]
        .rename(columns={
            "date":"Date","predicted":"Predicted Pax","lower":"Lower CI","upper":"Upper CI",
            "is_weekend":"Weekend","surge_risk":"Surge Risk"
        }),
        use_container_width=True, height=320,
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — TRAIN FLEET
# ═══════════════════════════════════════════════════════════════════════
if active_tab == "🚄 Train Fleet":
    st.markdown("#### 🔍 Train Fleet Explorer")

    # Search — only affects the table, NOT the charts
    search = st.text_input("Search by Train ID or Route", placeholder="e.g. TRN-0012 or Mumbai")

    # Charts always use the sidebar-filtered data (fdf), never the search term
    chart_df = fdf.copy()

    # Table uses both sidebar filters + search
    table_df = fdf.copy()
    if search:
        table_df = table_df[
            table_df["train_id"].str.contains(search, case=False) |
            table_df["route"].str.contains(search, case=False)
        ]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📊 Occupancy Distribution")
        st.caption("Showing full filtered dataset — not affected by search")
        fig_hist = px.histogram(
            chart_df, x="occupancy_pct", nbins=30,
            color_discrete_sequence=[COLORS[0]],
            labels={"occupancy_pct": "Occupancy %"},
        )
        fig_hist.add_vline(x=90, line_dash="dash", line_color=COLORS[4],
                           annotation_text="Critical (90%)")
        fig_hist.add_vline(x=70, line_dash="dash", line_color=COLORS[3],
                           annotation_text="High (70%)")
        fig_hist.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.markdown("#### 🧩 Coaches vs Occupancy")
        st.caption("Showing full filtered dataset — not affected by search")
        sample_df = chart_df.sample(min(500, len(chart_df)), random_state=1) if len(chart_df) > 0 else chart_df
        fig_scat = px.scatter(
            sample_df,
            x="coaches", y="occupancy_pct",
            color="train_type", color_discrete_sequence=COLORS,
            hover_data=["train_id", "route"],
            labels={"coaches":"Coaches", "occupancy_pct":"Occupancy %"},
        )
        fig_scat.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig_scat, use_container_width=True)

    # Delay heatmap — always uses fdf (sidebar filters only)
    st.markdown("#### ⏱️ Delay Heatmap by Route & Day of Week")
    st.caption("Showing full filtered dataset — not affected by search")
    delay_heat = (
        chart_df.assign(day=pd.to_datetime(chart_df["date"]).dt.day_name())
        .groupby(["route","day"])["delay_min"].mean().reset_index()
        .pivot(index="route", columns="day", values="delay_min")
        .reindex(columns=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        .fillna(0)
    )
    fig_heat = px.imshow(
        delay_heat, color_continuous_scale=["#111827","#F97316","#EF4444"],
        labels=dict(color="Avg Delay (min)"),
        aspect="auto",
    )
    fig_heat.update_layout(**PLOT_LAYOUT, height=320)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Full table — uses search filter
    st.markdown("#### 📋 Full Train Schedule")
    if search:
        st.caption(f"🔍 Search results for **'{search}'** — {len(table_df)} trains found")
    else:
        st.caption(f"{len(table_df):,} trains shown")

    cols_show = ["train_id","route","train_type","date","departure_time",
                 "coaches","capacity","occupancy","occupancy_pct","platform_label","delay_min"]

    if len(table_df) == 0:
        st.warning("⚠️ No trains found matching your search. Try a different Train ID or route name.")
    else:
        show = table_df[cols_show].sort_values("occupancy_pct", ascending=False)
        st.dataframe(
            show.style.background_gradient(subset=["occupancy_pct"], cmap="RdYlGn_r")
                      .background_gradient(subset=["delay_min"], cmap="Oranges"),
            use_container_width=True, height=400,
        )
        csv = show.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered Data (CSV)", csv,
                           "railway_data_filtered.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — PLATFORMS
# ═══════════════════════════════════════════════════════════════════════
if active_tab == "🏛️ Platforms":

    # Station selectbox lives here — uses session_state key so active_tab is preserved on rerun
    all_stations_tab = ["All Stations"] + sorted(fdf["station"].unique().tolist())
    if "sel_station" not in st.session_state:
        st.session_state.sel_station = "All Stations"

    sel_station_tab = st.selectbox(
        "🏙️ Select Station",
        all_stations_tab,
        key="sel_station",
    )
    station_df = fdf if sel_station_tab == "All Stations" else fdf[fdf["station"] == sel_station_tab]

    # Aggregate by station + platform
    plat_agg = (
        station_df.groupby(["station", "platform", "platform_label"])
        .agg(
            trains    =("train_id",     "count"),
            passengers=("occupancy",    "sum"),
            avg_delay =("delay_min",    "mean"),
            avg_occ   =("occupancy_pct","mean"),
        )
        .reset_index()
        .sort_values("trains", ascending=False)
    )
    plat_agg["utilization_pct"] = (plat_agg["trains"] / plat_agg["trains"].sum() * 100).round(1)
    plat_agg["avg_delay"] = plat_agg["avg_delay"].round(1)
    plat_agg["avg_occ"]   = plat_agg["avg_occ"].round(1)

    # Station-level summary
    station_summary = (
        station_df.groupby("station")
        .agg(
            total_trains  =("train_id",     "count"),
            total_pax     =("occupancy",    "sum"),
            avg_delay     =("delay_min",    "mean"),
            avg_occ       =("occupancy_pct","mean"),
            platforms_used=("platform",     "nunique"),
        )
        .reset_index()
        .sort_values("total_trains", ascending=False)
    )

    # KPIs
    p1, p2, p3, p4 = st.columns(4)
    busiest_row  = plat_agg.iloc[0]
    quietest_row = plat_agg.iloc[-1]
    with p1: st.metric("🏙️ Stations",          station_summary["station"].nunique())
    with p2: st.metric("🔥 Busiest Platform",   busiest_row["platform_label"],
                       delta=f"{busiest_row['trains']} trains")
    with p3: st.metric("😴 Quietest Platform",  quietest_row["platform_label"],
                       delta=f"{quietest_row['trains']} trains")
    with p4: st.metric("⏱️ Avg Delay",          f"{round(plat_agg['avg_delay'].mean(), 1)} min")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🚉 Train Count per Platform")
        fig_p1 = px.bar(
            plat_agg.head(20),
            x="platform_label", y="trains",
            color="station", color_discrete_sequence=COLORS,
            labels={"platform_label":"Platform","trains":"Trains","station":"Station"},
            text="trains",
        )
        fig_p1.update_layout(**PLOT_LAYOUT, height=300, xaxis_tickangle=-35)
        st.plotly_chart(fig_p1, use_container_width=True, key="plat_bar")

    with c2:
        st.markdown("#### 📊 Trains by Station")
        fig_p2 = px.pie(
            station_summary, names="station", values="total_trains",
            color_discrete_sequence=COLORS, hole=0.45,
        )
        fig_p2.update_layout(**PLOT_LAYOUT, height=300)
        st.plotly_chart(fig_p2, use_container_width=True, key="plat_pie")

    st.markdown("#### ⏱️ Avg Delay & Occupancy per Platform")
    fig_p3 = make_subplots(specs=[[{"secondary_y": True}]])
    top_plats = plat_agg.head(15)
    fig_p3.add_bar(x=top_plats["platform_label"], y=top_plats["avg_delay"],
                   name="Avg Delay (min)", marker_color=COLORS[3])
    fig_p3.add_scatter(x=top_plats["platform_label"], y=top_plats["avg_occ"],
                       mode="lines+markers", name="Avg Occupancy %",
                       line=dict(color=COLORS[0], width=2), secondary_y=True)
    fig_p3.update_layout(**PLOT_LAYOUT, height=300)
    fig_p3.update_xaxes(tickangle=-35, gridcolor="#1E2A3D")
    st.plotly_chart(fig_p3, use_container_width=True, key="plat_dual")

    st.markdown("#### 🏙️ Station-level Summary")
    st.dataframe(
        station_summary.rename(columns={
            "station":"Station","total_trains":"Total Trains",
            "total_pax":"Total Passengers","avg_delay":"Avg Delay (min)",
            "avg_occ":"Avg Occupancy %","platforms_used":"Platforms Used",
        }).round(1)
        .style.background_gradient(subset=["Avg Occupancy %"], cmap="RdYlGn_r")
              .background_gradient(subset=["Avg Delay (min)"], cmap="Oranges"),
        use_container_width=True, key="station_summary_table",
    )

    st.markdown("#### 📋 Platform Detail")
    st.dataframe(
        plat_agg[["platform_label","station","trains","passengers","avg_delay","avg_occ","utilization_pct"]]
        .rename(columns={
            "platform_label":"Platform","station":"Station","trains":"Trains",
            "passengers":"Total Pax","avg_delay":"Avg Delay (min)",
            "avg_occ":"Avg Occupancy %","utilization_pct":"Share (%)",
        })
        .style.background_gradient(subset=["Avg Occupancy %"], cmap="RdYlGn_r")
              .background_gradient(subset=["Avg Delay (min)"], cmap="Oranges"),
        use_container_width=True, key="plat_detail_table",
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 5 — AI RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════
if active_tab == "📋 Resource Recommendations":
    # Generate recommendations live from sidebar-filtered data
    recs = generate_recommendations(fdf, forecast_df)

    r1, r2, r3, r4 = st.columns(4)
    with r1: st.metric("📋 Planning Insights",   len(recs))
    with r2: st.metric("🔴 Immediate Action",    sum(1 for r in recs if "High" in r["priority"]))
    with r3: st.metric("🟡 Plan Within 48h",  sum(1 for r in recs if "Medium" in r["priority"]))
    with r4: st.metric("🟢 Optimize When Possible",     sum(1 for r in recs if "Low" in r["priority"]))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Resource Allocation Suggestions")

    col_recs, col_health = st.columns([2, 1])

    with col_recs:
        for rec in recs:
            tags_html = "".join(f'<span class="rec-tag">{t}</span>' for t in rec["tags"])
            st.markdown(f"""
            <div class="rec-card" style="border-left-color:{rec['color']}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div class="rec-title">{rec['title']}</div>
                <span style="background:{rec['color']}22;color:{rec['color']};
                  border:1px solid {rec['color']}44;padding:2px 10px;border-radius:99px;
                  font-size:11px;font-weight:700;white-space:nowrap;margin-left:8px">
                  {rec['priority']}
                </span>
              </div>
              <div class="rec-detail">{rec['detail']}</div>
              <div style="margin-top:8px">{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_health:
        st.markdown("#### 📊 Resource Health Score")
        health_items = [
            ("Fleet Utilization",       round(fdf["occupancy_pct"].mean(), 1), "#00D4FF"),
            ("On-time Performance",     round((1 - len(fdf[fdf["delay_min"]>0]) / max(len(fdf),1)) * 100, 1), "#10B981"),
            ("Model Accuracy",          metrics["accuracy"], "#7C3AED"),
            ("Capacity Optimization",   round(100 - fdf["occupancy_pct"].std(), 1), "#F59E0B"),
            ("Platform Efficiency",     round(100 - (fdf.groupby("platform")["train_id"].count().std() / fdf.groupby("platform")["train_id"].count().mean() * 30), 1), "#00D4FF"),
        ]
        for label, val, color in health_items:
            val = max(0, min(100, val))
            st.markdown(f"""
            <div style="margin-bottom:14px">
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <span style="color:#94A3B8;font-size:12px">{label}</span>
                <span style="color:{color};font-weight:700;font-size:12px">{val}%</span>
              </div>
              <div style="height:6px;background:#1E2A3D;border-radius:3px">
                <div style="width:{val}%;height:100%;background:{color};border-radius:3px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Planning Issue Breakdown")
        issues = pd.DataFrame({
            "Issue":  ["Overcrowding","Delays","Coach Gap","Platform Load","Holiday Surge"],
            "Count":  [
                len(fdf[fdf["occupancy_pct"]>=90]),
                len(fdf[fdf["delay_min"]>15]),
                len(fdf[fdf["occupancy_pct"]>=85]),
                0,
                int(forecast_df["predicted"].max() / forecast_df["predicted"].mean() * 5),
            ]
        })
        fig_issues = px.pie(issues, names="Issue", values="Count",
                            color_discrete_sequence=COLORS, hole=0.4)
        fig_issues.update_layout(**PLOT_LAYOUT, height=240)
        fig_issues.update_layout(legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_issues, use_container_width=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1E2A3D;margin-top:40px'>
<div style='text-align:center;color:#1E2A3D;font-size:11px;padding:10px 0 20px;
font-family:Space Mono,monospace'>
  RailSmart Planner v1.0 &nbsp;·&nbsp; Synthetic Data Only &nbsp;·&nbsp;
  Python · Pandas · NumPy · scikit-learn · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)