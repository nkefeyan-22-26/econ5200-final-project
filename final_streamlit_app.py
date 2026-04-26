import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.formula.api as smf
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eviction Moratoriums: Causal Analysis",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Do Local Eviction Moratoriums Reduce Eviction Filings?")
st.markdown(
    "**ECON 5200 Final Project** | Difference-in-Differences Analysis | "
    "Data: Princeton Eviction Lab ETS"
)

# ── Moratorium dates ──────────────────────────────────────────────────────────
BASE_MORATORIUM_DATES = {
    "Albuquerque, NM": ("2020-03-27", "2021-07-25"),
    "Boston, MA":      ("2020-04-20", "2021-10-17"),
    "Bridgeport, CT":  ("2020-03-19", "2021-06-30"),
    "Cincinnati, OH":  ("2020-04-01", "2020-07-25"),
    "Cleveland, OH":   ("2020-07-01", "2021-06-30"),
    "Columbus, OH":    ("2020-03-23", "2020-07-26"),
}
CONTROL_CITIES = ["Charleston, SC", "Dallas, TX", "Fort Lauderdale, FL"]
ALL_CITIES = list(BASE_MORATORIUM_DATES.keys()) + CONTROL_CITIES

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        "all_sites_weekly_2020_2021.csv",
        parse_dates=["week_date"],
        low_memory=False
    )
    df["city"] = df["city"].str.replace("Fort Lauderdale", "Fort Lauderdale, FL", regex=False)

    # Aggregate to city-week
    df_city = df.groupby(["city", "week_date"]).agg(
        filings_2020=("filings_2020", "sum"),
        filings_baseline=("filings_avg_prepandemic_baseline", "sum")
    ).reset_index()
    df_city["filing_ratio"] = (
        df_city["filings_2020"] / df_city["filings_baseline"].replace(0, np.nan)
    )

    # Drop problem cities
    df_city = df_city[~df_city["city"].isin(["Atlanta, GA", "Eugene, OR", "Austin, TX"])].copy()
    return df_city

df_raw = load_data()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Analysis Parameters")

# City selector
selected_cities = st.sidebar.multiselect(
    "Select cities to include",
    options=sorted(ALL_CITIES),
    default=sorted(ALL_CITIES)
)

# Date range
min_date = df_raw["week_date"].min().to_pydatetime()
max_date = df_raw["week_date"].max().to_pydatetime()
date_range = st.sidebar.slider(
    "Analysis date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM"
)

# Counterfactual: moratorium duration multiplier
st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Counterfactual Scenario")
duration_multiplier = st.sidebar.slider(
    "Moratorium duration multiplier",
    min_value=0.25,
    max_value=3.0,
    value=1.0,
    step=0.25,
    help="1.0 = actual duration. 2.0 = moratoriums lasted twice as long."
)
st.sidebar.caption(
    "Extends or shortens moratorium end dates proportionally from their start date."
)

# ── Build treatment variable ──────────────────────────────────────────────────
def build_treatment(df, moratorium_dates):
    def had_moratorium(row):
        city = row["city"]
        if city not in moratorium_dates:
            return 0
        start = pd.Timestamp(moratorium_dates[city][0])
        end   = pd.Timestamp(moratorium_dates[city][1])
        return int(start <= row["week_date"] <= end)

    df = df.copy()
    df["moratorium"]    = df.apply(had_moratorium, axis=1)
    df["ever_treated"]  = df["city"].isin(moratorium_dates).astype(int)
    return df

def scale_moratorium_dates(base_dates, multiplier):
    scaled = {}
    for city, (start_str, end_str) in base_dates.items():
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str)
        duration_days = (end - start).days
        new_end = start + pd.Timedelta(days=int(duration_days * multiplier))
        scaled[city] = (start_str, new_end.strftime("%Y-%m-%d"))
    return scaled

# ── Run DiD regression ────────────────────────────────────────────────────────
def run_twfe(df):
    df_model = df.dropna(subset=["filing_ratio"]).copy()
    if df_model["city"].nunique() < 2 or df_model["moratorium"].sum() == 0:
        return None, None, None
    try:
        model = smf.ols(
            "filing_ratio ~ moratorium + C(city) + C(week_date)",
            data=df_model
        ).fit(cov_type="HC1")
        est = model.params["moratorium"]
        ci  = model.conf_int().loc["moratorium"].values
        return est, ci[0], ci[1]
    except Exception:
        return None, None, None

# ── Filter data based on sidebar ──────────────────────────────────────────────
df_filtered = df_raw[
    (df_raw["city"].isin(selected_cities)) &
    (df_raw["week_date"] >= pd.Timestamp(date_range[0])) &
    (df_raw["week_date"] <= pd.Timestamp(date_range[1]))
].copy()

# Actual moratorium dates
df_actual = build_treatment(df_filtered, BASE_MORATORIUM_DATES)

# Counterfactual moratorium dates
scaled_dates = scale_moratorium_dates(BASE_MORATORIUM_DATES, duration_multiplier)
df_counter   = build_treatment(df_filtered, scaled_dates)

# Regressions
est_actual, ci_lo_actual, ci_hi_actual     = run_twfe(df_actual)
est_counter, ci_lo_counter, ci_hi_counter  = run_twfe(df_counter)

# ── Layout: metrics row ───────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cities included", df_actual["city"].nunique())
with col2:
    st.metric("City-week observations", f"{df_actual.dropna(subset=['filing_ratio']).shape[0]:,}")
with col3:
    if est_actual is not None:
        st.metric(
            "DiD Estimate (actual)",
            f"{est_actual:.3f}",
            delta=f"95% CI: [{ci_lo_actual:.3f}, {ci_hi_actual:.3f}]",
            delta_color="off"
        )
    else:
        st.metric("DiD Estimate", "N/A")
with col4:
    if est_counter is not None and duration_multiplier != 1.0:
        st.metric(
            f"DiD Estimate ({duration_multiplier}x duration)",
            f"{est_counter:.3f}",
            delta=f"95% CI: [{ci_lo_counter:.3f}, {ci_hi_counter:.3f}]",
            delta_color="off"
        )
    elif duration_multiplier == 1.0:
        st.metric("Counterfactual", "Adjust slider →")
    else:
        st.metric("Counterfactual Estimate", "N/A")

# ── Plot 1: Parallel trends ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Parallel Trends: Treated vs. Control Cities")

trends = (
    df_actual.dropna(subset=["filing_ratio"])
    .groupby(["week_date", "ever_treated"])["filing_ratio"]
    .mean()
    .reset_index()
)

fig_trends = go.Figure()
colors = {0: "#2196F3", 1: "#FF9800"}
labels = {0: "Control cities", 1: "Treated cities"}

for group in [0, 1]:
    grp = trends[trends["ever_treated"] == group]
    fig_trends.add_trace(go.Scatter(
        x=grp["week_date"], y=grp["filing_ratio"],
        mode="lines", name=labels[group],
        line=dict(color=colors[group], width=2)
    ))

fig_trends.add_hline(y=1.0, line_dash="dash", line_color="gray",
                     annotation_text="Pre-pandemic baseline")
fig_trends.add_vline(x=pd.Timestamp("2020-03-27").timestamp() * 1000,
                     line_dash="dot", line_color="red",
                     annotation_text="First moratorium")
fig_trends.update_layout(
    yaxis_title="Filing Ratio (1.0 = pre-pandemic baseline)",
    xaxis_title="Week",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=400,
    template="plotly_white"
)
st.plotly_chart(fig_trends, use_container_width=True)

# ── Plot 2: Coefficient plot — actual vs counterfactual ───────────────────────
st.markdown("---")
st.subheader("🔮 Counterfactual: What If Moratorium Duration Changed?")

if est_actual is not None and est_counter is not None:
    labels_cf  = ["Actual duration", f"{duration_multiplier}x duration"]
    estimates  = [est_actual, est_counter]
    ci_lows    = [ci_lo_actual, ci_lo_counter]
    ci_highs   = [ci_hi_actual, ci_hi_counter]
    bar_colors = ["#2196F3", "#FF9800"]

    fig_cf = go.Figure()
    for i, (label, est, lo, hi, color) in enumerate(
        zip(labels_cf, estimates, ci_lows, ci_highs, bar_colors)
    ):
        fig_cf.add_trace(go.Scatter(
            x=[label], y=[est],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[hi - est],
                arrayminus=[est - lo],
                color=color
            ),
            mode="markers",
            marker=dict(size=14, color=color),
            name=label
        ))

    fig_cf.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_cf.update_layout(
        yaxis_title="Estimated effect on filing ratio",
        xaxis_title="Scenario",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig_cf, use_container_width=True)

    # Counterfactual interpretation
    direction = "larger" if abs(est_counter) > abs(est_actual) else "smaller"
    pct_change = ((est_counter - est_actual) / abs(est_actual)) * 100 if est_actual != 0 else 0
    st.info(
        f"**Counterfactual result:** If moratorium duration were **{duration_multiplier}x** "
        f"actual length, the estimated effect would be **{est_counter:.3f}** "
        f"(95% CI: [{ci_lo_counter:.3f}, {ci_hi_counter:.3f}]), "
        f"a **{abs(pct_change):.1f}% {'increase' if pct_change < 0 else 'decrease'}** "
        f"in magnitude compared to the actual estimate of {est_actual:.3f}."
    )
else:
    st.warning("Not enough data to compute estimates. Try adjusting city or date filters.")

# ── Plot 3: City-level filing ratios ──────────────────────────────────────────
st.markdown("---")
st.subheader("🏙️ City-Level Filing Ratios Over Time")

city_options = sorted(df_actual["city"].unique())
selected_city = st.selectbox("Select a city to inspect", city_options)

city_df = df_actual[df_actual["city"] == selected_city].sort_values("week_date")
is_treated = city_df["ever_treated"].iloc[0] == 1

fig_city = go.Figure()
fig_city.add_trace(go.Scatter(
    x=city_df["week_date"], y=city_df["filing_ratio"],
    mode="lines", name="Weekly ratio",
    line=dict(color="#90CAF9", width=1), opacity=0.5
))
# 4-week rolling average
rolling = city_df["filing_ratio"].rolling(4, center=True).mean()
fig_city.add_trace(go.Scatter(
    x=city_df["week_date"], y=rolling,
    mode="lines", name="4-week average",
    line=dict(color="#1565C0", width=2.5)
))

# Shade moratorium period if treated
if is_treated and selected_city in BASE_MORATORIUM_DATES:
    m_start = pd.Timestamp(BASE_MORATORIUM_DATES[selected_city][0])
    m_end   = pd.Timestamp(BASE_MORATORIUM_DATES[selected_city][1])
    fig_city.add_vrect(x0=m_start.timestamp() * 1000, x1=m_end.timestamp() * 1000,
        fillcolor="orange", opacity=0.15,
        annotation_text="Moratorium active", annotation_position="top left"
    )

fig_city.add_hline(y=1.0, line_dash="dash", line_color="gray")
fig_city.update_layout(
    yaxis_title="Filing Ratio (1.0 = baseline)",
    xaxis_title="Week",
    title=f"{selected_city} ({'Treated' if is_treated else 'Control'})",
    height=380,
    template="plotly_white"
)
st.plotly_chart(fig_city, use_container_width=True)

# ── Model summary table ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Estimate Summary")

summary_data = {
    "Model":    ["Naive OLS (no controls)", "DiD TWFE (actual duration)", f"DiD TWFE ({duration_multiplier}x duration)"],
    "Estimate": [-0.5527, round(est_actual, 4) if est_actual else "N/A",
                 round(est_counter, 4) if est_counter else "N/A"],
    "CI Lower": [-0.6196, round(ci_lo_actual, 4) if ci_lo_actual else "N/A",
                 round(ci_lo_counter, 4) if ci_lo_counter else "N/A"],
    "CI Upper": [-0.4858, round(ci_hi_actual, 4) if ci_hi_actual else "N/A",
                 round(ci_hi_counter, 4) if ci_hi_counter else "N/A"],
    "Note": [
        "Biased — confounds COVID shocks with policy",
        "Controls for city + time FEs",
        "Counterfactual duration scenario"
    ]
}
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.caption(
    "Data: Princeton Eviction Lab Eviction Tracking System | "
    "Method: Two-Way Fixed Effects DiD with HC1 robust SEs | "
    "ECON 5200, Northeastern University, Spring 2026"
)
