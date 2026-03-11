import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Renewable Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df        = pd.read_csv('data/energy_weather_london.csv',    index_col='time', parse_dates=True)
    daily     = pd.read_csv('data/daily_energy_weather_london.csv', index_col='time', parse_dates=True)
    forecasts = pd.read_csv('data/forecasts.csv',                index_col='date', parse_dates=True)
    metrics   = pd.read_csv('data/model_metrics.csv')
    return df, daily, forecasts, metrics

df, daily, forecasts, metrics = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/e/e8/Shell_pecten.svg/200px-Shell_pecten.svg.png", width=60)
st.sidebar.title("Controls")

energy_type = st.sidebar.selectbox("Energy Source", ["Solar", "Wind", "Both"])
model_type  = st.sidebar.selectbox("Forecast Model", ["Prophet", "SARIMA", "Both"])
show_raw    = st.sidebar.checkbox("Show raw hourly data", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** UK Renewable Energy Forecasting  \n**Location:** London, UK  \n**Data:** 2019–2023")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("UK Renewable Energy Forecasting Dashboard")
st.markdown("Comparing **SARIMA** and **Prophet** models for solar and wind capacity factor forecasting — London, UK")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.subheader("Model Performance (2023 Test Year)")

col1, col2, col3, col4 = st.columns(4)

def get_metric(metrics, model, target, metric):
    row = metrics[(metrics['model'] == model) & (metrics['target'] == target)]
    return round(row[metric].values[0], 4) if len(row) else "N/A"

col1.metric("SARIMA Solar RMSE",   get_metric(metrics, 'SARIMA',  'Solar', 'rmse'))
col2.metric("Prophet Solar RMSE",  get_metric(metrics, 'Prophet', 'Solar', 'rmse'))
col3.metric("SARIMA Wind RMSE",    get_metric(metrics, 'SARIMA',  'Wind',  'rmse'))
col4.metric("Prophet Wind RMSE",   get_metric(metrics, 'Prophet', 'Wind',  'rmse'))

st.markdown("---")

# ── Forecast Plot ─────────────────────────────────────────────────────────────
st.subheader("Forecast vs Actual — 2023")

targets = ['Solar', 'Wind'] if energy_type == 'Both' else [energy_type]
colors  = {'Solar': '#f5a623', 'Wind': '#4a90d9'}
model_colors = {'SARIMA': '#e74c3c', 'PROPHET': '#2ecc71'}

for target in targets:
    fig, ax = plt.subplots(figsize=(13, 4))
    col_actual = f"{target.lower()}_actual"

    # Plot actual
    ax.plot(forecasts.index, forecasts[col_actual],
            color=colors[target], linewidth=1.5, label=f'Actual {target}')

    # Plot selected models
    models_to_plot = ['sarima', 'prophet'] if model_type == 'Both' else [model_type.lower()]
    for m in models_to_plot:
        col = f"{target.lower()}_{m}"
        ax.plot(forecasts.index, forecasts[col],
                color=model_colors[m.upper()],
                linewidth=1.2, linestyle='--', label=f'{m.upper()} Forecast')

    ax.set_title(f'{target} Capacity Factor — Forecast vs Actual (2023)')
    ax.set_ylabel('Capacity Factor')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Seasonality Section ───────────────────────────────────────────────────────
st.subheader("Seasonality Analysis")

tab1, tab2, tab3 = st.tabs(["Monthly Patterns", "Hourly Patterns", "Complementarity"])

with tab1:
    monthly = daily.groupby(daily.index.month)[['solar_cf', 'wind_cf']].mean()
    monthly = monthly.reindex(range(1, 13))
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(12), monthly['solar_cf'].values, color='#f5a623',
            linewidth=2.5, marker='o', markersize=6, label='Solar')
    ax.plot(range(12), monthly['wind_cf'].values,  color='#4a90d9',
            linewidth=2.5, marker='o', markersize=6, label='Wind')
    ax.fill_between(range(12), monthly['solar_cf'].values, alpha=0.1, color='#f5a623')
    ax.fill_between(range(12), monthly['wind_cf'].values,  alpha=0.1, color='#4a90d9')
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel('Mean Capacity Factor')
    ax.set_title('Monthly Seasonality — Solar vs Wind (2019-2023 average)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    hourly = df.groupby(df.index.hour)[['solar_cf', 'wind_cf']].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hourly.index, hourly['solar_cf'], color='#f5a623',
                 linewidth=2, marker='o', markersize=4)
    axes[0].set_title('Average Solar Output by Hour of Day')
    axes[0].set_xlabel('Hour (UTC)')
    axes[0].set_ylabel('Mean Capacity Factor')
    axes[0].set_xticks(range(0, 24, 2))

    axes[1].plot(hourly.index, hourly['wind_cf'], color='#4a90d9',
                 linewidth=2, marker='o', markersize=4)
    axes[1].set_title('Average Wind Output by Hour of Day')
    axes[1].set_xlabel('Hour (UTC)')
    axes[1].set_xticks(range(0, 24, 2))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    daily['combined_cf'] = (daily['solar_cf'] + daily['wind_cf']) / 2
    daily_plot = daily[['solar_cf', 'wind_cf', 'combined_cf']].resample('W').mean()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily_plot.index, daily_plot['solar_cf'],    color='#f5a623',
            linewidth=0.8, alpha=0.7, label='Solar')
    ax.plot(daily_plot.index, daily_plot['wind_cf'],     color='#4a90d9',
            linewidth=0.8, alpha=0.7, label='Wind')
    ax.plot(daily_plot.index, daily_plot['combined_cf'], color='#2ecc71',
            linewidth=1.8, label='Combined (avg)')
    ax.set_ylabel('Weekly Mean Capacity Factor')
    ax.set_title('Solar + Wind Complementarity — Weekly Average (2019-2023)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Model Comparison Bar Chart ────────────────────────────────────────────────
st.subheader("SARIMA vs Prophet — Head to Head")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
model_colors_list = ['#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71']
x_labels = ['SARIMA\n(Solar)', 'Prophet\n(Solar)', 'SARIMA\n(Wind)', 'Prophet\n(Wind)']

rmse_vals = [
    get_metric(metrics, 'SARIMA',  'Solar', 'rmse'),
    get_metric(metrics, 'Prophet', 'Solar', 'rmse'),
    get_metric(metrics, 'SARIMA',  'Wind',  'rmse'),
    get_metric(metrics, 'Prophet', 'Wind',  'rmse'),
]
mae_vals = [
    get_metric(metrics, 'SARIMA',  'Solar', 'mae'),
    get_metric(metrics, 'Prophet', 'Solar', 'mae'),
    get_metric(metrics, 'SARIMA',  'Wind',  'mae'),
    get_metric(metrics, 'Prophet', 'Wind',  'mae'),
]

axes[0].bar(x_labels, rmse_vals, color=model_colors_list, alpha=0.85, edgecolor='white')
axes[0].set_title('RMSE (lower is better)')
axes[0].set_ylabel('RMSE')

axes[1].bar(x_labels, mae_vals, color=model_colors_list, alpha=0.85, edgecolor='white')
axes[1].set_title('MAE (lower is better)')
axes[1].set_ylabel('MAE')

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Raw data explorer ─────────────────────────────────────────────────────────
if show_raw:
    st.subheader("Raw Hourly Data Explorer")
    year = st.slider("Select year", 2019, 2023, 2022)
    subset = df[df.index.year == year][['solar_cf', 'wind_cf', 'temp_c', 'wind_speed_ms', 'radiation_wm2']]
    st.dataframe(subset.resample('D').mean().round(4), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: Renewables Ninja + Open-Meteo | Models: SARIMA (statsmodels) + Prophet (Meta) | Built for Shell Internship 2026")
