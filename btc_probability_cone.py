"""
BTC Probability Cone Projection
--------------------------------
Тянет дневные свечи BTC-USDT с KuCoin, считает EGARCH волатильность (с учётом асимметрии),
HMM определяет текущий режим (дрифт), Monte Carlo с Student-t шоками (жирные хвосты)
строит веер будущих цен -> перцентили 5/25/50/75/95 -> график-конус.

Запуск: python btc_probability_cone.py
Зависимости: pip install requests pandas numpy matplotlib arch hmmlearn
"""

import os
import json

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

# ---------------- CONFIG ----------------
SYMBOL = "BTC-USDT"
TIMEFRAME = "1day"
LOOKBACK_DAYS = 500
FORECAST_BARS = 20
N_SIMS = 10000
T_DOF = 4                # степени свободы Student-t (меньше = жирнее хвосты)

# Пишем прямо в docs/ — эту папку GitHub Pages отдаёт как сайт
OUTPUT_DIR = "docs"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "btc_probability_cone.png")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "btc_probability_cone.json")


# ---------------- 1. ДАННЫЕ ----------------
def fetch_kucoin_daily(symbol=SYMBOL, days=LOOKBACK_DAYS):
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - days * 86400
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {"type": TIMEFRAME, "symbol": symbol, "startAt": start, "endAt": end}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()["data"]  # [time, open, close, high, low, volume, turnover], newest first
    df = pd.DataFrame(data, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
    df = df.astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("time").reset_index(drop=True)


# ---------------- 2. EGARCH ВОЛАТИЛЬНОСТЬ (асимметрия: падения пугают рынок сильнее роста) ----------------
def fit_garch_vol(log_returns):
    am = arch_model(log_returns * 100, vol="EGARCH", p=1, o=1, q=1, dist="normal")
    res = am.fit(disp="off")
    forecast = res.forecast(horizon=1)
    sigma_today = np.sqrt(forecast.variance.values[-1, 0]) / 100
    return sigma_today


# ---------------- 3. HMM РЕЖИМ ----------------
def detect_regime(log_returns, n_states=3):
    X = log_returns.values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                         n_iter=200, random_state=42)
    model.fit(X)
    states = model.predict(X)
    current_state = states[-1]
    regime_drift = model.means_.flatten()[current_state]
    return regime_drift, current_state


# ---------------- 4. MONTE CARLO ----------------
def simulate_paths(last_price, drift, sigma, n_sims=N_SIMS, horizon=FORECAST_BARS, t_dof=T_DOF):
    t_raw = np.random.standard_t(t_dof, size=(n_sims, horizon))
    t_scale = sigma / np.sqrt(t_dof / (t_dof - 2))  # масштабируем t под нужную сигму
    shocks = t_raw * t_scale
    log_paths = np.cumsum(drift + shocks, axis=1)
    return last_price * np.exp(log_paths)


# ---------------- 5. ГРАФИК ----------------
def plot_cone(hist_df, price_paths, forecast_bars=FORECAST_BARS):
    last_price = hist_df["close"].iloc[-1]
    p5, p25, p50, p75, p95 = np.percentile(price_paths, [5, 25, 50, 75, 95], axis=0)

    hist_dates = hist_df["time"].iloc[-90:]
    hist_prices = hist_df["close"].iloc[-90:]
    last_date = hist_df["time"].iloc[-1]
    fwd_dates = pd.date_range(last_date, periods=forecast_bars + 1, freq="D")[1:]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.plot(hist_dates, hist_prices, color="#e0e0e0", linewidth=1.4, label="История")
    ax.fill_between(fwd_dates, p5, p95, color="#3a7bd5", alpha=0.18, label="90% интервал")
    ax.fill_between(fwd_dates, p25, p75, color="#3a7bd5", alpha=0.35, label="50% интервал")
    ax.plot(fwd_dates, p50, color="#f5a623", linestyle="--", linewidth=1.8, label="Медиана")
    ax.scatter([last_date], [last_price], color="white", s=25, zorder=5)

    p_up = (price_paths[:, -1] > last_price).mean() * 100
    ax.annotate(f"P(рост через {forecast_bars} дней) ≈ {p_up:.0f}%",
                xy=(fwd_dates[-1], p50[-1]), xytext=(-170, 20),
                textcoords="offset points", color="#f5a623", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#f5a623", alpha=0.7))

    ax.set_title(f"BTC-USDT Probability Cone ({forecast_bars}d forecast)", color="#e0e0e0")
    ax.legend(loc="upper left", framealpha=0.2, fontsize=9)
    ax.grid(alpha=0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, facecolor="#1a1a1a")
    return p_up, (p5[-1], p25[-1], p50[-1], p75[-1], p95[-1])


# ---------------- 6. JSON ДЛЯ СТРАНИЦЫ ----------------
def write_json(p_up, levels, last_price, regime, sigma, drift):
    p5, p25, p50, p75, p95 = levels
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "last_price": round(float(last_price), 2),
        "forecast_bars": FORECAST_BARS,
        "p_up_pct": round(float(p_up), 1),
        "regime_state": int(regime),
        "garch_sigma": round(float(sigma), 5),
        "drift": round(float(drift), 5),
        "percentiles": {
            "p5": round(float(p5), 2),
            "p25": round(float(p25), 2),
            "p50": round(float(p50), 2),
            "p75": round(float(p75), 2),
            "p95": round(float(p95), 2),
        },
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print("JSON сохранён:", OUTPUT_JSON)


# ---------------- MAIN ----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = fetch_kucoin_daily()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    log_returns = df["log_ret"].dropna()

    sigma = fit_garch_vol(log_returns)
    drift, regime = detect_regime(log_returns)

    last_price = df["close"].iloc[-1]
    paths = simulate_paths(last_price, drift, sigma)
    p_up, levels = plot_cone(df, paths)

    print(f"Текущая цена: {last_price:.2f}")
    print(f"Режим (HMM state): {regime}, дрифт: {drift:.5f}, GARCH sigma: {sigma:.5f}")
    print(f"P(рост через {FORECAST_BARS} дней): {p_up:.1f}%")
    print(f"5/25/50/75/95 перцентили: {[round(x, 2) for x in levels]}")

    write_json(p_up, levels, last_price, regime, sigma, drift)


if __name__ == "__main__":
    main()
