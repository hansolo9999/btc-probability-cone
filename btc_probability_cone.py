"""
BTC Probability Cone Projection
--------------------------------
Тянет дневные свечи BTC-USDT с KuCoin, считает GARCH(1,1) волатильность,
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

# --- Объёмы и ликвидации ---
BINANCE_FUTURES_SYMBOL = "BTCUSDT"
LEVERAGE_TIERS = [10, 25, 50, 100]     # для приближённой оценки зон ликвидации
VOLUME_SPIKE_WINDOW = 20               # дней для базовой линии объёма
VOLUME_SPIKE_Z = 2.0                   # порог z-score для флага "аномальный объём"
LARGE_TRADE_MULTIPLIER = 5             # во сколько раз сделка больше медианной = "крупная"

# --- Новости у резких движений цены ---
NEWS_LOOKBACK_DAYS = 30                 # искать резкие движения в последних N днях
NEWS_MOVE_Z = 2.5                       # порог z-score дневной доходности = "резкое движение"
NEWS_MAX_EVENTS = 3                     # сколько последних резких дней подтягивать новостями
NEWS_PER_EVENT = 2                      # сколько заголовков брать на одно событие

# --- Бумажная торговля (виртуальный счёт, без реальных денег) ---
PAPER_INITIAL_BALANCE = 10000.0
PAPER_RISK_PCT = 0.02                   # риск на сделку — 2% от текущего баланса
PAPER_LONG_THRESHOLD = 65.0             # P(рост) >= этого — открыть виртуальный лонг
PAPER_SHORT_THRESHOLD = 35.0            # P(рост) <= этого — открыть виртуальный шорт
ATR_WINDOW = 14
ATR_TP_MULT = 4                         # TP = 4 ATR (твоя проверенная формула)
ATR_SL_MULT = 1                         # SL = 1 ATR

# Пишем прямо в docs/ — эту папку GitHub Pages отдаёт как сайт
OUTPUT_DIR = "docs"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "btc_probability_cone.png")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "btc_probability_cone.json")
PAPER_STATE_PATH = os.path.join(OUTPUT_DIR, "paper_trading_state.json")
PAPER_EQUITY_PNG = os.path.join(OUTPUT_DIR, "paper_trading_equity.png")


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
def compute_volume_zscores(df, window=VOLUME_SPIKE_WINDOW):
    vol = df["volume"].astype(float)
    rolling_mean = vol.rolling(window).mean().shift(1)
    rolling_std = vol.rolling(window).std().shift(1)
    z = (vol - rolling_mean) / rolling_std
    return z.fillna(0)


def plot_cone(hist_df, price_paths, liq_zones=None, volume_info=None, news_events=None, forecast_bars=FORECAST_BARS):
    last_price = hist_df["close"].iloc[-1]
    p5, p25, p50, p75, p95 = np.percentile(price_paths, [5, 25, 50, 75, 95], axis=0)

    hist_dates = hist_df["time"].iloc[-90:]
    hist_prices = hist_df["close"].iloc[-90:]
    hist_volumes = hist_df["volume"].iloc[-90:].astype(float)
    vol_z = compute_volume_zscores(hist_df).iloc[-90:]
    last_date = hist_df["time"].iloc[-1]
    fwd_dates = pd.date_range(last_date, periods=forecast_bars + 1, freq="D")[1:]

    plt.style.use("dark_background")
    fig, (ax, ax_vol) = plt.subplots(
        2, 1, figsize=(11, 8), dpi=150,
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    # --- цена + конус ---
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

    # --- зоны ликвидации (только 10x и 25x — иначе график превращается в кашу) ---
    if liq_zones:
        for z in liq_zones:
            if z["leverage"] not in (10, 25):
                continue
            alpha = 0.55 if z["leverage"] == 10 else 0.32
            ax.axhline(z["long_liq"], color="#2ecc71", linestyle=":", linewidth=1.1, alpha=alpha)
            ax.axhline(z["short_liq"], color="#e74c3c", linestyle=":", linewidth=1.1, alpha=alpha)
            ax.text(hist_dates.iloc[0], z["long_liq"], f' {z["leverage"]}x long стоп',
                    color="#2ecc71", fontsize=7, va="bottom", alpha=min(alpha + 0.3, 1))
            ax.text(hist_dates.iloc[0], z["short_liq"], f' {z["leverage"]}x short стоп',
                    color="#e74c3c", fontsize=7, va="top", alpha=min(alpha + 0.3, 1))

    # --- метки дней с резким движением (факт, не прогноз) ---
    if news_events:
        for ev in news_events:
            ev_date = pd.to_datetime(ev["date"])
            if ev_date < hist_dates.iloc[0]:
                continue
            color = "#2ecc71" if ev["move_pct"] > 0 else "#e74c3c"
            marker = "^" if ev["move_pct"] > 0 else "v"
            ax.axvline(ev_date, color=color, linestyle="-", linewidth=0.8, alpha=0.35)
            ax.scatter([ev_date], [ev["price"]], color=color, marker=marker, s=60, zorder=6, edgecolors="white", linewidths=0.5)
            ax.annotate(f'{ev["move_pct"]:+.1f}%', xy=(ev_date, ev["price"]),
                        xytext=(0, 12 if ev["move_pct"] > 0 else -16),
                        textcoords="offset points", color=color, fontsize=7,
                        ha="center", fontweight="bold")

    ax.set_title(f"BTC-USDT Probability Cone ({forecast_bars}d forecast)", color="#e0e0e0")
    ax.legend(loc="upper left", framealpha=0.2, fontsize=9)
    ax.grid(alpha=0.1)

    # --- объём: аномальные дни (z >= порога) подсвечены оранжевым ---
    bar_colors = ["#f5a623" if zz >= VOLUME_SPIKE_Z else "#3a7bd5" for zz in vol_z]
    ax_vol.bar(hist_dates, hist_volumes, color=bar_colors, width=0.8)
    if volume_info:
        ax_vol.axhline(volume_info["avg_volume"], color="#888", linestyle="--", linewidth=1, alpha=0.6)
    ax_vol.set_ylabel("Объём", color="#888", fontsize=9)
    ax_vol.grid(alpha=0.1)
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, facecolor="#1a1a1a")
    return p_up, (p5[-1], p25[-1], p50[-1], p75[-1], p95[-1])


# ---------------- 6. ВСПЛЕСК ОБЪЁМА ----------------
def detect_volume_spike(df, window=VOLUME_SPIKE_WINDOW, z_thresh=VOLUME_SPIKE_Z):
    vol = df["volume"].astype(float)
    baseline = vol.iloc[-(window + 1):-1]
    current = vol.iloc[-1]
    mean = baseline.mean()
    std = baseline.std()
    z = (current - mean) / std if std > 0 else 0.0
    return {
        "current_volume": round(float(current), 2),
        "avg_volume": round(float(mean), 2),
        "z_score": round(float(z), 2),
        "is_spike": bool(z >= z_thresh),
    }


# ---------------- 7. ПРИБЛИЖЁННЫЕ ЗОНЫ ЛИКВИДАЦИЙ ----------------
def fetch_binance_futures_data(symbol=BINANCE_FUTURES_SYMBOL):
    try:
        oi = requests.get("https://fapi.binance.com/fapi/v1/openInterest",
                           params={"symbol": symbol}, timeout=10)
        oi.raise_for_status()
        funding = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                params={"symbol": symbol}, timeout=10)
        funding.raise_for_status()
        oi_data = oi.json()
        f_data = funding.json()
        return {
            "open_interest_btc": round(float(oi_data["openInterest"]), 2),
            "funding_rate_pct": round(float(f_data["lastFundingRate"]) * 100, 4),
            "mark_price": round(float(f_data["markPrice"]), 2),
        }
    except Exception as e:
        print("Binance futures data недоступны:", e)
        return None


def estimate_liquidation_zones(current_price, leverages=LEVERAGE_TIERS):
    zones = []
    for lev in leverages:
        long_liq = current_price * (1 - 1 / lev)
        short_liq = current_price * (1 + 1 / lev)
        zones.append({
            "leverage": lev,
            "long_liq": round(long_liq, 2),
            "short_liq": round(short_liq, 2),
        })
    return zones


# ---------------- 8. КРУПНЫЕ СДЕЛКИ ----------------
def fetch_kucoin_recent_trades(symbol=SYMBOL):
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/histories",
                          params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        data = r.json()["data"]
        return [{"price": float(t["price"]), "size": float(t["size"]),
                  "side": t["side"], "time": int(t["time"])} for t in data]
    except Exception as e:
        print("KuCoin trade history недоступна:", e)
        return []


def detect_large_trades(trades, multiplier=LARGE_TRADE_MULTIPLIER, top_n=5):
    if not trades:
        return []
    sizes = [t["size"] for t in trades]
    median_size = np.median(sizes)
    if median_size == 0:
        return []
    large = [t for t in trades if t["size"] >= median_size * multiplier]
    large.sort(key=lambda t: t["size"], reverse=True)
    return large[:top_n]


# ---------------- 9. НОВОСТИ У РЕЗКИХ ДВИЖЕНИЙ ----------------
def compute_return_zscores(df, window=VOLUME_SPIKE_WINDOW):
    ret = np.log(df["close"] / df["close"].shift(1))
    rolling_mean = ret.rolling(window).mean().shift(1)
    rolling_std = ret.rolling(window).std().shift(1)
    z = (ret - rolling_mean) / rolling_std
    return z.fillna(0)


def detect_big_move_days(df, lookback=NEWS_LOOKBACK_DAYS, z_thresh=NEWS_MOVE_Z, max_events=NEWS_MAX_EVENTS):
    z = compute_return_zscores(df)
    recent = df.iloc[-lookback:].copy()
    recent["z"] = z.iloc[-lookback:].values
    recent["ret_pct"] = (recent["close"] / recent["close"].shift(1) - 1) * 100
    big_moves = recent[recent["z"].abs() >= z_thresh]
    big_moves = big_moves.sort_values("time", ascending=False).head(max_events)
    return big_moves[["time", "close", "ret_pct", "z"]].to_dict("records")


def fetch_news_near_date(target_dt, max_articles=NEWS_PER_EVENT):
    try:
        target_ts = int(target_dt.timestamp())
        r = requests.get("https://min-api.cryptocompare.com/data/v2/news/",
                          params={"lang": "EN", "lTs": target_ts + 86400}, timeout=10)
        r.raise_for_status()
        articles = r.json().get("Data", [])
        articles.sort(key=lambda a: abs(a["published_on"] - target_ts))
        return [{
            "title": a["title"],
            "source": a.get("source_info", {}).get("name", a.get("source", "")),
            "url": a["url"],
            "published_on": datetime.fromtimestamp(a["published_on"], tz=timezone.utc).isoformat(),
        } for a in articles[:max_articles]]
    except Exception as e:
        print("Новости недоступны для даты", target_dt, ":", e)
        return []


def build_news_events(df):
    big_moves = detect_big_move_days(df)
    events = []
    for m in big_moves:
        news = fetch_news_near_date(m["time"])
        events.append({
            "date": m["time"].isoformat(),
            "move_pct": round(float(m["ret_pct"]), 2),
            "price": round(float(m["close"]), 2),
            "z_score": round(float(m["z"]), 2),
            "news": news,
        })
    return events


# ---------------- 10. JSON ДЛЯ СТРАНИЦЫ ----------------
def write_json(p_up, levels, last_price, regime, sigma, drift,
                volume_info, liq_zones, futures_data, large_trades, news_events, paper_stats):
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
        "volume": volume_info,
        "liquidation_zones": liq_zones,
        "futures": futures_data,
        "large_trades": [
            {**t, "time_iso": datetime.fromtimestamp(t["time"] / 1e9, tz=timezone.utc).isoformat()}
            for t in large_trades
        ] if large_trades else [],
        "news_events": news_events or [],
        "paper_trading": paper_stats,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print("JSON сохранён:", OUTPUT_JSON)


# ---------------- 11. БУМАЖНАЯ ТОРГОВЛЯ (виртуальный счёт) ----------------
# Торгует не "ИИ по ощущениям", а уже посчитанный статистический сигнал (P(рост)).
# Деньги виртуальные — цель: честно проверить, есть ли у сигнала реальное преимущество.
def compute_atr(df, window=ATR_WINDOW):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return float(atr.iloc[-1])


def load_paper_state():
    try:
        with open(PAPER_STATE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"balance": PAPER_INITIAL_BALANCE, "position": None, "trade_history": [], "equity_curve": []}


def save_paper_state(state):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PAPER_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def open_paper_position(state, side, price, atr, now_iso):
    risk_amount = state["balance"] * PAPER_RISK_PCT
    sl_distance = atr * ATR_SL_MULT
    tp_distance = atr * ATR_TP_MULT
    size = risk_amount / sl_distance if sl_distance > 0 else 0
    tp = price + tp_distance if side == "long" else price - tp_distance
    sl = price - sl_distance if side == "long" else price + sl_distance
    state["position"] = {
        "side": side, "entry_price": round(price, 2), "entry_date": now_iso,
        "size": round(size, 6), "tp": round(tp, 2), "sl": round(sl, 2), "atr": round(atr, 2),
    }


def close_paper_position(state, exit_price, now_iso, reason):
    pos = state["position"]
    if pos["side"] == "long":
        pnl = (exit_price - pos["entry_price"]) * pos["size"]
    else:
        pnl = (pos["entry_price"] - exit_price) * pos["size"]
    state["balance"] = round(state["balance"] + pnl, 2)
    trade = {
        "side": pos["side"], "entry_price": pos["entry_price"], "entry_date": pos["entry_date"],
        "exit_price": round(exit_price, 2), "exit_date": now_iso, "pnl": round(pnl, 2),
        "reason": reason, "size": pos["size"],
    }
    state.setdefault("trade_history", []).append(trade)
    state["trade_history"] = state["trade_history"][-100:]
    state["position"] = None
    return trade


def update_paper_trading(df, p_up, last_price):
    state = load_paper_state()
    atr = compute_atr(df)
    now_iso = datetime.now(timezone.utc).isoformat()

    if state.get("position"):
        pos = state["position"]
        if pos["side"] == "long":
            if last_price >= pos["tp"]:
                close_paper_position(state, last_price, now_iso, "tp_hit")
            elif last_price <= pos["sl"]:
                close_paper_position(state, last_price, now_iso, "sl_hit")
        else:
            if last_price <= pos["tp"]:
                close_paper_position(state, last_price, now_iso, "tp_hit")
            elif last_price >= pos["sl"]:
                close_paper_position(state, last_price, now_iso, "sl_hit")

    if not state.get("position"):
        if p_up >= PAPER_LONG_THRESHOLD:
            open_paper_position(state, "long", last_price, atr, now_iso)
        elif p_up <= PAPER_SHORT_THRESHOLD:
            open_paper_position(state, "short", last_price, atr, now_iso)

    equity = state["balance"]
    if state.get("position"):
        pos = state["position"]
        unrealized = ((last_price - pos["entry_price"]) if pos["side"] == "long"
                      else (pos["entry_price"] - last_price)) * pos["size"]
        equity += unrealized
    state.setdefault("equity_curve", []).append({"date": now_iso, "equity": round(equity, 2), "price": round(last_price, 2)})
    state["equity_curve"] = state["equity_curve"][-500:]

    save_paper_state(state)
    return state


def compute_paper_stats(state):
    trades = state.get("trade_history", [])
    total = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = round(wins / total * 100, 1) if total > 0 else None
    total_return_pct = round((state["balance"] - PAPER_INITIAL_BALANCE) / PAPER_INITIAL_BALANCE * 100, 2)
    return {
        "balance": round(state["balance"], 2),
        "total_return_pct": total_return_pct,
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": win_rate,
        "position": state.get("position"),
    }


def plot_paper_equity(state):
    curve = state.get("equity_curve", [])
    if len(curve) < 2:
        return
    dates = [pd.to_datetime(c["date"]) for c in curve]
    equity = [c["equity"] for c in curve]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 3.5), dpi=150)
    color = "#2ecc71" if equity[-1] >= PAPER_INITIAL_BALANCE else "#e74c3c"
    ax.plot(dates, equity, color=color, linewidth=1.5)
    ax.axhline(PAPER_INITIAL_BALANCE, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_between(dates, equity, PAPER_INITIAL_BALANCE, color=color, alpha=0.1)
    ax.set_title("Бумажный счёт — виртуальный equity (старт $10,000)", color="#e0e0e0", fontsize=10)
    ax.grid(alpha=0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(PAPER_EQUITY_PNG, facecolor="#1a1a1a")
    plt.close(fig)


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

    volume_info = detect_volume_spike(df)
    liq_zones = estimate_liquidation_zones(last_price)
    futures_data = fetch_binance_futures_data()
    trades = fetch_kucoin_recent_trades()
    large_trades = detect_large_trades(trades)
    news_events = build_news_events(df)

    p_up, levels = plot_cone(df, paths, liq_zones=liq_zones, volume_info=volume_info, news_events=news_events)

    paper_state = update_paper_trading(df, p_up, last_price)
    paper_stats = compute_paper_stats(paper_state)
    plot_paper_equity(paper_state)

    print(f"Текущая цена: {last_price:.2f}")
    print(f"Режим (HMM state): {regime}, дрифт: {drift:.5f}, GARCH sigma: {sigma:.5f}")
    print(f"P(рост через {FORECAST_BARS} дней): {p_up:.1f}%")
    print(f"5/25/50/75/95 перцентили: {[round(x, 2) for x in levels]}")
    print(f"Объём: текущий {volume_info['current_volume']}, z-score {volume_info['z_score']}, спайк: {volume_info['is_spike']}")
    print(f"Крупных сделок найдено: {len(large_trades)}")
    print(f"Резких дней с новостями найдено: {len(news_events)}")
    print(f"Бумажный счёт: ${paper_stats['balance']} ({paper_stats['total_return_pct']:+.2f}%), сделок: {paper_stats['total_trades']}, win rate: {paper_stats['win_rate']}")

    write_json(p_up, levels, last_price, regime, sigma, drift,
               volume_info, liq_zones, futures_data, large_trades, news_events, paper_stats)


if __name__ == "__main__":
    main()
