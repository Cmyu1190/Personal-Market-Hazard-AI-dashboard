import streamlit as st
import hmac
import requests
import numpy as np
import pandas as pd

st.set_page_config(page_title="Personal Market Hazard AI (Stable)", layout="wide")

# --------------------------
# Auth: Simple Password Gate
# --------------------------
def check_password():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
        return True

    st.title("🔒 Private Dashboard")
    st.caption("輸入密碼才能進入（個人使用）")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        expected = st.secrets.get("APP_PASSWORD", "")
        if expected and hmac.compare_digest(pw, expected):
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("密碼錯誤")
    return False

if not check_password():
    st.stop()

# --------------------------
# API endpoints
# --------------------------
BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# --------------------------
# Robust fetchers (never crash UI)
# --------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_spot_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame | None:
    try:
        url = f"{BINANCE_SPOT_BASE}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()

        cols = [
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","n_trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close","high","low","volume"]).reset_index(drop=True)
        df.attrs["source"] = "spot"
        return df
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_funding_rate_safe(symbol: str) -> float:
    """
    Optional enhancer. If blocked -> return NaN, never crash.
    """
    try:
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": 1}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return float("nan")
        data = r.json()
        if not data:
            return float("nan")
        return float(data[-1]["fundingRate"])  # decimal
    except Exception:
        return float("nan")

# --------------------------
# Indicators
# --------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

def compute_features(df: pd.DataFrame, lookback_sr: int = 100) -> dict:
    close = df["close"]
    vol = df["volume"]

    ema20 = ema(close, 20)
    atr14 = atr(df, 14)
    rsi14 = rsi(close, 14)

    atr_avg = atr14.rolling(50).mean()
    vol_avg = vol.rolling(50).mean()

    lb = min(lookback_sr, len(df))
    recent = df.iloc[-lb:]
    res = float(recent["high"].max())
    sup = float(recent["low"].min())

    price = float(close.iloc[-1])

    return {
        "price": price,
        "ema20": float(ema20.iloc[-1]),
        "atr14": float(atr14.iloc[-1]) if np.isfinite(atr14.iloc[-1]) else float("nan"),
        "atr_avg": float(atr_avg.iloc[-1]) if np.isfinite(atr_avg.iloc[-1]) else float("nan"),
        "rsi14": float(rsi14.iloc[-1]) if np.isfinite(rsi14.iloc[-1]) else float("nan"),
        "vol_current": float(vol.iloc[-1]),
        "vol_avg": float(vol_avg.iloc[-1]) if np.isfinite(vol_avg.iloc[-1]) else float("nan"),
        "res": res,
        "sup": sup,
    }

def flags_from_features(feats: dict, funding_rate: float):
    price = feats["price"]
    ema20 = feats["ema20"]

    atr_high = False
    ema_stretched = False
    position_edge = False
    volume_high = False
    funding_extreme = False

    if np.isfinite(feats["atr14"]) and np.isfinite(feats["atr_avg"]):
        atr_high = feats["atr14"] > feats["atr_avg"]

    if ema20 != 0:
        deviation_percent = abs(price - ema20) / abs(ema20) * 100
        ema_stretched = deviation_percent > 3.0

    res = feats["res"]
    sup = feats["sup"]
    rng = res - sup
    if rng > 0:
        dist_to_res = abs(res - price)
        dist_to_sup = abs(price - sup)
        position_edge = (dist_to_res < rng * 0.15) or (dist_to_sup < rng * 0.15)

    if np.isfinite(feats["vol_avg"]):
        volume_high = feats["vol_current"] > feats["vol_avg"] * 1.5

    # funding extreme if available
    if np.isfinite(funding_rate):
        funding_extreme = abs(funding_rate) >= 0.0005  # 0.05%

    return atr_high, ema_stretched, position_edge, volume_high, funding_extreme

def score_one_tf(atr_high, ema_stretched, position_edge, rsi_val, volume_high, funding_extreme):
    score = 0
    factors = []
    if atr_high:
        score += 20; factors.append("波動放大")
    if ema_stretched:
        score += 20; factors.append("價格衝太快")
    if position_edge:
        score += 20; factors.append("快撞到牆壁了")
    if np.isfinite(rsi_val) and (rsi_val > 70 or rsi_val < 30):
        score += 15; factors.append("大家都太瘋狂")
    if volume_high:
        score += 15; factors.append("人擠人太吵雜")
    if funding_extreme:
        score += 10; factors.append("借錢代價太高")
    return max(0, min(100, score)), factors

def regime_and_rules(score: int, factors: list[str]):
    def simple_reason(score, factors):
        if score <= 30:
            return "現在市場就像平靜的公園，大家都乖乖排隊，沒有奇怪的事情發生，是最適合玩耍（交易）的時候。"
        if score <= 60:
            return f"現在有點像快要下雨了（{'、'.join(factors)}），路面變得有點滑。雖然還能出門，但你要走慢一點，不要跑太快。"
        if score <= 80:
            return f"外面正在刮大風（{'、'.join(factors)}），如果你一定要出門，必須穿好盔甲（小倉位），隨時準備跑回家躲起來。"
        return f"現在像是大地震或龍捲風（{'、'.join(factors)}），路上到處是陷阱。聰明的小孩現在應該乖乖待在家裡看書，絕對不要出門！"

    if score <= 30:
        return "Normal (正常)", "🟢 評分優良：環境穩定，可以按照原定計畫進場。", 1.0, \
               ["允許正常交易計畫進場", "保持常規風險控管", "可依訊號順勢加碼"], simple_reason(score, factors)
    if score <= 60:
        return "Caution (注意)", "🟡 風險升溫：可以進場，但必須縮減一半倉位，嚴禁貪心。", 0.5, \
               ["減少一半倉位大小", "禁止追單（錯過不追）", "提高止盈敏感度", "若虧損一單即暫停觀察"], simple_reason(score, factors)
    if score <= 80:
        return "Survival (生存模式)", "🟠 極高風險：不建議新開單。若非進不可，僅能用極小資金試探。", 0.25, \
               ["只允許極小倉位試單", "只允許一次進場，絕對禁止加碼", "嚴格設定硬止損", "目標轉換為「活下來」而非獲利"], simple_reason(score, factors)
    return "No Fight (禁止交易)", "🔴 絕對禁止：目前環境不適合任何策略，進場等於送錢給市場。", 0.0, \
           ["立刻關閉交易軟體", "目前市場極度危險或處於隨機波動", "去散步、看書或睡覺", "保護本金，等待市場結構重置"], simple_reason(score, factors)

def aggregate_mtf(scores: dict[str, int]) -> tuple[int, list[str]]:
    # A: 5m+15m+1h
    w = {"5m": 0.25, "15m": 0.35, "1h": 0.40}
    base = int(round(sum(scores[k] * w[k] for k in scores)))
    spread = max(scores.values()) - min(scores.values())

    notes = [f"加權分數（5m/15m/1h）= {base}"]
    if spread >= 60:
        base = min(100, base + 20)
        notes.append(f"時間框架衝突很大（spread={spread}）→ +20（掃單/假突破風險）")
    elif spread >= 40:
        base = min(100, base + 10)
        notes.append(f"時間框架衝突偏大（spread={spread}）→ +10（盤勢不一致）")
    else:
        notes.append(f"時間框架一致性良好（spread={spread}）")
    return base, notes

# --------------------------
# UI
# --------------------------
st.title("🛡️ Personal Market Hazard AI（穩定版：Spot K線 + MTF）")
st.caption("使用 Binance Spot K線（雲端更穩），Funding 只做加分項：抓不到就自動忽略")

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    symbol = st.selectbox("標的", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], index=0)
with c2:
    lookback_sr = st.number_input("支撐/阻力回看根數", min_value=50, max_value=300, value=100, step=10)
with c3:
    limit = st.number_input("K 線拉取根數", min_value=200, max_value=1500, value=500, step=50)
with c4:
    use_funding = st.toggle("使用 Funding 加成（可選）", value=False)

if st.button("🔄 Refresh now"):
    fetch_spot_klines.clear()
    fetch_funding_rate_safe.clear()
    st.cache_data.clear()
    st.rerun()

funding = float("nan")
if use_funding:
    funding = fetch_funding_rate_safe(symbol)

tfs = ["5m", "15m", "1h"]
dfs = {}
feats_map = {}
score_map = {}
factors_map = {}

with st.spinner("Fetching Spot data (5m/15m/1h)..."):
    for tf in tfs:
        df = fetch_spot_klines(symbol, tf, limit=int(limit))
        if df is None or df.empty:
            st.error(f"❌ 無法取得 {symbol} {tf} K線（Spot API 可能暫時限流）。請稍後再試。")
            st.stop()

        dfs[tf] = df
        feats = compute_features(df, lookback_sr=int(lookback_sr))
        feats_map[tf] = feats

        atr_high, ema_stretched, position_edge, volume_high, funding_extreme = flags_from_features(feats, funding)
        s, fac = score_one_tf(atr_high, ema_stretched, position_edge, feats["rsi14"], volume_high, funding_extreme)
        score_map[tf] = s
        factors_map[tf] = fac

total_score, mtf_notes = aggregate_mtf(score_map)

combined_factors = []
for tf in tfs:
    for f in factors_map[tf]:
        if f not in combined_factors:
            combined_factors.append(f)

regime, diagnosis, multiplier, rules, reason = regime_and_rules(total_score, combined_factors)

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.subheader("📌 MTF Scores（Spot）")
    mtf_df = pd.DataFrame([{
        "5m_score": score_map["5m"],
        "15m_score": score_map["15m"],
        "1h_score": score_map["1h"],
        "funding_% (optional)": (funding * 100) if np.isfinite(funding) else np.nan,
        "data_source": "Binance Spot",
    }])
    st.dataframe(mtf_df, use_container_width=True)

    st.subheader("🧾 MTF Notes")
    for n in mtf_notes:
        st.write(f"- {n}")

    st.subheader("📈 Price charts (close)")
    tab5, tab15, tab1h = st.tabs(["5m", "15m", "1h"])
    with tab5:
        st.line_chart(dfs["5m"][["open_time", "close"]].set_index("open_time"))
    with tab15:
        st.line_chart(dfs["15m"][["open_time", "close"]].set_index("open_time"))
    with tab1h:
        st.line_chart(dfs["1h"][["open_time", "close"]].set_index("open_time"))

with right:
    st.subheader("✅ Final Hazard Output")
    st.metric("Hazard Score (MTF)", total_score)
    st.metric("Regime", regime)
    st.metric("Position Multiplier", f"{multiplier}x")
    st.info(diagnosis)

    st.markdown("#### 為什麼？（科普版）")
    st.write(f"「{reason}」")

    st.markdown("#### Rules")
    for rule in rules:
        st.write(f"- {rule}")

    st.markdown("#### 觸發因素（合併）")
    st.write("、".join(combined_factors) if combined_factors else "（目前沒有明顯觸發因素）")
