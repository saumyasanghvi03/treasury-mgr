# app.py — Enterprise-ready Treasury Management (Optimized + Explained)
# Saumya — designed for large datasets (enterprise_transactions_v2 / enterprise_payments_v2)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linprog
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Treasury Control Tower — Enterprise",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional CSS
def inject_css():
    try:
        with open("styles.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

inject_css()

# -----------------------------
# Paths (auto-detect enterprise v2 datasets)
# -----------------------------
ENTERPRISE_TXN = "enterprise_transactions_v2.csv"
ENTERPRISE_PAY = "enterprise_payments_v2.csv"
ENTERPRISE_ZIP = "enterprise_treasury_datasets_v2.zip"

# -----------------------------
# Utils — IO and parsing (fast)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_transactions(path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load transactions CSV in a memory-efficient manner.
    Expects columns: date, amount, type
    """
    usecols = None  # load all — file already curated
    parse_dates = ["date"]
    df = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, nrows=nrows)
    df.columns = [c.strip().lower() for c in df.columns]
    # normalize type
    if "type" in df.columns:
        df["type"] = df["type"].str.lower().str.strip()
    # ensure amount numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(show_spinner=False)
def aggregate_daily(df_txn: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data into daily inflow/outflow/net.
    Returns DataFrame with columns: date, inflow, outflow, net
    """
    df = df_txn.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "type" in df.columns:
        inflows = df[df["type"].str.contains("in", na=False)].groupby("date")["amount"].sum()
        outflows = df[df["type"].str.contains("out", na=False)].groupby("date")["amount"].sum()
    else:
        inflows = df[df["amount"] > 0].groupby("date")["amount"].sum()
        outflows = -df[df["amount"] < 0].groupby("date")["amount"].sum()

    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D").date
    daily = pd.DataFrame({"date": all_days})
    daily["inflow"] = daily["date"].map(inflows).fillna(0.0)
    daily["outflow"] = daily["date"].map(outflows).fillna(0.0)
    daily["net"] = daily["inflow"] - daily["outflow"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily

@st.cache_data(show_spinner=False)
def load_payments(path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load payment-level intraday CSV. Expects timestamp, amount, direction (in/out).
    """
    parse_dates = ["timestamp"]
    df = pd.read_csv(path, parse_dates=parse_dates, nrows=nrows)
    df.columns = [c.strip().lower() for c in df.columns]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.lower().str.strip()
    return df

# -----------------------------
# Feature & ML utilities (cached)
# -----------------------------
def make_lagged_df(series: pd.Series, lags=(1,2,3,7)):
    df = pd.DataFrame({"y": series.reset_index(drop=True)})
    for l in lags:
        df[f"lag_{l}"] = series.shift(l).reset_index(drop=True)
    df["dow"] = np.arange(len(df)) % 7
    return df.dropna()

@st.cache_resource(show_spinner=False)
def train_light_rf(series: pd.Series):
    """
    Train and cache a small RandomForest model.
    Optimized for speed with modest predictive power (good for presentations).
    """
    ser = series.reset_index(drop=True)
    df = make_lagged_df(ser)
    X = df.drop(columns=["y"])
    y = df["y"]
    model = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=13, n_jobs=-1)
    model.fit(X, y)
    return model

def rolling_rf_forecast(series: pd.Series, horizon: int = 30):
    """
    Rolling one-step forecast using cached RF model.
    Forecast is generated only when user requests it (lazy).
    """
    hist = series.reset_index(drop=True)
    model = train_light_rf(hist)
    preds = []
    last = hist.copy()
    for _ in range(horizon):
        tmp = make_lagged_df(last)
        X_last = tmp.drop(columns=["y"]).iloc[[-1]]
        yhat = float(model.predict(X_last)[0])
        preds.append(yhat)
        last = pd.concat([last, pd.Series([yhat])], ignore_index=True)
    return np.array(preds)

# -----------------------------
# Financial computations (explanatory comments included)
# -----------------------------
def compute_lcr(h1, h2a, h2b, avg_outflow_30d, inflow_cap_ratio=0.75):
    """
    Simple Basel III LCR calculator:
      - HQLA adjusted by haircuts: L1 (1.0), L2A (0.85), L2B (0.75)
      - Inflows capped at inflow_cap_ratio of outflows (regulatory heuristic)
      - LCR = Adjusted HQLA / (Net Outflows)
    """
    adj_hqla = h1 + 0.85*h2a + 0.75*h2b
    estimated_inflows = min(avg_outflow_30d * inflow_cap_ratio, avg_outflow_30d * 0.5)
    net_out = max(avg_outflow_30d - estimated_inflows, 0.0)
    lcr = (adj_hqla / (net_out + 1e-9)) * 100.0 if net_out > 0 else np.inf
    return adj_hqla, net_out, lcr

def monte_carlo_var(mu=0.0003, sigma=0.015, days=1, sims=20000, cl=0.99, notional=1.0):
    rng = np.random.default_rng(7)
    draws = rng.normal(mu*days, sigma*np.sqrt(days), sims)
    q = np.quantile(draws, 1 - cl)
    return -q * notional

def payments_intraday_curve(df_pay: pd.DataFrame, freq='15min'):
    """
    Convert raw payments to cumulative net curve at given frequency (default 15min).
    Returns pandas Series indexed by timestamp and peak net outflow.
    """
    d = df_pay.copy()
    if "timestamp" not in d.columns or "amount" not in d.columns:
        raise ValueError("Payments file requires 'timestamp' and 'amount' columns.")
    d["slot"] = d["timestamp"].dt.floor(freq)
    if "direction" in d.columns:
        d["sign"] = np.where(d["direction"].str.startswith("in"), 1, -1)
    else:
        d["sign"] = np.where(d["amount"] >= 0, 1, -1)
    d["signed"] = d["amount"] * d["sign"]
    curve = d.groupby("slot")["signed"].sum().sort_index().cumsum()
    peak = -curve.min()
    return curve, peak

def optimize_portfolio(hqla_min_ratio, horizon_days, yields, liq_days, haircuts, total_funds=5e9, max_duration_days=120):
    """
    Linear programming-based allocator:
    - objective: maximize sum(yield_i * x_i)
    - constraints: duration cap, HQLA within horizon, full investment
    """
    instr = list(yields.keys())
    y = np.array([yields[k] for k in instr])
    dur = np.array([liq_days[k] for k in instr], dtype=float)
    hqla_adj = np.array([haircuts[k] for k in instr], dtype=float)

    A_ub = [dur.tolist()]
    b_ub = [max_duration_days * total_funds]

    projected_outflow = 0.006 * total_funds * (horizon_days / 30.0)
    required = hqla_min_ratio * projected_outflow
    A_ub.append((-hqla_adj).tolist())
    b_ub.append(-required)

    A_eq = [np.ones(len(instr)).tolist()]
    b_eq = [total_funds]

    bounds = [(0, total_funds) for _ in instr]

    res = linprog(c=(-y), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    return instr, res

# -----------------------------
# UI: Sidebar options and dataset selection
# -----------------------------
st.sidebar.title("Dataset & Performance")
dataset_choice = st.sidebar.selectbox(
    "Select dataset",
    ("Use built-in sample", "Use enterprise v2 CSVs", "Upload custom CSVs")
)

if dataset_choice == "Use enterprise v2 CSVs":
    if not (os.path.exists(ENTERPRISE_TXN) and os.path.exists(ENTERPRISE_PAY)):
        st.sidebar.error("Enterprise v2 CSVs not found in project root. Please upload or regenerate.")
        st.stop()
    tx_path = ENTERPRISE_TXN
    pay_path = ENTERPRISE_PAY
    st.sidebar.success("Enterprise v2 CSVs detected and selected.")
elif dataset_choice == "Upload custom CSVs":
    uploaded_tx = st.sidebar.file_uploader("Upload transactions.csv", type="csv")
    uploaded_pay = st.sidebar.file_uploader("Upload payments.csv", type="csv")
    if uploaded_tx is None:
        st.sidebar.info("Upload your transactions CSV to proceed.")
        st.stop()
    tx_path = uploaded_tx
    pay_path = uploaded_pay
else:
    # built-in small demo dataset (fast)
    tx_path = None
    pay_path = None

# Performance knobs
max_tx_preview = st.sidebar.slider("Preview transactions rows", 100, 50000, 2000, step=100)
preview_pay_rows = st.sidebar.slider("Preview payments rows", 100, 50000, 2000, step=100)

# -----------------------------
# Load data (cached)
# -----------------------------
if tx_path is None:
    # small sample: generate
    sample_daily = load_sample_data(days=365)
    daily = sample_daily.copy()
    transactions_df = None
else:
    with st.spinner("Loading transactions — this may take a moment for enterprise files..."):
        transactions_df = load_transactions(tx_path)
        daily = aggregate_daily(transactions_df)

if pay_path is None:
    payments_df = None
else:
    with st.spinner("Loading payments — this may take a moment..."):
        payments_df = load_payments(pay_path)

# -----------------------------
# Top-level KPIs & Explanations
# -----------------------------
st.title("Treasury Control Tower — Enterprise Edition")
st.markdown("**Overview:** High-throughput treasury simulation & analytics. Use the left pane to change datasets and analysis parameters.")

# Basic KPIs (30-day)
kpi_30 = daily.tail(30)
avg_in = kpi_30["inflow"].mean()
avg_out = kpi_30["outflow"].mean()
avg_net = kpi_30["net"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Daily Inflow (30d)", f"₹{avg_in:,.0f}")
col2.metric("Avg Daily Outflow (30d)", f"₹{avg_out:,.0f}")
col3.metric("Avg Net (30d)", f"₹{avg_net:,.0f}")
col4.metric("Data Range", f"{daily['date'].min().date()} → {daily['date'].max().date()}")

st.markdown("**Explanation:** These KPIs summarize recent cash activity. Use them to quickly assess funding pressure and working capital cycles.")

# -----------------------------
# Navigation
# -----------------------------
section = st.radio("Navigate", ["Dashboard", "Forecast", "Basel III (LCR)", "ALM Gap", "Market Risk & VaR", "Intraday Liquidity", "Optimizer", "Data Preview"], index=0)

# -----------------------------
# Dashboard
# -----------------------------
if section == "Dashboard":
    st.header("Executive Dashboard")
    st.markdown("Trend and distribution of inflows/outflows to guide funding decisions.")
    fig = px.line(daily, x="date", y=["inflow", "outflow", "net"], title="Daily Inflows / Outflows / Net")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.info("This time-series view helps spot structural shifts, seasonality, and abnormal spikes (e.g., quarter-ends).")

    # 30-day Waterfall
    last30 = daily.tail(30)
    wf_fig = go.Figure(go.Waterfall(
        name="30d",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Inflows", "Outflows", "Net"],
        y=[last30["inflow"].sum(), -last30["outflow"].sum(), last30["inflow"].sum() - last30["outflow"].sum()],
        text=[f"₹{last30['inflow'].sum():,.0f}", f"-₹{last30['outflow'].sum():,.0f}", f"₹{(last30['inflow'].sum() - last30['outflow'].sum()):,.0f}"]
    ))
    wf_fig.update_layout(title="30-Day Aggregate Waterfall")
    st.plotly_chart(wf_fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Waterfall shows the net position after aggregating inflows and outflows over the recent 30-day window — useful for short-term policy decisions.")

# -----------------------------
# Forecast (ML)
# -----------------------------
elif section == "Forecast":
    st.header("Cash Flow Forecast (Fast RF)")
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    st.markdown("Forecast uses a lightweight RandomForest over lag features (presentation-grade accuracy).")
    with st.spinner("Training / Forecasting..."):
        preds = rolling_rf_forecast(daily["net"], horizon=horizon)
    f_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=horizon)
    fc = pd.DataFrame({"date": f_dates, "forecast_net": preds})
    fig = px.line()
    fig.add_scatter(x=daily["date"], y=daily["net"], mode="lines", name="Historical")
    fig.add_scatter(x=fc["date"], y=fc["forecast_net"], mode="lines", name="Forecast")
    fig.update_layout(title=f"{horizon}-day Net Cash Forecast")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.success("Explanation: Forecast allows treasury to plan funding, short-term investments, and contingency buffers ahead of expected net outflow days.")
    st.download_button("Download Forecast CSV", data=fc.to_csv(index=False).encode("utf-8"), file_name="cash_forecast.csv")

# -----------------------------
# Basel III LCR
# -----------------------------
elif section == "Basel III (LCR)":
    st.header("Basel III — Liquidity Coverage Ratio (LCR)")
    st.markdown("Configure HQLA mix and inflow caps to test regulatory coverage.")

    c1, c2, c3 = st.columns(3)
    with c1:
        h1 = st.number_input("Level 1 HQLA (₹)", value=20_000_000_000.0, step=10_000_000.0, format="%.2f")
    with c2:
        h2a = st.number_input("Level 2A HQLA (₹)", value=5_000_000_000.0, step=10_000_000.0, format="%.2f")
    with c3:
        h2b = st.number_input("Level 2B HQLA (₹)", value=3_000_000_000.0, step=10_000_000.0, format="%.2f")
    inflow_cap_pct = st.slider("Inflow cap (% of outflows)", 50, 100, 75)
    inflow_cap = inflow_cap_pct / 100.0

    avg_out_30 = float(daily["outflow"].tail(30).mean())
    adj_hqla, net_out, lcr = compute_lcr(h1, h2a, h2b, avg_out_30, inflow_cap)
    k1, k2, k3 = st.columns(3)
    k1.metric("Adjusted HQLA", f"₹{adj_hqla:,.0f}")
    k2.metric("Net 30-day Outflows", f"₹{net_out:,.0f}")
    k3.metric("LCR (%)", f"{lcr:,.1f}%")
    st.info("Explanation: LCR >= 100% indicates sufficient high-quality liquid assets to absorb a 30-day stress scenario. Use sensitivity to test resilience.")

# -----------------------------
# ALM Gap (simple)
# -----------------------------
elif section == "ALM Gap":
    st.header("ALM Gap — Repricing Buckets")
    buckets = ["0-7d", "8-30d", "31-90d", "91-365d", ">1y"]
    defaults_rsa = [15e9, 25e9, 22e9, 18e9, 10e9]
    defaults_rsl = [12e9, 27e9, 25e9, 20e9, 8e9]

    st.markdown("Enter RSA (assets) and RSL (liabilities) by repricing bucket.")
    rsa = []
    rsl = []
    cols = st.columns(len(buckets))
    for i, b in enumerate(buckets):
        with cols[i]:
            rsa.append(st.number_input(f"RSA {b}", value=defaults_rsa[i], step=1e8, key=f"rsa_{i}"))
    cols = st.columns(len(buckets))
    for i, b in enumerate(buckets):
        with cols[i]:
            rsl.append(st.number_input(f"RSL {b}", value=defaults_rsl[i], step=1e8, key=f"rsl_{i}"))

    gap = np.array(rsa) - np.array(rsl)
    df_gap = pd.DataFrame({"Bucket": buckets, "GAP": gap, "Cumulative": np.cumsum(gap)})
    st.plotly_chart(px.bar(df_gap, x="Bucket", y="GAP", title="ALM GAP by Bucket"), use_container_width=True)
    st.plotly_chart(px.line(df_gap, x="Bucket", y="Cumulative", markers=True, title="Cumulative GAP"), use_container_width=True)
    st.info("Explanation: ALM gap highlights timing mismatches between asset and liability repricing — essential for interest-rate risk planning.")

# -----------------------------
# Market Risk & VaR
# -----------------------------
elif section == "Market Risk & VaR":
    st.header("Market Risk — VaR")
    cl = st.slider("Confidence level (%)", 90, 99, 99)
    mu = st.number_input("Mean daily return (μ)", -0.01, 0.01, 0.0003)
    sigma = st.number_input("Daily volatility (σ)", 0.001, 0.05, 0.015)
    notional = st.number_input("Notional (₹)", value=1_000_000_000.0, step=1e7)

    with st.spinner("Calculating VaR..."):
        mc_var = monte_carlo_var(mu, sigma, 1, 20000, cl/100, notional)

    st.metric(f"MC VaR ({cl}%)", f"₹{mc_var:,.0f}")
    st.info("VaR shows the maximum expected loss over 1 day at chosen confidence. Use alongside stress tests for robust risk management.")

# -----------------------------
# Intraday Liquidity
# -----------------------------
elif section == "Intraday Liquidity":
    st.header("Intraday Liquidity — Payments Analytics")
    if payments_df is None:
        st.info("No payments dataset loaded. Use 'Use enterprise v2 CSVs' or upload a payments CSV.")
    else:
        freq = st.selectbox("Bucket frequency", ["15min", "30min", "60min"], index=0)
        with st.spinner("Computing intraday cumulative curve..."):
            curve, peak = payments_intraday_curve(payments_df, freq)
        st.line_chart(curve)
        st.metric("Peak Net Cumulative Outflow", f"₹{peak:,.0f}")
        st.info("Peak net outflow indicates settlement corridor stress. Use for RTGS/clearing sizing and intraday liquidity buffers.")

# -----------------------------
# Optimizer
# -----------------------------
elif section == "Optimizer":
    st.header("Investment Optimizer — Yield vs Liquidity")
    total = st.number_input("Total funds (₹)", value=5_000_000_000.0, step=1e8)
    horizon = st.slider("Horizon (days)", 7, 90, 30)
    lcr_floor = st.slider("HQLA Coverage Floor (%)", 100, 200, 125)
    dur_cap = st.slider("Max weighted avg duration (days)", 7, 365, 120)
    y_cash = st.number_input("Cash yield", value=0.035)
    y_tb = st.number_input("T-Bills", value=0.064)
    y_gs = st.number_input("G-Secs", value=0.072)
    y_cp = st.number_input("CP/CD", value=0.078)

    yields = {"Cash": y_cash, "T-Bills": y_tb, "G-Secs": y_gs, "CP/CD": y_cp}
    liq_days = {"Cash": 0, "T-Bills": 30, "G-Secs": 180, "CP/CD": 90}
    haircuts = {"Cash": 1.0, "T-Bills": 0.85, "G-Secs": 0.85, "CP/CD": 0.75}

    with st.spinner("Running optimization..."):
        instr, res = optimize_portfolio(lcr_floor/100, horizon, yields, liq_days, haircuts, total, dur_cap)

    if res.success:
        alloc = pd.Series(res.x, index=instr)
        df_alloc = pd.DataFrame({"Instrument": instr, "Allocation (₹)": alloc, "Share (%)": (alloc / total) * 100})
        st.plotly_chart(px.pie(df_alloc, names="Instrument", values="Allocation (₹)", title="Optimal Allocation"), use_container_width=True)
        st.dataframe(df_alloc.style.format({"Allocation (₹)": "₹{:,.0f}", "Share (%)": "{:.2f}%"}))
        st.info("Optimizer output balances yield with liquidity constraints and LCR requirements — use scenario changes to evaluate policy trade-offs.")
    else:
        st.error("Optimization failed — relax constraints and retry.")

# -----------------------------
# Data Preview
# -----------------------------
elif section == "Data Preview":
    st.header("Data Preview & Diagnostics")
    if transactions_df is not None:
        st.subheader("Transactions (preview)")
        st.dataframe(transactions_df.head(max_tx_preview))
        st.markdown(f"Total transaction rows: **{len(transactions_df):,}**")
    else:
        st.info("No transaction-level dataset loaded (using sample).")

    if payments_df is not None:
        st.subheader("Payments (preview)")
        st.dataframe(payments_df.head(preview_pay_rows))
        st.markdown(f"Total payment rows: **{len(payments_df):,}**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Enterprise datasets v2 integrated • Optimized for high-volume treasury demonstrations")
