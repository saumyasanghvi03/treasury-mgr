# app.py — Treasury Control Tower (Ultra-Lite, Fast-Load)
# Focus: instant load, simple math, executive-ready visuals

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Treasury Control Tower — Ultra-Lite",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styles (optional)
# -----------------------------
def inject_css():
    css = """
    .stMetric { border:1px solid rgba(14,116,144,0.2); border-radius:12px; padding:8px 12px; }
    .block-container { padding-top:1rem; }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_css()

# -----------------------------
# Fast, Cached Data Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_sample_daily(days: int = 365, seed: int = 7) -> pd.DataFrame:
    """Tiny synthetic daily series for instant demo."""
    rng = np.random.default_rng(seed)
    start = (datetime.today().date() - timedelta(days=days - 1))
    dates = pd.date_range(start, periods=days, freq="D")
    inflow = 10_000_000 + 800_000 * np.sin(np.arange(days) / 10) + rng.normal(0, 350_000, days)
    outflow = 9_400_000 + 900_000 * np.cos(np.arange(days) / 13) + rng.normal(0, 360_000, days)
    df = pd.DataFrame({"date": dates, "inflow": inflow, "outflow": outflow})
    df["net"] = df["inflow"] - df["outflow"]
    return df

@st.cache_data(show_spinner=False)
def load_transactions_csv(path_or_file, nrows: int | None = None) -> pd.DataFrame:
    """Expect columns: date, amount, type (inflow/outflow)."""
    df = pd.read_csv(path_or_file, parse_dates=["date"], nrows=nrows)
    df.columns = [c.strip().lower() for c in df.columns]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower().str.strip()
    return df

@st.cache_data(show_spinner=False)
def aggregate_to_daily(df_txn: pd.DataFrame) -> pd.DataFrame:
    df = df_txn.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "type" in df.columns:
        inflows = df[df["type"].str.startswith("in", na=False)].groupby("date")["amount"].sum()
        outflows = df[df["type"].str.startswith("out", na=False)].groupby("date")["amount"].sum()
    else:
        inflows = df[df["amount"] > 0].groupby("date")["amount"].sum()
        outflows = -df[df["amount"] < 0].groupby("date")["amount"].sum()
    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D").date
    daily = pd.DataFrame({"date": all_days})
    daily["inflow"]  = daily["date"].map(inflows).fillna(0.0)
    daily["outflow"] = daily["date"].map(outflows).fillna(0.0)
    daily["net"] = daily["inflow"] - daily["outflow"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily

@st.cache_data(show_spinner=False)
def load_payments_csv(path_or_file, nrows: int | None = None) -> pd.DataFrame:
    """Expect columns: timestamp, amount, direction (in/out)."""
    df = pd.read_csv(path_or_file, parse_dates=["timestamp"], nrows=nrows)
    df.columns = [c.strip().lower() for c in df.columns]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.lower().str.strip()
    return df

# -----------------------------
# Ultra-Lite Analytics
# -----------------------------
def forecast_net_simple(series: pd.Series, horizon: int = 30, alpha: float = 0.2) -> np.ndarray:
    """
    Ultra-fast EWMA + weekly seasonal mean:
    forecast_t = EWMA + seasonal_mean(dow)
    """
    s = series.reset_index(drop=True).astype(float)
    # EWMA trend
    ewma = s.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    # Seasonal weekly mean by day-of-week
    dow = (np.arange(len(s)) % 7)
    df = pd.DataFrame({"y": s, "dow": dow})
    seas = df.groupby("dow")["y"].mean().reindex(range(7)).fillna(df["y"].mean())
    preds = []
    last_index = len(s) - 1
    for i in range(1, horizon + 1):
        d = (last_index + i) % 7
        preds.append(float(0.6 * ewma + 0.4 * seas[d]))
    return np.array(preds)

def compute_lcr(h1, h2a, h2b, avg_out_30d, inflow_cap_ratio=0.75):
    """Basel-lite: L1(100%), L2A(85%), L2B(75%), inflows capped."""
    hqla = h1 + 0.85 * h2a + 0.75 * h2b
    inflows = min(avg_out_30d * inflow_cap_ratio, avg_out_30d * 0.5)
    net_out = max(avg_out_30d - inflows, 0.0)
    lcr = (hqla / (net_out + 1e-9)) * 100 if net_out > 0 else np.inf
    return hqla, net_out, lcr

def mc_var(mu=0.0003, sigma=0.015, cl=0.99, notional=1.0, sims=10000):
    """Very fast Monte Carlo VaR."""
    ret = np.random.normal(mu, sigma, sims)
    q = np.quantile(ret, 1 - cl)
    return -q * notional

def intraday_curve(df_pay: pd.DataFrame, freq="15min"):
    d = df_pay.copy()
    if "timestamp" not in d.columns or "amount" not in d.columns:
        raise ValueError("Payments CSV must have 'timestamp' and 'amount'.")
    d["slot"] = d["timestamp"].dt.floor(freq)
    if "direction" in d.columns:
        sign = np.where(d["direction"].str.startswith("in"), 1, -1)
    else:
        sign = np.where(d["amount"] >= 0, 1, -1)
    curve = (d["amount"] * sign).groupby(d["slot"]).sum().sort_index().cumsum()
    peak_out = float(-curve.min())
    return curve, peak_out

def optimize_portfolio(hqla_min_ratio, horizon_days, yields, liq_days, haircuts,
                       total_funds=5e9, max_duration_days=120):
    instr = list(yields.keys())
    y = np.array([yields[k] for k in instr], float)
    dur = np.array([liq_days[k] for k in instr], float)
    hqla_adj = np.array([haircuts[k] for k in instr], float)

    # Constraints: duration, HQLA, full investment
    A_ub = [dur.tolist(), (-hqla_adj).tolist()]
    b_ub = [max_duration_days * total_funds,
            -hqla_min_ratio * (0.006 * total_funds * (horizon_days / 30.0))]
    A_eq = [np.ones(len(instr)).tolist()]
    b_eq = [total_funds]
    bounds = [(0, total_funds) for _ in instr]

    res = linprog(c=(-y), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    return instr, res

# -----------------------------
# Sidebar — Data Selection
# -----------------------------
st.sidebar.title("Data & Performance")
choice = st.sidebar.selectbox(
    "Dataset source",
    ["Built-in sample (fastest)", "Upload transactions.csv / payments.csv"]
)

tx_file = None
pay_file = None

if choice == "Upload transactions.csv / payments.csv":
    tx_file = st.sidebar.file_uploader("Upload transactions.csv (date, amount, type)", type=["csv"])
    pay_file = st.sidebar.file_uploader("Upload payments.csv (timestamp, amount, direction)", type=["csv"])

# -----------------------------
# Load Data (order safe)
# -----------------------------
if choice == "Built-in sample (fastest)":
    daily = load_sample_daily()
    payments_df = None
else:
    if tx_file is None:
        st.info("Upload transactions.csv to continue, or switch to Built-in sample.")
        st.stop()
    tx_df = load_transactions_csv(tx_file)
    daily = aggregate_to_daily(tx_df)
    payments_df = load_payments_csv(pay_file) if pay_file is not None else None

# -----------------------------
# KPIs
# -----------------------------
st.title("Treasury Control Tower — Ultra-Lite")
last30 = daily.tail(30)
avg_in = float(last30["inflow"].mean())
avg_out = float(last30["outflow"].mean())
avg_net = float(last30["net"].mean())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg Inflow (30d)", f"₹{avg_in:,.0f}")
k2.metric("Avg Outflow (30d)", f"₹{avg_out:,.0f}")
k3.metric("Avg Net (30d)", f"₹{avg_net:,.0f}")
k4.metric("Data Horizon", f"{daily['date'].min().date()} → {daily['date'].max().date()}")
st.caption("KPIs snapshot recent liquidity conditions to guide funding posture and overnight placement strategy.")

# -----------------------------
# Navigation
# -----------------------------
section = st.radio("Navigate", [
    "Dashboard",
    "Forecast (Ultra-fast)",
    "Basel III (LCR)",
    "Risk (VaR)",
    "Intraday Liquidity",
    "Optimizer",
    "Data Preview"
], index=0, horizontal=True)

# -----------------------------
# Dashboard
# -----------------------------
if section == "Dashboard":
    st.subheader("Executive Dashboard")
    fig = px.line(daily, x="date", y=["inflow", "outflow", "net"],
                  title="Daily Inflows / Outflows / Net (₹)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Trend view highlights seasonality and spikes; use for short-term liquidity planning and variance analysis.")

    # 30-day Waterfall
    w_in, w_out = last30["inflow"].sum(), last30["outflow"].sum()
    wf = go.Figure(go.Waterfall(
        name="30d", orientation="v",
        measure=["relative", "relative", "total"],
        x=["Inflows", "Outflows", "Net"],
        y=[w_in, -w_out, w_in - w_out],
        text=[f"₹{w_in:,.0f}", f"-₹{w_out:,.0f}", f"₹{(w_in-w_out):,.0f}"]
    ))
    wf.update_layout(title="30-Day Aggregate Waterfall")
    st.plotly_chart(wf, use_container_width=True, config={"displayModeBar": False})
    st.caption("Waterfall reconciles 30-day inflows and outflows to net effect, enabling crisp communication with ALCO.")

# -----------------------------
# Forecast (Ultra-fast)
# -----------------------------
elif section == "Forecast (Ultra-fast)":
    st.subheader("Net Cash Forecast — EWMA + Weekly Seasonality")
    horizon = st.slider("Horizon (days)", 7, 60, 30)
    with st.spinner("Projecting…"):
        preds = forecast_net_simple(daily["net"], horizon=horizon, alpha=0.25)
    f_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=horizon)
    fc = pd.DataFrame({"date": f_dates, "forecast_net": preds})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["net"], name="Historical"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_net"], name="Forecast"))
    fig.update_layout(title="Projected Net Liquidity (₹)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Ultra-fast heuristic forecast for day-ahead positioning. Swap in your institutional model later without changing the UI.")
    st.download_button("Download forecast.csv", fc.to_csv(index=False).encode("utf-8"), "forecast.csv")

# -----------------------------
# Basel III (LCR)
# -----------------------------
elif section == "Basel III (LCR)":
    st.subheader("Basel III — Liquidity Coverage Ratio")
    c1, c2, c3 = st.columns(3)
    with c1:
        h1 = st.number_input("Level 1 HQLA (₹)", min_value=0.0, value=20_000_000_000.0, step=10_000_000.0, format="%.2f")
    with c2:
        h2a = st.number_input("Level 2A HQLA (₹)", min_value=0.0, value=5_000_000_000.0, step=10_000_000.0, format="%.2f")
    with c3:
        h2b = st.number_input("Level 2B HQLA (₹)", min_value=0.0, value=3_000_000_000.0, step=10_000_000.0, format="%.2f")
    inflow_cap = st.slider("Inflow cap (% of outflows)", 50, 100, 75) / 100.0

    avg_out_30d = float(last30["outflow"].mean())
    hqla, net_out, lcr = compute_lcr(h1, h2a, h2b, avg_out_30d, inflow_cap)
    m1, m2, m3 = st.columns(3)
    m1.metric("Adjusted HQLA", f"₹{hqla:,.0f}")
    m2.metric("Net Outflows (30d)", f"₹{net_out:,.0f}")
    m3.metric("LCR (%)", f"{lcr:,.1f}%")
    st.caption("LCR ≥ 100% indicates sufficient HQLA to survive a 30-day stress scenario under Basel. Use sliders to test structural resilience.")

# -----------------------------
# Risk (VaR)
# -----------------------------
elif section == "Risk (VaR)":
    st.subheader("Market Risk — 1-Day VaR (Monte Carlo)")
    cl = st.slider("Confidence Level (%)", 90, 99, 99)
    vol = st.number_input("Daily Volatility (σ)", min_value=0.001, max_value=0.05, value=0.015, step=0.001)
    mu = st.number_input("Mean Daily Return (μ)", min_value=-0.01, max_value=0.01, value=0.0003, step=0.0001)
    notional = st.number_input("Notional (₹)", min_value=0.0, value=1_000_000_000.0, step=10_000_000.0, format="%.2f")

    var = mc_var(mu=mu, sigma=vol, cl=cl/100, notional=notional, sims=10000)
    st.metric(f"VaR ({cl}%, 1-day)", f"₹{var:,.0f}")
    st.caption("VaR estimates worst expected loss over one day at the chosen confidence, under normal conditions. Use with stress tests for governance.")

# -----------------------------
# Intraday Liquidity
# -----------------------------
elif section == "Intraday Liquidity":
    st.subheader("Intraday Liquidity — Cumulative Net Curve")
    if choice == "Built-in sample (fastest)" or (payments_df is None):
        st.info("Upload a payments.csv (timestamp, amount, direction) in Sidebar to view intraday analytics.")
    else:
        freq = st.selectbox("Aggregation frequency", ["15min", "30min", "60min"], index=0)
        with st.spinner("Computing curve…"):
            curve, peak = intraday_curve(payments_df, freq=freq)
        st.line_chart(curve)
        st.metric("Peak Net Cumulative Outflow", f"₹{peak:,.0f}")
        st.caption("Peak outflow indicates RTGS/settlement corridor stress and informs intraday liquidity buffers.")

# -----------------------------
# Optimizer
# -----------------------------
elif section == "Optimizer":
    st.subheader("Investment Optimizer — Yield vs Liquidity vs LCR")
    total = st.number_input("Total Funds (₹)", min_value=0.0, value=5_000_000_000.0, step=10_000_000.0, format="%.2f")
    horizon = st.slider("Liquidity Horizon (days)", 7, 60, 30)
    lcr_floor = st.slider("HQLA Coverage Floor (%)", 100, 200, 125)
    dur_cap = st.slider("Max Weighted Avg Duration (days)", 7, 365, 120)

    c1, c2, c3, c4 = st.columns(4)
    y_cash = c1.number_input("Cash Yield", value=0.035, step=0.001, format="%.3f")
    y_tb   = c2.number_input("T-Bills Yield", value=0.064, step=0.001, format="%.3f")
    y_gs   = c3.number_input("G-Secs Yield", value=0.072, step=0.001, format="%.3f")
    y_cp   = c4.number_input("CP/CD Yield", value=0.078, step=0.001, format="%.3f")

    yields = {"Cash": y_cash, "T-Bills": y_tb, "G-Secs": y_gs, "CP/CD": y_cp}
    liq_days = {"Cash": 0, "T-Bills": 30, "G-Secs": 180, "CP/CD": 90}
    haircuts = {"Cash": 1.00, "T-Bills": 0.85, "G-Secs": 0.85, "CP/CD": 0.75}

    with st.spinner("Optimizing…"):
        instr, res = optimize_portfolio(lcr_floor/100, horizon, yields, liq_days, haircuts, total, dur_cap)

    if res.success:
        alloc = pd.Series(res.x, index=instr)
        df_alloc = pd.DataFrame({
            "Instrument": instr,
            "Allocation (₹)": alloc,
            "Share (%)": (alloc/total)*100
        })
        st.plotly_chart(px.pie(df_alloc, names="Instrument", values="Allocation (₹)", title="Optimal Allocation Mix"),
                        use_container_width=True, config={"displayModeBar": False})
        st.dataframe(df_alloc.style.format({"Allocation (₹)": "₹{:,.0f}", "Share (%)": "{:.2f}%"}), use_container_width=True)
        exp_yield = float(np.dot([yields[k] for k in instr], res.x) / total)
        st.success(f"Expected Annual Yield: {exp_yield:.2%}")
        st.caption("Allocation maximizes yield within LCR and duration guardrails. Adjust sliders to explore trade-offs.")
    else:
        st.error("Optimization infeasible — relax constraints and retry.")

# -----------------------------
# Data Preview
# -----------------------------
else:
    st.subheader("Data Preview")
    st.dataframe(daily.tail(100), use_container_width=True)
    st.caption("Preview of the daily series used across modules. Upload CSVs in the sidebar for your own data.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Ultra-Lite edition • Fast on Streamlit Cloud • Swap in institutional models without changing UI")
