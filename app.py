# app.py — Treasury Control Tower (Ultra-Lite+ Advanced Edition)
# Fast loading • Enterprise-grade UX • Innovative Treasury Insights

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog

# ----------------------------------
# Streamlit Page Setup
# ----------------------------------
st.set_page_config(
    page_title="Treasury Control Tower — Ultra-Lite+",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------
# Light Styling
# ----------------------------------
def inject_css():
    css = """
    .stMetric { border:1px solid rgba(0,105,150,0.25); border-radius:12px; padding:10px!important; }
    .block-container { padding-top:1rem; }
    h1,h2,h3,h4 { color:#0f4c81; }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_css()

# ----------------------------------
# Cached Data Utilities
# ----------------------------------
@st.cache_data(show_spinner=False)
def load_sample_daily(days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime.today().date() - timedelta(days=days - 1)
    dates = pd.date_range(start, periods=days, freq="D")

    season = np.sin(np.arange(days) / np.pi)
    inflow = 9_000_000 + 800_000 * season + rng.normal(0, 250_000, days)
    outflow = 8_400_000 + 700_000 * np.cos(np.arange(days) / 17) + rng.normal(0, 260_000, days)

    df = pd.DataFrame({"date": dates, "inflow": inflow, "outflow": outflow})
    df["net"] = df["inflow"] - df["outflow"]
    return df

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_transactions_csv(path_or_buf, nrows=None) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf, parse_dates=["date"], nrows=nrows)
    df.columns = df.columns.str.lower().str.strip()

    # Normalize amount column
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # Robust type column handling
    if "type" in df.columns:
        df["type"] = (
            df["type"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"credit": "inflow", "debit": "outflow"})
        )
    else:
        # If type column absent → infer using amount sign
        df["type"] = np.where(df["amount"] >= 0, "inflow", "outflow")

    return df


@st.cache_data(show_spinner=False)
def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    inflow = df[df["amount"] > 0].groupby("date")["amount"].sum()
    outflow = -df[df["amount"] < 0].groupby("date")["amount"].sum()

    span = pd.date_range(df["date"].min(), df["date"].max(), freq="D").date
    daily = pd.DataFrame({"date": span})
    daily["inflow"] = daily["date"].map(inflow).fillna(0.0)
    daily["outflow"] = daily["date"].map(outflow).fillna(0.0)
    daily["net"] = daily["inflow"] - daily["outflow"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily

@st.cache_data(show_spinner=False)
def load_payments_csv(path_or_buf, nrows=None) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf, parse_dates=["timestamp"], nrows=nrows)
    df.columns = df.columns.str.lower().str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df

# ----------------------------------
# Analytics Modules
# ----------------------------------
def forecast_simple(series: pd.Series, horizon: int = 30) -> np.ndarray:
    s = series.reset_index(drop=True)
    trend = s.ewm(alpha=0.2, adjust=False).mean().iloc[-1]
    dow_means = s.groupby(np.arange(len(s)) % 7).mean()

    preds = []
    for i in range(horizon):
        d = (len(s) + i) % 7
        preds.append(0.65 * trend + 0.35 * dow_means[d])
    return np.array(preds)

def compute_lcr(h1, h2a, h2b, avg_outflow, inflow_cap=0.75):
    adj = h1 + 0.85*h2a + 0.75*h2b
    inflow = min(avg_outflow * inflow_cap, avg_outflow * 0.5)
    net = max(avg_outflow - inflow, 0)
    return adj, net, (adj / (net + 1e-9)) * 100

def mc_var(mu=0.0003, sigma=0.015, cl=0.99, notional=1e9, sims=5000):
    ret = np.random.normal(mu, sigma, sims)
    q = np.quantile(ret, 1 - cl)
    return -q * notional

def intraday_curve(df: pd.DataFrame, freq="15min"):
    d = df.copy()
    d["slot"] = d["timestamp"].dt.floor(freq)
    sign = np.where(d.get("direction", "in").str.startswith("in"), 1, -1)
    curve = (d["amount"] * sign).groupby(d["slot"]).sum().sort_index().cumsum()
    return curve, float(-curve.min())

def optimize_portfolio(hqla_min_ratio, horizon_days, yields, liq_days,
                       haircuts, total_funds, max_dur):
    instr = list(yields.keys())
    y = np.array([yields[i] for i in instr])
    dur = np.array([liq_days[i] for i in instr], float)
    hqla_adj = np.array([haircuts[i] for i in instr], float)

    projected_outflow = 0.006 * total_funds * (horizon_days / 30)
    required = hqla_min_ratio * projected_outflow

    A_ub = [
        dur.tolist(),
        (-hqla_adj).tolist(),
    ]
    b_ub = [
        max_dur * total_funds,
        -required,
    ]

    A_eq = [np.ones(len(instr)).tolist()]
    b_eq = [total_funds]

    bounds = [(0, total_funds)] * len(instr)

    res = linprog(
        c=-y,
        A_ub=np.array(A_ub), b_ub=np.array(b_ub),
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=bounds, method="highs"
    )
    return instr, res

# ----------------------------------
# Sidebar – Data Mode
# ----------------------------------
st.sidebar.title("⚙ Data Mode")
mode = st.sidebar.radio("Select Data Source", [
    "Built-in Sample (Fastest)",
    "Upload CSVs"
])

tx_file, pay_file = None, None

if mode == "Upload CSVs":
    tx_file = st.sidebar.file_uploader("Transactions CSV", type=["csv"])
    pay_file = st.sidebar.file_uploader("Payments CSV", type=["csv"])

# ----------------------------------
# Load Data
# ----------------------------------
if mode == "Built-in Sample (Fastest)":
    daily = load_sample_daily()
    payments_df = None
else:
    if tx_file is None:
        st.warning("Upload transactions.csv to proceed.")
        st.stop()
    tx_df = load_transactions_csv(tx_file)
    daily = aggregate_to_daily(tx_df)
    payments_df = load_payments_csv(pay_file) if pay_file else None

# ----------------------------------
# KPIs
# ----------------------------------
st.title("💠 Treasury Control Tower — Ultra-Lite+ Advanced")
last30 = daily.tail(30)

avg_in  = last30["inflow"].mean()
avg_out = last30["outflow"].mean()
avg_net = last30["net"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Inflow (30d)", f"₹{avg_in:,.0f}")
c2.metric("Avg Outflow (30d)", f"₹{avg_out:,.0f}")
c3.metric("Avg Net (30d)", f"₹{avg_net:,.0f}")
c4.metric("Data Range", f"{daily['date'].min().date()} → {daily['date'].max().date()}")

# Liquidity Alert
if avg_net < 0:
    st.error("⚠ Liquidity Warning: Negative average net flows. Funding stress elevated.")
elif avg_net < avg_out * 0.1:
    st.warning("⚠ Low Surplus: Net liquidity thin. Monitor volatility.")
else:
    st.success("✅ Liquidity Stable: Surplus flows supportive.")

# ----------------------------------
# Navigation
# ----------------------------------
section = st.radio("Navigate", [
    "Dashboard",
    "AI Forecast",
    "Basel III LCR",
    "Market Risk (VaR)",
    "Intraday Liquidity",
    "Investment Optimizer",
    "Data Preview"
], horizontal=True)

# ----------------------------------
# Dashboard
# ----------------------------------
if section == "Dashboard":
    st.subheader("📊 Executive Liquidity Dashboard")

    fig = px.line(daily, x="date", y=["inflow", "outflow", "net"],
                  title="Daily Liquidity Profile")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Shows long-term liquidity rhythm, seasonality, and operational patterns.")

    w_in, w_out = last30["inflow"].sum(), last30["outflow"].sum()
    wf = go.Figure(go.Waterfall(
        measure=["relative", "relative", "total"],
        x=["Inflows", "Outflows", "Net"],
        y=[w_in, -w_out, w_in - w_out]
    ))
    wf.update_layout(title="30-Day Liquidity Reconciliation")
    st.plotly_chart(wf, use_container_width=True)

# ----------------------------------
# AI Forecast
# ----------------------------------
elif section == "AI Forecast":
    st.subheader("🤖 Cash Flow Forecast (EWMA + Seasonality)")

    horizon = st.slider("Forecast Horizon (days)", 7, 60, 30)
    preds = forecast_simple(daily["net"], horizon)

    f_dates = pd.date_range(daily["date"].max() + timedelta(days=1), periods=horizon)
    fc = pd.DataFrame({"date": f_dates, "forecast": preds})

    fig = px.line()
    fig.add_scatter(x=daily["date"], y=daily["net"], name="Historical")
    fig.add_scatter(x=fc["date"], y=fc["forecast"], name="Forecast")
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Forecast blends trend + weekly seasonality for fast decision support.")

# ----------------------------------
# Basel III LCR
# ----------------------------------
elif section == "Basel III LCR":
    st.subheader("🛡 Basel III – Liquidity Coverage Ratio")

    l1 = st.number_input("Level 1 HQLA", value=20_000_000_000.0)
    l2a = st.number_input("Level 2A HQLA", value=5_000_000_000.0)
    l2b = st.number_input("Level 2B HQLA", value=3_000_000_000.0)
    inflow_cap = st.slider("Inflow Cap (%)", 50, 100, 75) / 100

    avg_out = last30["outflow"].mean()
    adj, net, ratio = compute_lcr(l1, l2a, l2b, avg_out, inflow_cap)

    c1, c2, c3 = st.columns(3)
    c1.metric("Adjusted HQLA", f"₹{adj:,.0f}")
    c2.metric("Net Outflows", f"₹{net:,.0f}")
    c3.metric("LCR (%)", f"{ratio:,.1f}%")

# ----------------------------------
# Market Risk (VaR)
# ----------------------------------
elif section == "Market Risk (VaR)":
    st.subheader("⚠ Market Risk – VaR")

    cl = st.slider("Confidence Level (%)", 90, 99, 99)
    sigma = st.number_input("Volatility (σ)", value=0.015)
    mu = st.number_input("Mean Return (μ)", value=0.0003)
    notional = st.number_input("Notional (₹)", value=1_000_000_000.0)

    var = mc_var(mu, sigma, cl/100, notional)

    st.metric(f"VaR ({cl}%)", f"₹{var:,.0f}")
    st.caption("Shows maximum expected 1-day loss under normal conditions.")

# ----------------------------------
# Intraday Liquidity
# ----------------------------------
elif section == "Intraday Liquidity":
    st.subheader("⏱ Intraday Liquidity Analysis")

    if payments_df is None:
        st.info("Upload payments.csv to enable intraday analytics.")
    else:
        freq = st.selectbox("Time Bucket", ["15min", "30min", "60min"])
        curve, peak = intraday_curve(payments_df, freq)

        st.line_chart(curve)
        st.metric("Peak Outflow", f"₹{peak:,.0f}")

# ----------------------------------
# Optimizer
# ----------------------------------
elif section == "Investment Optimizer":
    st.subheader("🧠 Optimal Treasury Allocation")

    total = st.number_input("Total Funds", value=5_000_000_000.0)
    horizon = st.slider("Liquidity Horizon (days)", 7, 60, 30)
    lcr_floor = st.slider("Min HQLA Coverage (%)", 100, 200, 125)
    dur_cap = st.slider("Max Duration (days)", 30, 365, 120)

    y_cash = st.number_input("Cash Yield", value=0.03)
    y_tb = st.number_input("T-Bills Yield", value=0.06)
    y_gs = st.number_input("G-Secs Yield", value=0.07)
    y_cp = st.number_input("CP/CD Yield", value=0.075)

    yields = {"Cash": y_cash, "T-Bills": y_tb, "G-Secs": y_gs, "CP/CD": y_cp}
    liq_days = {"Cash": 0, "T-Bills": 30, "G-Secs": 180, "CP/CD": 90}
    hair = {"Cash": 1.0, "T-Bills": 0.85, "G-Secs": 0.85, "CP/CD": 0.75}

    instr, res = optimize_portfolio(lcr_floor/100, horizon, yields, liq_days, hair, total, dur_cap)

    if res.success:
        alloc = pd.Series(res.x, index=instr)
        df_alloc = pd.DataFrame({
            "Instrument": instr,
            "Allocation": alloc,
            "Share (%)": (alloc/total)*100
        })
        st.dataframe(df_alloc.style.format({"Allocation": "₹{:,.0f}", "Share (%)": "{:.2f}%"}))
        st.success(f"Expected Yield: {np.dot(res.x, [yields[k] for k in instr])/total:.2%}")
    else:
        st.error("Allocation infeasible — relax constraints.")

# ----------------------------------
# Data Preview
# ----------------------------------
else:
    st.subheader("🗂 Data Preview")
    st.dataframe(daily)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Ultra-Lite+ edition • Super fast • Boardroom-ready • Designed by ChatGPT for Saumya Sanghvi")
