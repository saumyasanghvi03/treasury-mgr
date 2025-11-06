# ------------------------------------------------------------
# Treasury Management Solution – Advanced (Optimized + Explained)
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linprog

# ------------------------------------------------------------
# Streamlit Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Treasury Management Solution – Advanced",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Caching Layers (Speed Optimizations)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sample_data(days: int = 365, seed: int = 42):
    """Generate synthetic inflow/outflow transactions for demonstration."""
    rng = np.random.default_rng(seed)
    start = datetime.today().date() - timedelta(days=days - 1)
    dates = pd.date_range(start, periods=days, freq="D")
    inflow = 10_000_000 + rng.normal(0, 500_000, days)
    outflow = 9_500_000 + rng.normal(0, 500_000, days)
    df = pd.DataFrame({"date": dates, "inflow": inflow, "outflow": outflow})
    df["net"] = df["inflow"] - df["outflow"]
    return df

@st.cache_resource(show_spinner=False)
def fit_rf_model(series: pd.Series):
    """Fit a lightweight RandomForest model for cash forecasting."""
    df = pd.DataFrame({"y": series})
    for l in (1, 2, 3, 7):
        df[f"lag_{l}"] = series.shift(l)
    df["dow"] = np.arange(len(series)) % 7
    df = df.dropna()
    X, y = df.drop(columns=["y"]), df["y"]
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=13, n_jobs=-1)
    model.fit(X, y)
    return model

def rf_forecast(series: pd.Series, horizon: int = 30):
    """Generate forecast using cached RF model."""
    hist = series.reset_index(drop=True)
    model = fit_rf_model(hist)
    preds, last = [], hist.copy()
    for _ in range(horizon):
        df = pd.DataFrame({"y": last})
        for l in (1, 2, 3, 7):
            df[f"lag_{l}"] = last.shift(l)
        df["dow"] = np.arange(len(last)) % 7
        yhat = model.predict(df.drop(columns=["y"]).iloc[[-1]])[0]
        preds.append(yhat)
        last = pd.concat([last, pd.Series([yhat])], ignore_index=True)
    return np.array(preds)

# ------------------------------------------------------------
# Treasury Computations
# ------------------------------------------------------------
def compute_lcr(h1, h2a, h2b, avg_out, inflow_cap=0.75):
    hqla = h1 + 0.85 * h2a + 0.75 * h2b
    inflows = min(avg_out * inflow_cap, avg_out * 0.5)
    net_out = max(avg_out - inflows, 0)
    lcr = (hqla / (net_out + 1e-9)) * 100
    return hqla, net_out, lcr

def monte_carlo_var(mu=0.0003, sigma=0.015, cl=0.99, notional=1.0):
    ret = np.random.normal(mu, sigma, 20_000)
    return -np.quantile(ret, 1 - cl) * notional

def optimize_investment(hqla_ratio, horizon, yields, liq_days, haircuts, total=5e9, dur_cap=120):
    instr = list(yields.keys())
    y = np.array(list(yields.values()))
    dur = np.array([liq_days[k] for k in instr])
    hqla_adj = np.array([haircuts[k] for k in instr])
    A_ub = [dur.tolist(), (-hqla_adj).tolist()]
    b_ub = [dur_cap * total, -hqla_ratio * 0.006 * total * (horizon / 30)]
    A_eq = [np.ones(len(instr)).tolist()]
    b_eq = [total]
    res = linprog(-y, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, total)]*len(instr), method="highs")
    return instr, res

# ------------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------------
st.sidebar.title("💠 Treasury Management Control Tower")
section = st.sidebar.radio(
    "Navigate",
    ["Executive Dashboard", "Cash Flow Forecast", "Basel III LCR", "Market Risk (VaR)", "Investment Optimizer", "About"],
)

use_sample = st.sidebar.checkbox("Use Sample Data", True)
if use_sample:
    daily = load_sample_data()
else:
    uploaded = st.sidebar.file_uploader("Upload transactions.csv", type="csv")
    if uploaded:
        daily = pd.read_csv(uploaded)
    else:
        st.sidebar.warning("Please upload a transactions.csv file.")
        st.stop()

daily["date"] = pd.to_datetime(daily["date"])

# ------------------------------------------------------------
# Executive Dashboard
# ------------------------------------------------------------
if section == "Executive Dashboard":
    st.title("💠 Executive Dashboard")
    st.caption("A holistic view of treasury cash flows and liquidity trends.")

    avg_in, avg_out, avg_net = daily.tail(30)[["inflow", "outflow", "net"]].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Inflow (30d)", f"₹{avg_in:,.0f}")
    c2.metric("Avg Outflow (30d)", f"₹{avg_out:,.0f}")
    c3.metric("Net Position (30d)", f"₹{avg_net:,.0f}")

    fig = px.line(daily, x="date", y=["inflow", "outflow", "net"], title="Daily Cash Flow Movements")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.info("💡 This chart visualizes liquidity patterns across days, enabling early identification of structural imbalances between inflows and outflows.")

# ------------------------------------------------------------
# Cash Flow Forecast
# ------------------------------------------------------------
elif section == "Cash Flow Forecast":
    st.title("📈 ML Cash Flow Forecast")
    horizon = st.slider("Forecast Horizon (days)", 7, 60, 30)
    with st.spinner("Generating forecast..."):
        preds = rf_forecast(daily["net"], horizon=horizon)

    future_dates = pd.date_range(daily["date"].iloc[-1] + timedelta(days=1), periods=horizon)
    fc = pd.DataFrame({"date": future_dates, "forecast_net": preds})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["net"], name="Historical"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_net"], name="Forecast"))
    fig.update_layout(title="Forecasted Net Cash Position (₹)")
    st.plotly_chart(fig, use_container_width=True)
    st.success("🔍 The forecast projects future liquidity positions using historical net flows. It helps treasury teams plan short-term funding and placements efficiently.")

# ------------------------------------------------------------
# Basel III Liquidity Coverage Ratio
# ------------------------------------------------------------
elif section == "Basel III LCR":
    st.title("🧮 Basel III Liquidity Coverage Ratio (LCR)")
    st.caption("Assess regulatory liquidity compliance under stress scenarios.")

    h1 = st.number_input("Level 1 HQLA (₹)", 0.0, 5e10, 2e10, step=1e8)
    h2a = st.number_input("Level 2A HQLA (₹)", 0.0, 5e10, 5e9, step=1e8)
    h2b = st.number_input("Level 2B HQLA (₹)", 0.0, 5e10, 3e9, step=1e8)
    inflow_cap = st.slider("Inflow Cap (% of Outflows)", 50, 100, 75) / 100
    avg_out = daily["outflow"].tail(30).mean()
    hqla, net_out, lcr = compute_lcr(h1, h2a, h2b, avg_out, inflow_cap)

    c1, c2, c3 = st.columns(3)
    c1.metric("Adjusted HQLA", f"₹{hqla:,.0f}")
    c2.metric("Net 30-day Outflow", f"₹{net_out:,.0f}")
    c3.metric("LCR (%)", f"{lcr:,.1f}%")
    st.info("💬 LCR ≥ 100% indicates compliance. This ratio evaluates whether the bank maintains sufficient HQLA to withstand 30 days of stressed net outflows.")

# ------------------------------------------------------------
# Market Risk (VaR)
# ------------------------------------------------------------
elif section == "Market Risk (VaR)":
    st.title("⚠️ Market Risk – Value at Risk (VaR)")
    st.caption("Estimate potential daily loss under normal market volatility.")

    cl = st.slider("Confidence Level (%)", 90, 99, 99)
    sigma = st.number_input("Volatility (σ)", 0.001, 0.05, 0.015)
    mu = st.number_input("Mean Return (μ)", -0.01, 0.01, 0.0003)
    notional = st.number_input("Notional Exposure (₹)", 0.0, 1e10, 1e9, step=1e7)

    mc = monte_carlo_var(mu, sigma, cl / 100, notional)
    c1, c2 = st.columns(2)
    c1.metric(f"Monte Carlo VaR ({cl}%)", f"₹{mc:,.0f}")
    st.info("📊 VaR represents the maximum expected loss at the chosen confidence level. It helps measure downside risk exposure in trading or investment portfolios.")

# ------------------------------------------------------------
# Investment Optimizer
# ------------------------------------------------------------
elif section == "Investment Optimizer":
    st.title("🧠 Investment Optimizer – Yield vs Liquidity vs LCR")
    st.caption("Optimize portfolio allocations subject to liquidity and duration constraints.")

    total = st.number_input("Total Treasury Funds (₹)", 0.0, 1e10, 5e9, step=1e8)
    horizon = st.slider("Liquidity Horizon (days)", 7, 60, 30)
    lcr_floor = st.slider("Min HQLA Coverage (%)", 100, 200, 125)
    dur_cap = st.slider("Max Duration (days)", 7, 365, 120)
    y_cash = st.number_input("Cash Yield", 0.0, 0.2, 0.035)
    y_tb = st.number_input("T-Bills Yield", 0.0, 0.2, 0.064)
    y_gs = st.number_input("G-Secs Yield", 0.0, 0.2, 0.072)
    y_cp = st.number_input("CP/CD Yield", 0.0, 0.2, 0.078)

    yields = {"Cash": y_cash, "T-Bills": y_tb, "G-Secs": y_gs, "CP/CD": y_cp}
    liq_days = {"Cash": 0, "T-Bills": 30, "G-Secs": 180, "CP/CD": 90}
    haircuts = {"Cash": 1.0, "T-Bills": 0.85, "G-Secs": 0.85, "CP/CD": 0.75}

    with st.spinner("Optimizing portfolio…"):
        instr, res = optimize_investment(lcr_floor / 100, horizon, yields, liq_days, haircuts, total, dur_cap)
    if res.success:
        alloc = pd.Series(res.x, index=instr)
        df_alloc = pd.DataFrame({"Instrument": instr, "Allocation (₹)": alloc, "Share (%)": (alloc / total) * 100})
        st.plotly_chart(px.pie(df_alloc, names="Instrument", values="Allocation (₹)", title="Optimal Portfolio Mix"),
                        use_container_width=True, config={"displayModeBar": False})
        st.success(f"Expected Annual Yield: {np.dot(list(yields.values()), res.x) / total:.2%}")
        st.info("💼 The optimizer maximizes yield while meeting liquidity and LCR constraints. It aligns asset selection with risk appetite and regulatory requirements.")
    else:
        st.error("Optimization failed – adjust constraints and retry.")

# ------------------------------------------------------------
# About
# ------------------------------------------------------------
else:
    st.title("ℹ️ About This Application")
    st.markdown("""
This app demonstrates an **AI-enabled Treasury Management Solution** integrating:
- Machine Learning for cash flow forecasting  
- Basel III LCR and liquidity risk metrics  
- Market risk quantification via VaR  
- Yield optimization through constrained optimization  

Built with **Streamlit, Plotly, and scikit-learn**, optimized for MBA and banking demonstrations.
""")
