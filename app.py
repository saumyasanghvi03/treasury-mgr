# ------------------------------------------------------------
# Treasury Management Solution – Optimized + Explained Edition
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy.optimize import linprog

# ------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Treasury Management Solution – Advanced (Optimized)",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional CSS injection
def inject_css():
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass
inject_css()

# ------------------------------------------------------------
# Cached Utility Functions
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sample_data(days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generates demo inflow/outflow dataset."""
    rng = np.random.default_rng(seed)
    start = datetime.today().date() - timedelta(days=days - 1)
    dates = pd.date_range(start, periods=days, freq="D")
    inflow = 10_000_000 + rng.normal(0, 500_000, days)
    outflow = 9_500_000 + rng.normal(0, 500_000, days)
    df = pd.DataFrame({"date": dates, "inflow": inflow, "outflow": outflow})
    df["net"] = df["inflow"] - df["outflow"]
    return df

def lag_features(series: pd.Series, lags=(1, 2, 3, 7)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for l in lags:
        df[f"lag_{l}"] = series.shift(l)
    df["dow"] = np.arange(len(series)) % 7
    return df.dropna()

@st.cache_resource(show_spinner=False)
def fit_rf_model(series: pd.Series):
    df = lag_features(series)
    X, y = df.drop(columns=["y"]), df["y"]
    model = RandomForestRegressor(n_estimators=120, max_depth=6,
                                  random_state=13, n_jobs=-1)
    model.fit(X, y)
    return model

def rf_forecast(series: pd.Series, horizon: int = 30):
    """Fast rolling forecast using cached RF model."""
    hist = series.reset_index(drop=True)
    model = fit_rf_model(hist)
    preds, last = [], hist.copy()
    for _ in range(horizon):
        tmp = lag_features(last).iloc[-1:]
        yhat = model.predict(tmp.drop(columns=["y"]))[0]
        preds.append(yhat)
        last = pd.concat([last, pd.Series([yhat])], ignore_index=True)
    return np.array(preds)

# ------------------------------------------------------------
# Finance Computations
# ------------------------------------------------------------
def compute_lcr(h1, h2a, h2b, avg_out, inflow_cap=0.75):
    hqla = h1 + 0.85 * h2a + 0.75 * h2b
    inflows = min(avg_out * inflow_cap, avg_out * 0.5)
    net_out = max(avg_out - inflows, 0)
    lcr = (hqla / (net_out + 1e-9)) * 100
    return hqla, net_out, lcr

def historical_var(returns: np.ndarray, cl=0.99, notional=1.0):
    if len(returns) < 50: return None
    return -np.quantile(returns, 1 - cl) * notional

def monte_carlo_var(mu=0.0003, sigma=0.015, cl=0.99, notional=1.0):
    ret = np.random.normal(mu, sigma, 20_000)
    return -np.quantile(ret, 1 - cl) * notional

def optimize_investment(hqla_ratio, horizon, yields, liq_days, haircuts,
                        total=5e9, dur_cap=120):
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
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("💠 Treasury Management – Control Tower")
section = st.sidebar.radio(
    "Navigate",
    ["Executive Dashboard", "Cash Flow Forecast", "Basel III LCR",
     "Market Risk (VaR)", "Investment Optimizer", "About"],
)

use_sample = st.sidebar.checkbox("Use Sample Data", value=True)
if use_sample:
    daily = load_sample_data()
else:
    f = st.sidebar.file_uploader("Upload Transactions CSV", type="csv")
    daily = pd.read_csv(f) if f else load_sample_data()
daily["date"] = pd.to_datetime(daily["date"])

# ------------------------------------------------------------
# 1️⃣ Executive Dashboard
# ------------------------------------------------------------
if section == "Executive Dashboard":
    st.title("💠 Executive Dashboard")
    st.caption("High-level liquidity view and recent performance summary.")

    avg_in, avg_out, avg_net = daily.tail(30)[["inflow", "outflow", "net"]].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Inflow (30 d)", f"₹{avg_in:,.0f}")
    c2.metric("Avg Outflow (30 d)", f"₹{avg_out:,.0f}")
    c3.metric("Net Balance (30 d)", f"₹{avg_net:,.0f}")

    fig = px.line(daily, x="date", y=["inflow", "outflow", "net"],
                  title="Daily Cash Flows (₹)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.info("📊 This chart reveals daily liquidity movements, showing inflow–outflow patterns and volatility trends critical for day-ahead funding decisions.")

# ------------------------------------------------------------
# 2️⃣ Cash Flow Forecast
# ------------------------------------------------------------
elif section == "Cash Flow Forecast":
    st.title("📈 Cash Flow Forecast – Machine Learning (Random Forest)")
    horizon = st.slider("Forecast Horizon (days)", 7, 60, 30)
    with st.spinner("Forecasting…"):
        preds = rf_forecast(daily["net"], horizon=horizon)
    future_dates = pd.date_range(daily["date"].iloc[-1]+timedelta(days=1),
                                 periods=horizon)
    fc = pd.DataFrame({"date": future_dates, "forecast_net": preds})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["net"], name="Historical"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_net"], name="Forecast"))
    fig.update_layout(title="Projected Net Liquidity")
    st.plotly_chart(fig, use_container_width=True)
    st.success("🔍 This ML forecast anticipates upcoming net cash positions, empowering treasury teams to pre-empt liquidity stress and optimize overnight placements.")

# ------------------------------------------------------------
# 3️⃣ Basel III Liquidity Coverage Ratio
# ------------------------------------------------------------
elif section == "Basel III LCR":
    st.title("🧮 Basel III – Liquidity Coverage Ratio")
    st.caption("Compute regulatory LCR with customizable inputs.")

    h1 = st.number_input("Level 1 HQLA (₹)", 0.0, 5e10, 2e10, step=1e8)
    h2a = st.number_input("Level 2A HQLA (₹)", 0.0, 5e10, 5e9, step=1e8)
    h2b = st.number_input("Level 2B HQLA (₹)", 0.0, 5e10, 3e9, step=1e8)
    inflow_cap = st.slider("Inflow Cap (% of Outflows)", 50, 100, 75)/100
    avg_out = daily["outflow"].tail(30).mean()
    hqla, net_out, lcr = compute_lcr(h1, h2a, h2b, avg_out, inflow_cap)

    c1, c2, c3 = st.columns(3)
    c1.metric("Adj. HQLA", f"₹{hqla:,.0f}")
    c2.metric("Net Outflows (30 d)", f"₹{net_out:,.0f}")
    c3.metric("LCR %", f"{lcr:,.1f}%")
    st.info("💡 LCR ≥ 100% shows the bank can meet 30-day stress outflows via HQLA. Use this panel to simulate different HQLA structures and inflow assumptions.")

# ------------------------------------------------------------
# 4️⃣ Market Risk (VaR)
# ------------------------------------------------------------
elif section == "Market Risk (VaR)":
    st.title("⚠️ Market Risk – Value at Risk (VaR)")
    st.caption("Quantifies potential loss under normal market conditions.")

    cl = st.slider("Confidence Level (%)", 90, 99, 99)
    sigma = st.number_input("Daily Volatility (σ)", 0.001, 0.05, 0.015)
    mu = st.number_input("Mean Return (μ)", -0.01, 0.01, 0.0003)
    notional = st.number_input("Notional (₹)", 0.0, 1e10, 1e9, step=1e7)

    mc = monte_carlo_var(mu, sigma, cl/100, notional)
    rets = daily["net"].pct_change().dropna()
    hv = historical_var(rets, cl/100, notional)

    c1, c2 = st.columns(2)
    c1.metric(f"Monte Carlo VaR ({cl}%)", f"₹{mc:,.0f}")
    c2.metric(f"Historical VaR ({cl}%)", f"₹{hv:,.0f}" if hv else "N/A")
    st.info("📉 VaR estimates the worst-case loss for a given confidence level. Monte Carlo uses random simulations, while Historical VaR reflects past volatility realities.")

# ------------------------------------------------------------
# 5️⃣ Investment Optimizer
# ------------------------------------------------------------
elif section == "Investment Optimizer":
    st.title("🧠 Investment Optimizer – Yield vs Liquidity vs Compliance")
    st.caption("Linear program maximizes yield while meeting liquidity and Basel constraints.")

    total = st.number_input("Total Funds (₹)", 0.0, 1e10, 5e9, step=1e8)
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
        instr, res = optimize_investment(lcr_floor/100, horizon, yields, liq_days, haircuts,
                                         total, dur_cap)
    if res.success:
        alloc = pd.Series(res.x, index=instr)
        df_alloc = pd.DataFrame({
            "Instrument": instr,
            "Allocation (₹)": alloc,
            "Share (%)": (alloc / total) * 100
        })
        st.plotly_chart(px.pie(df_alloc, names="Instrument",
                               values="Allocation (₹)", title="Optimal Mix"),
                        use_container_width=True, config={"displayModeBar": False})
        st.dataframe(df_alloc.style.format({"Allocation (₹)": "₹{:,.0f}", "Share (%)": "{:.2f}%"}),
                     use_container_width=True)
        st.success(f"Expected Annual Yield: {np.dot(list(yields.values()), res.x)/total:.2%}")
        st.info("💼 This optimizer balances yield and regulatory liquidity requirements. Higher T-Bill and G-Sec allocation increases return within Basel LCR limits.")
    else:
        st.error("Optimization failed – adjust constraints.")

# ------------------------------------------------------------
# ℹ️ About
# ------------------------------------------------------------
else:
    st.title("ℹ️ About This Solution")
    st.markdown("""
**Treasury Management Solution (Optimized Edition)**  
Integrates AI-driven cash forecasting, Basel III liquidity analysis, VaR-based risk assessment,  
and an investment optimizer for banks and fintech treasuries.  

Built with Streamlit + Plotly + scikit-learn.  
Optimized using `st.cache_data` and `st.cache_resource` for instant loading.
""")
