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
import plotly.io as pio
pio.renderers.default = "svg"   # faster

@st.cache_data(show_spinner=False)
def load_sample_data():
    return make_sample_cash_txn(365)

@st.cache_data(show_spinner=False)
def preprocess_daily(df_raw):
    return to_daily(df_raw)

@st.cache_resource(show_spinner=False)
def load_model_training(history):
    train_df = lag_features(history)
    X, y = train_df.drop(columns=["y"]), train_df["y"]
    model = RandomForestRegressor(
        n_estimators=120,        # reduced from 400 → 3x faster
        max_depth=6,             # lighter model
        random_state=13,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


# ----------------------------- #
# Global Config & Styling Hook  #
# ----------------------------- #
st.set_page_config(
    page_title="Treasury Management Solution – Advanced",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS (expects styles.css in same folder)
def inject_css():
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

inject_css()

# ----------------------------- #
# Helpers                       #
# ----------------------------- #
def make_sample_cash_txn(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generate sample inflow/outflow transactions for demo."""
    rng = np.random.default_rng(seed)
    start = datetime.today().date() - timedelta(days=n_days - 1)
    dates = pd.date_range(start, periods=n_days, freq="D")
    # Baseline inflows/outflows with seasonality and noise
    inflow = np.maximum(0, 10_000_000 + 1_000_000*np.sin(np.arange(n_days)/3.2) + rng.normal(0, 600_000, n_days))
    outflow = np.maximum(0, 9_000_000 + 1_200_000*np.cos(np.arange(n_days)/5.4) + rng.normal(0, 650_000, n_days))
    # Random large corporate movements
    spikes = rng.choice(n_days, size=8, replace=False)
    outflow[spikes] += rng.uniform(5_000_000, 20_000_000, size=8)
    inflow[rng.choice(n_days, size=6, replace=False)] += rng.uniform(5_000_000, 15_000_000, size=6)

    df = pd.DataFrame({
        "date": dates,
        "inflow": inflow.astype(int),
        "outflow": outflow.astype(int),
    })
    df["net"] = df["inflow"] - df["outflow"]
    return df

def to_daily(df_txn: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded raw transactions (date, amount, type) to daily inflow/outflow."""
    df = df_txn.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "amount" not in df.columns:
        raise ValueError("File must include 'date' and 'amount' columns. Optional 'type' (inflow/outflow).")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "type" in df.columns:
        df["type"] = df["type"].str.lower().str.strip()
        inflows = df[df["type"].str.contains("in", na=False)].groupby("date")["amount"].sum()
        outflows = df[df["type"].str.contains("out", na=False)].groupby("date")["amount"].sum()
    else:
        # Heuristic: positive amounts are inflows, negatives are outflows
        inflows = df[df["amount"] > 0].groupby("date")["amount"].sum()
        outflows = -df[df["amount"] < 0].groupby("date")["amount"].sum()

    all_days = pd.date_range(min(df["date"]), max(df["date"]), freq="D").date
    daily = pd.DataFrame({"date": all_days})
    daily["inflow"] = daily["date"].map(inflows).fillna(0.0)
    daily["outflow"] = daily["date"].map(outflows).fillna(0.0)
    daily["net"] = daily["inflow"] - daily["outflow"]
    return daily

def lag_features(series: pd.Series, lags=(1,2,3,7,14,21)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for l in lags:
        df[f"lag_{l}"] = series.shift(l)
    df["dow"] = np.arange(len(series)) % 7
    return df.dropna()

def rf_forecast(daily_net: pd.Series, horizon: int = 30, random_state: int = 13):
    """Walk-forward RandomForest on lagged features, rolling one-step forecasts."""
    y = daily_net.reset_index(drop=True)
    preds = []
    history = y.copy()
    model = RandomForestRegressor(
        n_estimators=400, max_depth=8, random_state=random_state, n_jobs=-1
    )
    # Train on all available, then iteratively append predictions to generate horizon steps
    train_df = lag_features(history)
    X, y_train = train_df.drop(columns=["y"]), train_df["y"]
    model.fit(X, y_train)
    last = history.copy()
    for _ in range(horizon):
        tmp = lag_features(last)
        X_last = tmp.drop(columns=["y"]).iloc[[-1]]
        yhat = float(model.predict(X_last)[0])
        preds.append(yhat)
        last = pd.concat([last, pd.Series([yhat])], ignore_index=True)
    # Backtest MAE via time-series split
    mae = None
    if len(history) > 120:
        tss = TimeSeriesSplit(n_splits=3)
        maes = []
        base = lag_features(history)
        Xb, yb = base.drop(columns=["y"]), base["y"]
        for tr, te in tss.split(Xb):
            m = RandomForestRegressor(n_estimators=300, max_depth=7, random_state=random_state, n_jobs=-1)
            m.fit(Xb.iloc[tr], yb.iloc[tr])
            p = m.predict(Xb.iloc[te])
            maes.append(mean_absolute_error(yb.iloc[te], p))
        mae = float(np.mean(maes))
    return np.array(preds), mae

def compute_lcr(hqla_level1, hqla_level2a, hqla_level2b, avg_30d_outflow, inflow_cap_ratio=0.75):
    """
    Basel III LCR = Stock of HQLA / Net Cash Outflows over next 30 days.
    HQLA haircuts: Level 1 (0%), Level 2A (15%), Level 2B (25%).
    Inflows capped at 75% of outflows by default (configurable).
    """
    hqla_adj = (hqla_level1 * 1.0) + (hqla_level2a * 0.85) + (hqla_level2b * 0.75)
    # Assume inflows at 50% of average outflow; capped
    inflows_est = min(avg_30d_outflow * inflow_cap_ratio, avg_30d_outflow * 0.5)
    net_out = max(avg_30d_outflow - inflows_est, 0.0)
    lcr = (hqla_adj / (net_out + 1e-9)) * 100.0 if net_out > 0 else np.inf
    return hqla_adj, net_out, lcr

def historical_var(returns: np.ndarray, cl: float = 0.99, notional: float = 1.0):
    """Historical simulation VaR."""
    if len(returns) < 50:
        return None
    q = np.quantile(returns, 1 - cl)
    return -q * notional

def monte_carlo_var(mu=0.0003, sigma=0.015, days=1, sims=20_000, cl=0.99, notional=1.0, seed=7):
    rng = np.random.default_rng(seed)
    # Geometric Brownian Motion daily return draw ~ N(mu*days, sigma*sqrt(days))
    ret = rng.normal(mu * days, sigma * np.sqrt(days), sims)
    q = np.quantile(ret, 1 - cl)
    return -q * notional

def build_waterfall(inflow, outflow, title="Cash Flow Waterfall"):
    fig = go.Figure(go.Waterfall(
        name="24h",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Inflows", "Outflows", "Net"],
        textposition="outside",
        text=[f"{inflow:,.0f}", f"-{outflow:,.0f}", ""],
        y=[inflow, -outflow, inflow - outflow],
        connector={"line":{"dash":"dot"}}
    ))
    fig.update_layout(title=title)
    return fig

def payments_intraday_metrics(df_pay: pd.DataFrame):
    """
    Expected columns: timestamp, amount, direction (in/out).
    Returns cumulative net outflow curve and peak net outflow.
    """
    d = df_pay.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    if "timestamp" not in d.columns or "amount" not in d.columns:
        raise ValueError("Payments file requires 'timestamp' and 'amount'. Optional: 'direction' (in/out).")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d["hour"] = d["timestamp"].dt.floor("15min")
    if "direction" in d.columns:
        d["sign"] = np.where(d["direction"].str.lower().str.startswith("in"), 1, -1)
    else:
        # heuristic: positive amounts are inflows
        d["sign"] = np.where(d["amount"] >= 0, 1, -1)
    d["net"] = d["amount"] * d["sign"]
    curve = d.groupby("hour")["net"].sum().sort_index().cumsum()
    peak_draw = -curve.min()
    return curve, peak_draw

def optimize_investment(hqla_min_ratio, horizon_days, yields_dict, liquidity_days, haircuts, total_funds=5e9, max_duration_days=365):
    """
    Linear program: maximize expected return subject to:
    - Weighted average duration <= max_duration_days
    - HQLA-adjusted liquidity available within 30d >= hqla_min_ratio * projected net outflows
    - Full investment of total_funds
    Instruments: Cash(0d), T-Bills(30d), G-Sec(180d), CP/CD(90d)
    """
    instr = list(yields_dict.keys())
    y = np.array([yields_dict[k] for k in instr])          # annualized yield
    dur = np.array([liquidity_days[k] for k in instr], float)
    hqla_adj = np.array([haircuts[k] for k in instr])      # LCR haircuts applied

    # Constraints matrices for linprog (Ax <= b)
    # 1) Duration: sum(dur_i * xi) <= max_duration_days * total_funds
    A_ub = [dur.tolist()]
    b_ub = [max_duration_days * total_funds]

    # 2) HQLA within horizon: sum(hqla_adj_i * xi) >= required_liquidity
    projected_outflow = 0.006 * total_funds * (horizon_days / 30)  # simple stress proxy
    required = hqla_min_ratio * projected_outflow
    A_ub.append((-hqla_adj).tolist())  # convert >= to <=
    b_ub.append(-required)

    # 3) Full investment: sum(xi) == total_funds
    A_eq = [np.ones(len(instr)).tolist()]
    b_eq = [total_funds]

    bounds = [(0, total_funds) for _ in instr]

    res = linprog(
        c=(-y), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=bounds, method="highs"
    )
    return instr, res

# ----------------------------- #
# Sidebar: Inputs & Navigation  #
# ----------------------------- #
st.sidebar.title("💠 Treasury Management – Control Tower")
section = st.sidebar.radio(
    "Navigate", 
    ["Executive Dashboard", "Cash Flow Forecast", "Basel III LCR", "ALM Gap", "Risk (VaR & Stress)", "Intraday Liquidity", "Investment Optimizer", "About the Project"],
    index=0
)

st.sidebar.subheader("Data Ingestion")
uploaded_txn = st.sidebar.file_uploader("Upload Transactions CSV (date, amount, type)", type=["csv"], key="txn")
uploaded_pay = st.sidebar.file_uploader("Upload Payments CSV (timestamp, amount, direction)", type=["csv"], key="pay")

use_sample = st.sidebar.checkbox("Use Sample Data", value=True)
if use_sample:
    daily = load_sample_data()
else:
    if uploaded_txn is not None:
        df_raw = pd.read_csv(uploaded_txn)
        try:
            daily = to_daily(df_raw)
        except Exception as e:
            st.sidebar.error(str(e))
            st.stop()
    else:
        st.sidebar.info("Upload a transactions CSV or tick 'Use Sample Data'.")
        st.stop()

# Common KPIs
last30 = daily.tail(30)
avg_in = last30["inflow"].mean()
avg_out = last30["outflow"].mean()
avg_net = last30["net"].mean()

# ----------------------------- #
# Executive Dashboard           #
# ----------------------------- #
if section == "Executive Dashboard":
    st.title("💠 Treasury Management Solution – Executive Dashboard")
    st.caption("AI-augmented cash control, intraday liquidity, and regulatory analytics for bank treasuries.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Daily Inflow (30d)", f"₹{avg_in:,.0f}")
    k2.metric("Avg Daily Outflow (30d)", f"₹{avg_out:,.0f}")
    k3.metric("Avg Net Position (30d)", f"₹{avg_net:,.0f}", delta=f"{(avg_net/avg_out*100 if avg_out else 0):.1f}% of outflows")
    k4.metric("Data Horizon", f"{len(daily):,} days")

    c1, c2 = st.columns([2,1])
    with c1:
        fig = px.line(daily, x="date", y=["inflow","outflow","net"], title="Daily Inflows / Outflows / Net")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        wf = build_waterfall(last30["inflow"].sum(), last30["outflow"].sum(), title="30-Day Waterfall")
        st.plotly_chart(wf, use_container_width=True)

    st.markdown("#### Top-Down Signals")
    s1, s2, s3 = st.columns(3)
    vol = daily["net"].rolling(30).std().iloc[-1]
    s1.metric("Net Volatility (30d σ)", f"₹{vol:,.0f}")
    neg_days = int((daily["net"].tail(30) < 0).sum())
    s2.metric("Negative Net Days (30d)", f"{neg_days}/30")
    draw = (daily["net"].cumsum().min())
    s3.metric("Max Cumulative Drawdown", f"₹{draw:,.0f}")

    st.markdown("---")
    st.markdown("**Download snapshot**")
    snap = daily.copy()
    snap["date"] = pd.to_datetime(snap["date"])
    st.download_button(
        "Export Daily Series (CSV)",
        data=snap.to_csv(index=False).encode("utf-8"),
        file_name="treasury_daily_series.csv",
        mime="text/csv"
    )

# ----------------------------- #
# Cash Flow Forecast            #
# ----------------------------- #
elif section == "Cash Flow Forecast":
    st.title("📈 Cash Flow Forecast – ML (Random Forest)")
    horizon = st.slider("Forecast Horizon (days)", 7, 60, 30, step=1)
    preds, mae = rf_forecast(daily["net"], horizon=horizon)
    f_dates = pd.date_range(daily["date"].iloc[-1] + timedelta(days=1), periods=horizon, freq="D")

    fc = pd.DataFrame({"date": f_dates, "forecast_net": preds})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["net"], name="Historical Net"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_net"], name="Forecast Net", mode="lines"))
    fig.update_layout(title="Net Cash Forecast")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Summary")
        st.write(f"Projected {horizon}-day net position: ₹{fc['forecast_net'].sum():,.0f}")
        if mae is not None:
            st.caption(f"Backtest MAE (3-fold time series split): ₹{mae:,.0f}")
    with c2:
        st.download_button(
            "Download Forecast (CSV)",
            data=fc.to_csv(index=False).encode("utf-8"),
            file_name="cash_forecast.csv",
            mime="text/csv"
        )

# ----------------------------- #
# Basel III LCR                 #
# ----------------------------- #
elif section == "Basel III LCR":
    st.title("🧮 Basel III – Liquidity Coverage Ratio")
    st.caption("Compute LCR with configurable HQLA haircuts and inflow caps.")

    l1, l2, l3 = st.columns(3)
    with l1:
        h1 = st.number_input("Level 1 HQLA (₹)", min_value=0.0, value=20_000_000_000.0, step=10_000_000.0, format="%.2f")
    with l2:
        h2a = st.number_input("Level 2A HQLA (₹)", min_value=0.0, value=5_000_000_000.0, step=10_000_000.0, format="%.2f")
    with l3:
        h2b = st.number_input("Level 2B HQLA (₹)", min_value=0.0, value=3_000_000_000.0, step=10_000_000.0, format="%.2f")

    inflow_cap = st.slider("Inflow Cap (% of Outflows)", 50, 100, 75, step=5) / 100.0
    avg_out_30 = float(daily["outflow"].tail(30).mean())
    hqla_adj, net_out, lcr = compute_lcr(h1, h2a, h2b, avg_out_30, inflow_cap_ratio=inflow_cap)

    k1, k2, k3 = st.columns(3)
    k1.metric("Adjusted HQLA", f"₹{hqla_adj:,.0f}")
    k2.metric("Net Cash Outflows (30d)", f"₹{net_out:,.0f}")
    k3.metric("LCR (%)", f"{lcr:,.1f}%", delta=">= 100% required")

    st.markdown("#### Sensitivity")
    shocks = [-20, -10, 0, 10, 20, 30]
    lcrs = []
    for s in shocks:
        _, net_s, lcr_s = compute_lcr(h1, h2a, h2b, avg_out_30*(1+s/100), inflow_cap_ratio=inflow_cap)
        lcrs.append(lcr_s)
    fig = px.line(x=shocks, y=lcrs, markers=True, title="LCR vs. Outflow Shock (%)")
    fig.update_layout(xaxis_title="Outflow Shock (%)", yaxis_title="LCR (%)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ----------------------------- #
# ALM Gap Analysis              #
# ----------------------------- #
elif section == "ALM Gap":
    st.title("🏦 ALM Gap – Repricing Buckets")
    st.caption("Analyze RSA/RSL by bucket to quantify gap and cumulative gap.")

    buckets = ["0–7d","8–30d","31–90d","91–365d",">1y"]
    rsa, rsl = [], []
    defaults_rsa = [15e9, 25e9, 22e9, 18e9, 10e9]
    defaults_rsl = [12e9, 27e9, 25e9, 20e9, 8e9]
    cols = st.columns(len(buckets))
    st.write("**Rate-Sensitive Assets (RSA)**")
    for i,b in enumerate(buckets):
        with cols[i]:
            rsa.append(st.number_input(b, min_value=0.0, value=defaults_rsa[i], step=1e8, key=f"rsa_{i}"))
    cols = st.columns(len(buckets))
    st.write("**Rate-Sensitive Liabilities (RSL)**")
    for i,b in enumerate(buckets):
        with cols[i]:
            rsl.append(st.number_input(b, min_value=0.0, value=defaults_rsl[i], step=1e8, key=f"rsl_{i}"))

    gap = np.array(rsa) - np.array(rsl)
    cum_gap = np.cumsum(gap)
    df_gap = pd.DataFrame({"Bucket": buckets, "GAP": gap, "Cumulative GAP": cum_gap})
    fig1 = px.bar(df_gap, x="Bucket", y="GAP", title="ALM GAP by Bucket")
    fig2 = px.line(df_gap, x="Bucket", y="Cumulative GAP", markers=True, title="Cumulative GAP")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------- #
# Risk (VaR & Stress)           #
# ----------------------------- #
elif section == "Risk (VaR & Stress)":
    st.title("⚠️ Market Risk – VaR & Stress Testing")
    st.caption("Historical and Monte-Carlo VaR with simple shock scenarios.")

    cl = st.slider("Confidence Level (%)", 90, 99, 99, step=1)
    muref = st.number_input("Mean Daily Return (μ)", value=0.0003, step=0.0001, format="%.4f")
    sigref = st.number_input("Daily Volatility (σ)", value=0.0150, step=0.0010, format="%.4f")
    notional = st.number_input("Notional (₹)", value=1_000_000_000.0, step=10_000_000.0, format="%.2f")

    mc_var = monte_carlo_var(mu=muref, sigma=sigref, cl=cl/100, notional=notional)
    k1, k2 = st.columns(2)
    k1.metric(f"MC VaR {cl}% (1-day)", f"₹{mc_var:,.0f}")

    # Build synthetic returns from net series as proxy
    rets = pd.Series(daily["net"]).pct_change().replace([np.inf,-np.inf], np.nan).dropna().clip(-0.2, 0.2)
    hv = historical_var(rets.values, cl=cl/100, notional=notional)
    k2.metric(f"Historical VaR {cl}% (1-day)", f"₹{hv:,.0f}" if hv is not None else "N/A")

    st.markdown("#### Stress Scenarios")
    shocks = {
        "Rates +150 bps": -0.012,
        "Rates +300 bps": -0.025,
        "FX Depreciation 5%": -0.050,
        "Equities −10%": -0.100,
    }
    stress_df = pd.DataFrame({"Scenario": shocks.keys(), "P&L (₹)": [notional*v for v in shocks.values()]})
    fig = px.bar(stress_df, x="Scenario", y="P&L (₹)", title="Instantaneous Shock P&L")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.dataframe(stress_df, use_container_width=True)

# ----------------------------- #
# Intraday Liquidity            #
# ----------------------------- #
elif section == "Intraday Liquidity":
    st.title("⏱️ Intraday Liquidity – Payments Analytics")
    st.caption("Identify peak net cumulative outflows for RTGS/settlement corridors.")
    if uploaded_pay is not None:
        dfp = pd.read_csv(uploaded_pay)
        curve, peak = payments_intraday_metrics(dfp)
        fig = px.line(x=curve.index, y=curve.values, labels={"x":"Time", "y":"Cumulative Net (₹)"},
                      title="Intraday Cumulative Net Position (15-min)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.metric("Peak Net Cumulative Outflow", f"₹{peak:,.0f}")
    else:
        st.info("Upload a Payments CSV with columns: timestamp, amount, direction (in/out).")

# ----------------------------- #
# Investment Optimizer          #
# ----------------------------- #
elif section == "Investment Optimizer":
    st.title("🧠 Investment Optimizer – Yield vs. Liquidity vs. Regulation")
    st.caption("Linear program to allocate across Cash, T-Bills, G-Secs, CP/CD under LCR and duration constraints.")

    total = st.number_input("Total Treasury Funds (₹)", min_value=0.0, value=5_000_000_000.0, step=10_000_000.0, format="%.2f")
    horizon = st.slider("Liquidity Horizon (days)", 7, 60, 30, step=1)
    lcr_floor = st.slider("HQLA Coverage Floor vs. Projected Outflows", 100, 200, 125, step=5)
    dur_cap = st.slider("Max Weighted Avg Duration (days)", 7, 365, 120, step=1)

    st.markdown("#### Instrument Parameters (Annualized Yields & Haircuts)")
    c1, c2, c3, c4 = st.columns(4)
    y_cash = c1.number_input("Cash Yield", value=0.035, step=0.001, format="%.3f")
    y_tb   = c2.number_input("T-Bills Yield", value=0.064, step=0.001, format="%.3f")
    y_gs   = c3.number_input("G-Secs Yield", value=0.072, step=0.001, format="%.3f")
    y_cp   = c4.number_input("CP/CD Yield", value=0.078, step=0.001, format="%.3f")

    h_cash = 1.00  # L1 equivalent
    h_tb   = 0.85  # 2A proxy
    h_gs   = 0.85  # 2A proxy
    h_cp   = 0.75  # 2B proxy

    liq_days = {"Cash": 0, "T-Bills": 30, "G-Secs": 180, "CP/CD": 90}
    yields_dict = {"Cash": y_cash, "T-Bills": y_tb, "G-Secs": y_gs, "CP/CD": y_cp}
    haircuts = {"Cash": h_cash, "T-Bills": h_tb, "G-Secs": h_gs, "CP/CD": h_cp}

    instr, res = optimize_investment(
        hqla_min_ratio=lcr_floor/100, horizon_days=horizon, 
        yields_dict=yields_dict, liquidity_days=liq_days, haircuts=haircuts,
        total_funds=total, max_duration_days=dur_cap
    )
    if res.success:
        alloc = pd.Series(res.x, index=instr)
        df_alloc = pd.DataFrame({
            "Instrument": instr,
            "Allocation (₹)": alloc.values,
            "Share (%)": (alloc.values/total)*100,
        })
        fig = px.pie(df_alloc, names="Instrument", values="Allocation (₹)", title="Optimal Allocation Mix")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(df_alloc.style.format({"Allocation (₹)": "₹{:,.0f}", "Share (%)": "{:.2f}%"}), use_container_width=True)
        st.success(f"Expected Annual Yield: {np.dot([yields_dict[k] for k in instr], res.x)/total:.2%}")
    else:
        st.error("Optimization failed to converge. Adjust constraints and try again.")

# ----------------------------- #
# About                         #
# ----------------------------- #
elif section == "About the Project":
    st.title("ℹ️ About – Advanced Treasury Management")
    st.markdown("""
This application showcases a full-stack treasury control tower for banks:
- **ML Forecasting** for daily net cash using lagged features and Random Forest.
- **Basel III LCR** engine with configurable haircuts and inflow caps plus sensitivity analysis.
- **ALM GAP** explorer across standard repricing buckets.
- **Market Risk** with Historical & Monte-Carlo VaR and shock scenarios.
- **Intraday Liquidity** analytics to identify peak net settlement outflows.
- **Investment Optimizer** leveraging linear programming to maximize yield under regulatory and duration guardrails.

> **Presentation Tip (MBA):** Use the *Executive Dashboard* for the narrative arc, then deep-dive into **Forecast → LCR → ALM → Risk → Intraday → Optimizer** to illustrate decision-to-execution flow.
""")
