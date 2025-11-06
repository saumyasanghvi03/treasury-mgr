from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

# import functions from your app.py
from app import (
    rf_forecast,
    compute_lcr,
    optimize_investment,
    make_sample_cash_txn
)

app = FastAPI(title="Treasury Management API", version="1.0.0")

# ---------------------------------- #
# Models
# ---------------------------------- #

class ForecastRequest(BaseModel):
    net_series: List[float]
    horizon: int = 30

class LCRRequest(BaseModel):
    h1: float
    h2a: float
    h2b: float
    avg_outflow: float
    inflow_cap: float = 0.75

class OptimizerRequest(BaseModel):
    total_funds: float
    horizon_days: int
    lcr_floor: float
    yield_cash: float
    yield_tb: float
    yield_gs: float
    yield_cp: float

# ---------------------------------- #
# Endpoints
# ---------------------------------- #

@app.get("/")
def root():
    return {"status": "Treasury API is running"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    series = pd.Series(req.net_series)
    preds, mae = rf_forecast(series, horizon=req.horizon)
    return {
        "forecast": preds.tolist(),
        "mae": mae
    }

@app.post("/lcr")
def lcr_calc(req: LCRRequest):
    hqla_adj, net_out, lcr = compute_lcr(
        req.h1, req.h2a, req.h2b, req.avg_outflow, req.inflow_cap
    )
    return {
        "adjusted_hqla": hqla_adj,
        "net_outflows": net_out,
        "lcr_percent": lcr
    }

@app.post("/optimizer")
def run_optimizer(req: OptimizerRequest):
    yields_dict = {
        "Cash": req.yield_cash,
        "T-Bills": req.yield_tb,
        "G-Secs": req.yield_gs,
        "CP/CD": req.yield_cp
    }

    liq_days = {"Cash": 0, "T-Bills": 30, "G-Secs": 180, "CP/CD": 90}
    haircuts = {"Cash": 1.00, "T-Bills": 0.85, "G-Secs": 0.85, "CP/CD": 0.75}

    instr, res = optimize_investment(
        hqla_min_ratio=req.lcr_floor/100,
        horizon_days=req.horizon_days,
        yields_dict=yields_dict,
        liquidity_days=liq_days,
        haircuts=haircuts,
        total_funds=req.total_funds,
        max_duration_days=120,
    )

    if not res.success:
        return {"status": "Optimization failed"}

    alloc = dict(zip(instr, res.x.tolist()))
    return {"allocation": alloc}
