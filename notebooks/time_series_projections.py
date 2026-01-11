#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stationarity + differencing prep for weekly item revenue series.
Integrates with profit_analysis.ItemRevenueObj.revenue_trend(), which returns
a pivoted wide dataframe: index=order_datetime (weekly), columns=item_name.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

import profit_analysis


def adf_test(series: pd.Series, label: str, alpha: float = 0.05) -> dict:
    """
    Run ADF test and return results in a dict (and print summary).
    """
    s = series.astype(float)
    
    if s.nunique() < 2:
        return {
            "label": label,
            "adf_stat": np.nan,
            "p_value": 0.0,
            "used_lag": np.nan,
            "n_obs": len(s),
            "crit_vals": {},
            "stationary": True,  # constant series is technically stationary
        }

    
    result = adfuller(s, autolag="AIC")
    out = {
        "label": label,
        "adf_stat": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "crit_vals": result[4],
        "stationary": result[1] < alpha,
    }

    print(f"ADF — {label}")
    print(f"  ADF Statistic: {out['adf_stat']:.4f}")
    print(f"  p-value:       {out['p_value']:.4f}")
    print(f"  Stationary?    {'YES' if out['stationary'] else 'NO'}\n")
    return out


def difference_until_stationary(series: pd.Series, max_d: int = 2, alpha: float = 0.05):
    """
    Apply differencing iteratively until ADF indicates stationarity or max_d reached.
    Returns: (final_series, d_used, results_list)
    """
    results = []
    current = series.astype(float)

    # Original
    results.append(adf_test(current, "original", alpha=alpha))
    if results[-1]["stationary"]:
        return current, 0, results

    # Differencing loop
    for d in range(1, max_d + 1):
        current = current.diff()
        current = current.dropna()
        results.append(adf_test(current, f"diff(d={d})", alpha=alpha))
        if results[-1]["stationary"]:
            return current, d, results

    return current, max_d, results


def stationarity_pipeline(
    df: pd.DataFrame,
    col_names: dict,
    time_span: str = "Weekly",
    max_d: int = 2,
    alpha: float = 0.05,
    min_nonzero_periods: int = 8,
    asfreq_rule: str = "W-MON",
    plot: bool = True,
):
    """
    Build weekly revenue series per item using profit_analysis, then
    test/difference per item.

    Returns:
      - rev_wide: wide revenue df (index=time, columns=item)
      - summary: per-item differencing/stationarity summary dataframe
    """
    obj = profit_analysis.ItemRevenueObj(df, col_names)
    rev_wide = obj.revenue_trend(time_span=time_span)  # already pivoted + filled with 0 in your profit_analysis

    # Ensure index is datetime + regular weekly frequency.
    # (Your revenue_trend builds weeks via Grouper; this makes gaps explicit)
    rev_wide.index = pd.to_datetime(rev_wide.index)
    rev_wide = rev_wide.sort_index().asfreq(asfreq_rule, fill_value=0)

    summary_rows = []
    differenced_dict = {}
    d_used_dict = {}
    
    for item in rev_wide.columns:
        y = rev_wide[item].astype(float).fillna(0)

        # Skip items with too little signal (mostly zeros)
        nonzero = int((y > 0).sum())
        if nonzero < min_nonzero_periods:
            summary_rows.append({
                "item_name": item,
                "status": "skipped_low_signal",
                "nonzero_periods": nonzero,
                "d_used": np.nan,
                "p_original": np.nan,
                "p_final": np.nan,
            })
            continue

        print("\n" + "=" * 90)
        print(f"ITEM: {item} | nonzero periods: {nonzero}")
        print("=" * 90)

        # Plot original series
        if plot:
            plt.figure(figsize=(12, 4))
            plt.plot(y.index, y.values)
            plt.title(f"Weekly Revenue (Original) — {item}")
            plt.xlabel("Week")
            plt.ylabel("Revenue")
            plt.tight_layout()
            plt.show()

        y_final, d_used, results = difference_until_stationary(y, max_d=max_d, alpha=alpha)
        
        # Store differenced (or original) for ACF/PACF + ARIMA
        differenced_dict[item] = y_final
        d_used_dict[item] = d_used

        # Plot differenced series (if differenced)
        if plot and d_used > 0:
            plt.figure(figsize=(12, 4))
            plt.plot(y_final.index, y_final.values)
            plt.title(f"Weekly Revenue (Differenced d={d_used}) — {item}")
            plt.xlabel("Week")
            plt.ylabel("Differenced Revenue")
            plt.tight_layout()
            plt.show()

        summary_rows.append({
            "item_name": item,
            "status": "ok",
            "nonzero_periods": nonzero,
            "d_used": int(d_used),
            "p_original": float(results[0]["p_value"]),
            "p_final": float(results[-1]["p_value"]),
        })

    summary = pd.DataFrame(summary_rows).sort_values(["status", "p_final"], ascending=[True, True])
    
    return rev_wide, summary, differenced_dict, d_used_dict

def plot_differenced_rev(differenced_dict):
    for item, s in differenced_dict.items():
        plt.figure(figsize=(12, 4))
        plt.plot(s.index, s.values, color='orange')
        plt.title(f"Differenced/Stationary Weekly Revenue — {item}")
        plt.xlabel("Week")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()


def plot_acf_pacf_with_orders(
    series: pd.Series,
    item_name: str = "",
    max_lags: int = 40,
):
    s = series.dropna().astype(float)

    if len(s) < 10:
        print(f"Not enough points to analyze {item_name}")
        return 0, 0

    # Infer p and q
    p, q, acf_vals, pacf_vals = infer_p_q_from_acf_pacf(
        s, max_lags=min(max_lags, len(s)//2 - 1)
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(s, lags=min(max_lags, len(s)//2 - 1), ax=axes[0])
    plot_pacf(s, lags=min(max_lags, len(s)//2 - 1), ax=axes[1], method="ywm")

    axes[0].set_title(f"ACF — {item_name} (q≈{q})")
    axes[1].set_title(f"PACF — {item_name} (p≈{p})")

    plt.tight_layout()
    plt.show()

    return p, q

def infer_p_q_from_acf_pacf(series: pd.Series, max_lags: int = 20, cap: int = 3):
    s = series.dropna().astype(float)
    n = len(s)
    if n < 10:
        return 0, 0, None, None

    max_lags = min(max_lags, max(1, len(s)//2 - 1))

    acf_vals = acf(s, nlags=max_lags, fft=True)
    pacf_vals = pacf(s, nlags=max_lags, method="ywm")

    conf = 1.96 / np.sqrt(n)

    acf_sig_lags = np.where(np.abs(acf_vals[1:]) > conf)[0] + 1
    pacf_sig_lags = np.where(np.abs(pacf_vals[1:]) > conf)[0] + 1

    q = int(acf_sig_lags.max()) if len(acf_sig_lags) > 0 else 0
    p = int(pacf_sig_lags.max()) if len(pacf_sig_lags) > 0 else 0

    # Cap to avoid overfitting / convergence issues
    p = min(p, cap)
    q = min(q, cap)

    return p, q, acf_vals, pacf_vals


def arima_model(
    rev_wide: pd.DataFrame,
    differenced_dict: dict,
    d_used_dict: dict,
    items: list = None,
    test_frac: float = 0.2,
    max_lags: int = 20,
    visualize: bool = True
):
    def arima_view(train: pd.Series, test: pd.Series, forecast: pd.Series, item: str):
        plt.figure(figsize=(14, 6))
        plt.plot(train.index, train, label="Train")
        plt.plot(test.index, test, label="Test")
        plt.plot(test.index, forecast, label="Forecast")
        plt.title(f"Revenue Forecast — {item}")
        plt.xlabel("Week")
        plt.ylabel("Revenue")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Default: run all items that made it into differenced_dict
    if items is None:
        items = list(differenced_dict.keys())

    results = []

    for item in items:
        y = rev_wide[item].astype(float).fillna(0)   # ORIGINAL series for forecasting
        d = int(d_used_dict[item])

        # Use stationary series to infer p,q
        stationary_series = differenced_dict[item]
        p, q, _, _ = infer_p_q_from_acf_pacf(stationary_series, max_lags=max_lags)

        # Train/test split (keep time order)
        n = len(y)
        test_size = max(1, int(n * test_frac))
        train, test = y.iloc[:-test_size], y.iloc[-test_size:]

        if len(train) < 12:
            results.append({
                "item_name": item,
                "p": p, "d": d, "q": q,
                "aic": np.nan,
                "error": "skipped_short_series"
            })
            continue


        # Fit ARIMA on train
        try:
            model = ARIMA(train, order=(p, d, q))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=len(test))
            forecast.index = test.index  # align for plotting

            if visualize:
                arima_view(train, test, forecast, item)

            results.append({
                "item_name": item,
                "p": p, "d": d, "q": q,
                "aic": model_fit.aic
            })

        except Exception as e:
            results.append({
                "item_name": item,
                "p": p, "d": d, "q": q,
                "aic": np.nan,
                "error": str(e)
            })

    return pd.DataFrame(results).sort_values(["aic"], ascending=True)

    
        

# ----------------------------------- Testing -----------------------------------
if __name__ == "__main__":
    os.chdir(r"/Users/cody/Desktop/Projects/hana-pilot-pos-analytics/data/raw")
    line_items = pd.read_csv("indian_food_pos_raw.csv")

    col_names = {
        "order_id": "order_id",
        "date": "order_datetime",
        "transaction_amount": "line_total",
        "quantity": "quantity",
        "item_name": "item_name",
    }

    rev_wide, summary, differenced_dict, d_used_dict = stationarity_pipeline(
        line_items,
        col_names,
        time_span="Weekly",
        max_d=2,
        alpha=0.05,
        min_nonzero_periods=8,
        asfreq_rule="W-MON",
        plot=True
    )

    # Print summary once (not inside loop)
    print("\nStationarity summary (top 25):")
    print(summary.head(25))

    # Pick a few items to inspect: the most stationary (lowest p_final) among ok items
    top_items = (
        summary[summary["status"] == "ok"]
        .sort_values(["nonzero_periods", "p_final"], ascending=[False, True])
        .head(5)["item_name"]
        .tolist()
    )


    # Plot ACF/PACF for those items using differenced_dict
    for item in top_items:
        series = differenced_dict[item]  # <-- dict, not dataframe
        plot_acf_pacf_with_orders(series, item_name=f"{item} (d={d_used_dict[item]})", max_lags=40)

    # ARIMA model on top items
    arima_results = arima_model(
        rev_wide=rev_wide,
        differenced_dict=differenced_dict,
        d_used_dict=d_used_dict,
        items=top_items,
        visualize=True
    )
    
    print("\nARIMA results (top items):")
    print(arima_results)


