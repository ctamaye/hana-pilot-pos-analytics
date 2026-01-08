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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import profit_analysis


def adf_test(series: pd.Series, label: str, alpha: float = 0.05) -> dict:
    """
    Run ADF test and return results in a dict (and print summary).
    """
    s = series.dropna()
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
    differenced_list = []

    for item in rev_wide.columns:
        y = rev_wide[item].astype(float)

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
        if d_used > 0:
            differenced_list.append(y_final)

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
    differenced_df = pd.concat(differenced_list)
    
    return rev_wide, summary, differenced_df

def plot_differenced_rev(differenced_df):
    if isinstance(differenced_df, pd.DataFrame): #if DataFrame
        for current_item in differenced_df.columns:
            current_differenced = differenced_df[current_item].copy()
            
            plt.figure(figsize=(12, 4))
            plt.plot(current_differenced.index, current_differenced.values, color = 'orange')
            plt.title(f"Differenced Weekly Revenue Over Time — {current_item}")
            plt.xlabel("Week")
            plt.ylabel("Differenced Revenue (Rupees)")
            plt.tight_layout()
            plt.show()
    else:
            plt.figure(figsize=(12, 4))
            plt.plot(differenced_df.index, differenced_df.values, color = 'orange')
            plt.title(f"Differenced Weekly Revenue Over Time — {differenced_df.name}")
            plt.xlabel("Week")
            plt.ylabel("Differenced Revenue (Rupees)")
            plt.tight_layout()
            plt.show()

def autocorrelation_func():
    # # Plot ACF and PACF for the differenced series
    # fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    # # ACF plot
    # plot_acf(data['Close_Diff'].dropna(), lags=40, ax=axes[0])axes[0].set_title('Autocorrelation Function (ACF)')
    # # PACF plot
    # plot_pacf(data['Close_Diff'].dropna(), lags=40, ax=axes[1])axes[1].set_title('Partial Autocorrelation Function (PACF)')
    
    # plt.tight_layout()
    # plt.show()

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

    rev_wide, summary, differenced_df = stationarity_pipeline(
        line_items,
        col_names,
        time_span="Weekly",
        max_d=2,
        alpha=0.05,
        min_nonzero_periods=8,
        asfreq_rule="W-MON",
        plot=True
    )

    plot_differenced_rev(differenced_df)

    print("\nStationarity summary (top 25):")
    print(summary.head(25))
