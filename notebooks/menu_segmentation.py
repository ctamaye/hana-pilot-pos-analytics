#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 15:53:26 2026

@author: cody
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import profit_analysis

# Visualize revenue-frequency chart
def segment(df, col_names, revenue_cut='median', freq_cut='median'):
    segment_obj = profit_analysis.ItemRevenueObj(df, col_names)

    truth = (
        segment_obj.total_revenue()
        .merge(segment_obj.order_frequency(), on='item_name', how='outer')
        .merge(segment_obj.total_quantites_sold(), on='item_name', how='outer')
    )

    # Fill NaNs (in case of outer joins)
    truth[['total_revenue', 'order_frequency', 'total_quantity_sold']] = truth[
        ['total_revenue', 'order_frequency', 'total_quantity_sold']
    ].fillna(0)

    # Choose thresholds
    if revenue_cut == 'median':
        rev_thr = truth['total_revenue'].median()
    else:
        rev_thr = truth['total_revenue'].quantile(float(revenue_cut))  # e.g., 0.6

    if freq_cut == 'median':
        freq_thr = truth['order_frequency'].median()
    else:
        freq_thr = truth['order_frequency'].quantile(float(freq_cut))

    # Assign segments
    def label(row):
        high_rev = row['total_revenue'] >= rev_thr
        high_freq = row['order_frequency'] >= freq_thr
        if high_rev and high_freq:
            return 'Protect'
        if high_rev and not high_freq:
            return 'Promote'
        if not high_rev and high_freq:
            return 'Reprice'
        return 'Cut/Test'

    truth['segment'] = truth.apply(label, axis=1)

    return truth, rev_thr, freq_thr









# ******************************************* Testing *******************************************
if __name__ == '__main__':
    os.chdir(r'/Users/cody/Desktop/Projects/hana-pilot-pos-analytics/data/raw')
    line_items = pd.read_csv('indian_food_pos_raw.csv')

    col_names = {
        'order_id': 'order_id',
        'date': 'order_datetime',
        'transaction_amount': 'line_total',
        'quantity': 'quantity',
        'item_name': 'item_name'
    }

    # Run segmentation
    truth, rev_thr, freq_thr = segment(line_items, col_names)

    # Print top items by segment
    for seg in ['Protect', 'Promote', 'Reprice', 'Cut/Test']:
        print(f"\n=== {seg} ===")
        print(
            truth[truth['segment'] == seg]
                .sort_values('total_revenue', ascending=False)
                .head(10)
        )

    # ------------------ PLOT ------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        truth['order_frequency'],
        truth['total_revenue'],
        alpha=0.7
    )

    # Threshold lines
    ax.axhline(rev_thr, linestyle='--')
    ax.axvline(freq_thr, linestyle='--')

    # Log scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Order Frequency (Unique Orders)')
    ax.set_ylabel('Total Revenue')
    ax.set_title('Menu Segmentation: Revenue vs Order Frequency')

    # -------- REGION LABELS (axis-relative, always visible) --------
    ax.text(0.75, 0.80, 'PROTECT', transform=ax.transAxes,
            fontsize=11, weight='bold')

    ax.text(0.10, 0.80, 'PROMOTE', transform=ax.transAxes,
            fontsize=11, weight='bold')

    ax.text(0.75, 0.15, 'REPRICE', transform=ax.transAxes,
            fontsize=11, weight='bold')

    ax.text(0.10, 0.15, 'CUT / TEST', transform=ax.transAxes,
            fontsize=11, weight='bold')

    # -------- ITEM LABELS (top revenue items only) --------
    top_items = truth.sort_values('total_revenue', ascending=False).head(10)

    for _, row in top_items.iterrows():
        ax.annotate(
            row['item_name'],
            (row['order_frequency'], row['total_revenue']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.85
        )

    plt.tight_layout()
    plt.show()

    