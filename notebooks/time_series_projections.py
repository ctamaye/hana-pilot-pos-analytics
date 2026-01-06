#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 17:33:38 2025

@author: cody
"""
# Basic modules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import os
import profit_analysis

# Statistical and ARIMA modules
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
'''
The following function projects the next week's sales performance by item based on the existing data.
This is done by using an Autoregressive Integrated Moving Average (ARIMA) model

Input: DataFrame with weekly revenues for each item
Output:
    - A plot showing the prediction estimated for the next time frame
    - Print projections to console
'''

# ARIMA model
def arima_proj(df, col_names):
    proj_obj = profit_analysis.ItemRevenueObj(df, col_names)
    rev_trend = proj_obj.revenue_trend(time_span = 'Weekly')
    
    for current_item in rev_trend.columns:
        current_rev = rev_trend[current_item]
        
        plt.figure(figsize = (12, 5))
        plt.plot(current_rev.index, current_rev)
        plt.title(f'Weekly Revenue for {current_item}')
        plt.xlabel('Date')
        plt.ylabel('Weekly Revenue (Rupees)')
        plt.show()

        print(f'\n**************** Checking Stationarity for {current_item} Series ****************\n')

        # Perform the Augmented Dickey-Fuller test on the original series
        result_original = adfuller(current_rev)
        print(f"ADF Statistic (Original): {result_original[0]:.4f}")
        print(f"p-value (Original): {result_original[1]:.4f}")
        if result_original[1] < 0.05:    
            print("Interpretation: The original series is Stationary.\n")
        else:   
            print("Interpretation: The original series is Non-Stationary.\n")
            
        # Perform the Augmented Dickey-Fuller test on the differenced series
        result_diff = adfuller(current_rev.dropna())
        print(f"ADF Statistic (Differenced): {result_diff[0]:.4f}")
        print(f"p-value (Differenced): {result_diff[1]:.4f}")
        if result_diff[1] < 0.05:    
            print("Interpretation: The differenced series is Stationary.\n")
        else:    
            print("Interpretation: The differenced series is Non-Stationary.\n")
            
        # Plotting the differenced Close price
        
        
        plt.figure(figsize=(14, 7))
        plt.plot(current_rev.index, current_rev, label = f'Differenced Weekly Revenues for {current_item}', color='orange')
        plt.title(f'Differenced Weekly Revenues for {current_item} Over Time')
        plt.xlabel('Date')
        plt.ylabel('Differenced Weekly Revenue (Rupees)')
        plt.legend()
        plt.show()
        
    return rev_trend
    
    
# ******************************************* Testing *******************************************
if __name__ == '__main__':
    os.chdir(r'/Users/cody/Desktop/Projects/hana-pilot-pos-analytics/data/raw')
    line_items = pd.read_csv('indian_food_pos_raw.csv') #https://www.kaggle.com/datasets/rajatsurana979/fast-food-sales-report?utm_source=chatgpt.com#

    col_names = {
        'order_id': 'order_id',
        'date': 'order_datetime',
        'transaction_amount': 'line_total',
        'quantity': 'quantity',
        'item_name':'item_name'
        }

    rev_trend = arima_proj(line_items, col_names)
