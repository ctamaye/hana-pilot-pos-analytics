#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 17:46:35 2025

@author: cody
"""

from visualize import VisualObj
import pandas as pd
import os

if __name__ == '__main__':
    # Import and prepare data for test
    os.chdir(r'/Users/cody/Desktop/Projects/hana-pilot-pos-analytics/data/raw')
    line_items = pd.read_csv('indian_food_pos_raw.csv') #https://www.kaggle.com/datasets/rajatsurana979/fast-food-sales-report?utm_source=chatgpt.com#


    #previous name: new_name
    col_names = {
        'order_id': 'order_id',
        'date': 'order_datetime',
        'transaction_amount': 'line_total',
        'quantity': 'quantity',
        'item_name':'item_name'
        }
    
    test = VisualObj(line_items, col_names, year_wanted = None)
    
    test.data_prep()
    test.total_sales_weekday()
    test.avg_sales_by_hour_by_day()
    test.total_sales_daily()
    test.best_item_per_month()
    test.item_heatmap()    
