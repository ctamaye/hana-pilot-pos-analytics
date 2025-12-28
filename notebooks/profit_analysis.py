#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 10:37:41 2025

@author: cody
"""
import os
import numpy as np
import pandas as pd

col_names = {
    'order_id': 'order_id',
    'date': 'order_datetime',
    'transaction_amount': 'line_total',
    'quantity': 'quantity',
    'item_name':'item_name'
    }

# *********************** Descriptive Revenue Decomposition (per item) ***********************
'''
class ItemRevenueObj
input:
    - df: Original DataFrame filtered for current object
    - col_names: col_names dictionary for renaming column names
    - start_date (optional): Allow user to specify a specific start date for analysis
    - end_date (optional): Allow user to specify a specific end date for analysis
'''
class ItemRevenueObj:
    def __init__(self, pos_df, col_names, item = None, start_date = None, end_date = None): #format: YYYY-MM-DD
        self.df = pos_df.rename(columns = col_names).copy()
        self.df_filtered = self.df.copy()
        
        # If start_date and end_date are provided, then filter
        if (start_date is not None) and (end_date is not None):
            self.df_filtered = self.df_filtered.loc[(self.df_filtered['order_datetime'] >= start_date) &
                                  (self.df_filtered['order_datetime'] <= end_date)].copy()
        elif (start_date is None) and (end_date is not None):
            self.df_filtered = self.df_filtered.loc[self.df_filtered['order_datetime'] <= end_date]
        elif (start_date is not None) and (end_date is None):
            self.df_filtered = self.df_filtered.loc[self.df_filtered['order_datetime'] >= start_date]
    
        # If item is specified then, filter for that item
        if item is not None:
            self.df_filtered = self.df_filtered.loc[self.df_filtered['item_name'] == item].copy()
    
    # Total Revenue
    def total_revenue(self):
        current_item_total_revenue = (
            self.df_filtered
                .groupby('item_name')['line_total']
                .sum()
                .reset_index(drop = True)
                .rename(columns = {'line_total': 'total_revenue'})
            )
        return current_item_total_revenue
        
    # % of Total Revenue
    def pct_total_revenue(self):
        total_revenues = (
            self.df_filtered
                .groupby('item_name')['line_total']
                .sum()
                .reset_index(drop = True)
                .rename(columns = {'line_total': 'total_revenue'})
            )
        total = total_revenues['total_revenue'].sum()
        total_revenues['pct_total_revenue'] = (total_revenues['total_revenue']/total) * 100
        
        return total_revenues
    
    # Total Quantity Sold
    def toal_quantites_sold(self):
        total_quantities = (           
                self.df_filtered
                    .groupby('item_name')['quantity']
                    .sum()
                    .reset_index(drop = True)
                    .rename(columns = {'quantity': 'total_quantity_sold'})
                )
        return total_quantities

# Average Transaction Contribution

# Sales Frequency
    def sales_frequency(self, time_span: list(['Monthly'])):
        if time_span == 'Monthly':
            

# Revenue Volatility (std dev over time)

# Combine to return merged dfs

# ******************************************* Testing *******************************************
if __name__ == '__main__':
    os.chdir(r'/Users/cody/Desktop/Projects/hana-pilot-pos-analytics/data/raw')
    line_items = pd.read_csv('indian_food_pos_raw.csv') #https://www.kaggle.com/datasets/rajatsurana979/fast-food-sales-report?utm_source=chatgpt.com#

