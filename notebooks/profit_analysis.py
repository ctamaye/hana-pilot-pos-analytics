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
        
        # Check and reformat datetimes
        try:
            self.df['order_datetime'] = pd.to_datetime(self.df['order_datetime'])
        except ValueError:
            self.df['order_datetime'] = self.df['order_datetime'].apply(lambda x: x.replace(' ', ''))
            self.df['order_datetime'] = self.df['order_datetime'].apply(lambda x: x.replace('/', '-'))
            self.df['order_datetime'] = pd.to_datetime(self.df['order_datetime'])
            print('Datetime column reformatted successfully.\n')
        else:
            print('Check datetime column again.\n')

        # Add day of week and weekend flags
        self.df['Day of Week'] = self.df['order_datetime'].dt.day_name()
        self.df['Weekend?'] = self.df['Day of Week'].apply(lambda x: 1 if (x == 'Saturday') or (x == 'Sunday') else 0)

        if start_date is not None:
            start_date = pd.to_datetime(start_date)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)

        # If start_date and end_date are provided, then filter
        if (start_date is not None) and (end_date is not None):
            self.df = self.df.loc[(self.df['order_datetime'] >= start_date) &
                                  (self.df['order_datetime'] <= end_date)].copy()
            
        elif (start_date is None) and (end_date is not None):
            self.df = self.df.loc[self.df['order_datetime'] <= end_date]
            
        elif (start_date is not None) and (end_date is None):
            self.df = self.df.loc[self.df['order_datetime'] >= start_date]


        self.df_filtered = self.df.copy()


        # If item is specified then, filter for that item
        if item is not None:
            self.df_filtered = self.df_filtered.loc[self.df_filtered['item_name'] == item].copy()
    
    # Total Revenue
    def total_revenue(self):
        current_item_total_revenue = (
            self.df_filtered
                .groupby('item_name', as_index = False)['line_total']
                .sum()
                .rename(columns = {'line_total': 'total_revenue'})
            )
        return current_item_total_revenue
        
    # % of Total Revenue
    def pct_total_revenue(self):
        total_revenues = (
            self.df
                .groupby('item_name', as_index = False)['line_total']
                .sum()
                .rename(columns = {'line_total': 'total_revenue'})
            )
        total = total_revenues['total_revenue'].sum()
        total_revenues['pct_total_revenue'] = (total_revenues['total_revenue']/total) * 100
        
        return total_revenues
    
    # Total Quantity Sold
    def total_quantites_sold(self):
        total_quantities = (           
                self.df_filtered
                    .groupby('item_name', as_index = False)['quantity']
                    .sum()
                    .rename(columns = {'quantity': 'total_quantity_sold'})
                )
        return total_quantities

    # Average ticket contribution (avg share of order total when item appears)
    def avg_ticket_contribution(self):
        # revenue per (order, item)
        item_order_rev = (
            self.df_filtered
                .groupby(['order_id', 'item_name'], as_index=False)['line_total']
                .sum()
        )

        # total revenue per order (from full df in window, not df_filtered)
        order_totals = self.df.groupby('order_id')['line_total'].sum()

        item_order_rev['ticket_total'] = item_order_rev['order_id'].map(order_totals)
        item_order_rev['ticket_share'] = item_order_rev['line_total'] / item_order_rev['ticket_total']

        avg_share = (
            item_order_rev
                .groupby('item_name', as_index=False)['ticket_share']
                .mean()
                .rename(columns={'ticket_share': 'avg_ticket_share'})
        )
        return avg_share
    
    # Sales Frequency: unique orders per week/month
    def order_frequency_over_time(self, time_span: str):
        if time_span == 'Weekly':
            grp = pd.Grouper(key='order_datetime', freq='W-MON', label='left')
        elif time_span == 'Monthly':
            grp = pd.Grouper(key='order_datetime', freq='MS')
        else:
            raise ValueError("time_span must be 'Weekly' or 'Monthly'")
    
        return (
            self.df_filtered
                .groupby(['item_name', grp], as_index=False)['order_id']
                .nunique()
                .rename(columns={'order_id': 'order_frequency'})
        )

            

    # Revenue volatility over time: std dev of weekly/monthly item revenue
    def revenue_volatility(self, time_span: str):
        if time_span == 'Weekly':
            ts = (
                self.df_filtered
                    .groupby(['item_name', pd.Grouper(key='order_datetime', freq='W-MON', label='left')])['line_total']
                    .sum()
                    .reset_index()
            )
        elif time_span == 'Monthly':
            ts = (
                self.df_filtered
                    .groupby(['item_name', pd.Grouper(key='order_datetime', freq='MS')])['line_total']
                    .sum()
                    .reset_index()
            )
        else:
            raise ValueError("time_span must be 'Weekly' or 'Monthly'")

        vol = (
            ts.groupby('item_name', as_index=False)['line_total']
              .std()
              .rename(columns={'line_total': f'revenue_volatility_std_{time_span.lower()}'})
        )
        return vol


    # Revenue Trend
    def revenue_trend(self, time_span: str):
        if time_span == 'Weekly':
            rev_trend = self.df_filtered.groupby(['item_name', pd.Grouper(key = 'order_datetime', freq = 'W-MON', label = 'left')])['line_total'].sum().reset_index()
        elif time_span == 'Monthly':
            rev_trend = self.df_filtered.groupby(['item_name', pd.Grouper(key = 'order_datetime', freq = 'MS')])['line_total'].sum().reset_index()
            
        rev_trend_pivot = rev_trend.pivot(index = 'order_datetime', columns = 'item_name', values = 'line_total')
        rev_trend_pivot = rev_trend_pivot.fillna(0) #if the item didn't sell that week, then total revenue of item is 0
        
        return rev_trend_pivot

    # Quantity Trend
    def quantity_trend(self, time_span: str):
        if time_span == 'Weekly':
            quant_trend = self.df_filtered.groupby(['item_name', pd.Grouper(key = 'order_datetime', freq = 'W-MON', label = 'left')])['quantity'].sum().reset_index()
        elif time_span == 'Monthly':
            quant_trend = self.df_filtered.groupby(['item_name', pd.Grouper(key = 'order_datetime', freq = 'MS')])['quantity'].sum().reset_index()
        
        quant_trend_pivot = quant_trend.pivot(index = 'order_datetime', columns = 'item_name', values = 'line_total')
        quant_trend_pivot = quant_trend_pivot.fillna(0) #if the item didn't sell that week, then total quantity of item is 0

        return quant_trend_pivot

    # Order Frequency: number of unique orders that include the item
    def order_frequency(self):
        order_freq = (
            self.df_filtered
                .groupby('item_name', as_index=False)['order_id']
                .nunique()
                .rename(columns={'order_id': 'order_frequency'})
        )
        return order_freq
        
    
    # Merge and hand downstream as a single table 
    def build_truth_table(self, volatility_span: str = "Weekly"):
        total_rev = self.total_revenue()
        pct_rev = self.pct_total_revenue()[['item_name', 'pct_total_revenue']]
        qty = self.total_quantites_sold()
        orders = self.order_frequency()
        ticket = self.avg_ticket_contribution()
        vol = self.revenue_volatility(volatility_span)
    
        truth = total_rev.merge(pct_rev, on='item_name', how='left') \
                         .merge(qty, on='item_name', how='left') \
                         .merge(orders, on='item_name', how='left') \
                         .merge(ticket, on='item_name', how='left') \
                         .merge(vol, on='item_name', how='left')
    
        return truth.sort_values('total_revenue', ascending=False)


# ******************************************* Testing *******************************************
if __name__ == '__main__':
    os.chdir(r'/Users/cody/Desktop/Projects/hana-pilot-pos-analytics/data/raw')
    line_items = pd.read_csv('indian_food_pos_raw.csv') #https://www.kaggle.com/datasets/rajatsurana979/fast-food-sales-report?utm_source=chatgpt.com#

    test_obj = ItemRevenueObj(line_items, col_names)
    truth = test_obj.build_truth_table()
