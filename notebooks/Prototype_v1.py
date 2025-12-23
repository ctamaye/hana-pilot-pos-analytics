#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 21:46:46 2025

@author: cody

Prototype v1
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import calendar

#--------------------------------------- Data Prep ---------------------------------------
# Change 'order_datetime' column to datetime datatype
def basic_stats(df: pd.DataFrame):
    stats_types = df.info()
    stats_desc = df.describe()

    print(df.columns)
    print(stats_types)
    print(stats_desc)

    return stats_types, stats_desc
    




# ---------------------------------------------------------------------
# |       1. Sales in 2025 by item - histogram                         |
# |       2. Sales throughout year - line chart                        |
# |       3. Progression of qty of best item within each month         |
# ---------------------------------------------------------------------

# ------------------------ Global Plot Style ------------------------
def set_report_style():
    sns.set_theme(
        style="whitegrid",        # light grid, good for reports
        context="talk",           # slightly larger fonts
        palette="muted"           # soft, modern colors
    )

    plt.rcParams.update({
        "figure.figsize": (12, 5),
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.autolayout": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# Map weekday numbers to names
weekday_order = [0, 1, 2, 3, 4, 5, 6]
weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
weekday_map = dict(zip(weekday_order, weekday_labels))
        



class VisualObj:
    def __init__(self, df, col_names, year_wanted = None):
        self.df = df
        self.df = self.df.rename(columns = col_names)
        
        self.year_wanted = year_wanted
        

        
    def data_prep(self):
        # Apply global style
        set_report_style()
        
        # Set proper datatypes
        self.df['order_id'] = self.df['order_id'].astype(str)
        self.df['line_total'] = pd.to_numeric(self.df['line_total'], errors = 'coerce')
        self.df['quantity'] = pd.to_numeric(self.df['quantity'], errors = 'coerce').astype('Int64')
        self.df['item_name'] = self.df['item_name'].astype(str)
        self.df['order_datetime'] = pd.to_datetime(
            self.df['order_datetime'],
            errors = 'coerce',
            infer_datetime_format = True
            ) #ensures that hour elements are available
        
        # Info and stats
        print(self.df.dtypes, '\n')
        print(self.df.info(), '\n')
        print(self.df.describe(), '\n')
        
        # How many unique values?
        n_unique = self.df['order_id'].nunique()
        print(f'There are {n_unique} unique order ids, out of {self.df.shape[0]} original line items\n')

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

        self.df['order_year'] = pd.to_numeric(self.df['order_datetime'].dt.year, errors = 'coerce').astype('Int64')
        self.df['order_month'] = pd.to_numeric(self.df['order_datetime'].dt.month, errors = 'coerce').astype('Int64')


        # If year was provided when initialized, then filter data for only year == 'year'
        if self.year_wanted is not None:
            self.current_year_data = self.df.loc[self.df['order_year'] == self.year_wanted].copy()
        else:
            self.current_year_data = self.df.copy()
        
        self.current_year_data['order_date'] = self.current_year_data['order_datetime'].dt.date
        self.current_year_data['order_date'] = pd.to_datetime(self.current_year_data['order_date'])
        
        # Ensure supporting columns exist
        if 'weekday' not in self.current_year_data.columns:
            self.current_year_data['weekday'] = self.current_year_data['order_datetime'].dt.weekday
        

    def total_sales_weekday(self):
        sum_totals_weekday = (
            self.current_year_data
            .groupby('weekday', as_index=False)['line_total']
            .sum()
        )
        sum_totals_weekday['weekday_name'] = sum_totals_weekday['weekday'].map(weekday_map)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=sum_totals_weekday.sort_values('weekday'),
            x='weekday_name',
            y='line_total',
            ax=ax
        )
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel('Total Amount Sold ($)')
        if self.year_wanted is not None:
            ax.set_title(f'Total Sales by Day of Week in {self.year_wanted}')
        else:
            ax.set_title('Total Sales by Day of Week')
            
        # Add value labels on top of bars (optional but very “report-y”)
        for container in ax.containers:
            ax.bar_label(container, fmt='${:,.0f}'.format, padding=3)
        
        plt.tight_layout()
        plt.show()
       
    def avg_sales_by_hour_by_day(self):
        df = self.current_year_data.copy()
    
        # Ensure datetime
        df['order_datetime'] = pd.to_datetime(df['order_datetime'], errors='coerce')
        df = df.dropna(subset=['order_datetime'])
    
        # Ensure numeric sales
        df['line_total'] = pd.to_numeric(df['line_total'], errors='coerce')
        df = df.dropna(subset=['line_total'])
    
        # Ensure weekday exists and is int 0-6
        if 'weekday' not in df.columns:
            df['weekday'] = df['order_datetime'].dt.weekday
        else:
            # if it's names, map back to ints if possible
            if df['weekday'].dtype == 'O':
                inv_map = {v: k for k, v in weekday_map.items()}
                df['weekday'] = df['weekday'].map(inv_map)
    
        df['order_date'] = pd.to_datetime(df['order_datetime'].dt.date)
        df['order_hour'] = df['order_datetime'].dt.hour
    
        # Aggregate to store-hour totals per day (recommended for POS line-item data)
        order_hour = (
            df.groupby(['order_date', 'weekday', 'order_hour'], as_index=False)['line_total']
              .sum()
              .rename(columns={'line_total': 'hourly_sales'})
        )
    
        avg_hourly = (
            order_hour.groupby(['weekday', 'order_hour'], as_index=False)['hourly_sales']
                      .mean()
        )
    
        fig, ax = plt.subplots(figsize=(11, 5))
    
        for wd in weekday_order:  # [0..6]
            subset = avg_hourly[avg_hourly['weekday'] == wd].sort_values('order_hour')
            if subset.empty:
                continue
            ax.plot(subset['order_hour'], subset['hourly_sales'], linewidth=2, label=weekday_map[wd])
    
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Avg Sales per Hour ($)')
        ax.set_title(
            'Average Sales by Hour (Overlayed by Weekday)'
            + (f' — {self.year_wanted}' if self.year_wanted is not None else '')
        )
        ax.set_xticks(range(0, 24, 1))
        ax.legend(title='Day of Week', frameon=True, ncol=2)
        plt.tight_layout()
        plt.show()


        
    def total_sales_daily(self):
        sum_totals_daily = (
            self.current_year_data
            .groupby('order_date', as_index=False)['line_total']
            .sum()
            .rename(columns={'line_total': 'daily_sales'})
        )

        # Optional: add a 7-day rolling average to smooth noise
        sum_totals_daily['roll_7d'] = sum_totals_daily['daily_sales'].rolling(7, center=True).mean()

        fig, ax = plt.subplots(figsize=(12, 5))
        # Raw daily values (lighter)
        sns.lineplot(
            data=sum_totals_daily,
            x='order_date',
            y='daily_sales',
            ax=ax,
            alpha=0.4,
            linewidth=1,
            label='Daily Sales'
        )
        # Smoothed trend
        sns.lineplot(
            data=sum_totals_daily,
            x='order_date',
            y='roll_7d',
            ax=ax,
            linewidth=2.2,
            label='7-Day Rolling Avg'
        )

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))

        ax.set_xlabel('Month')
        ax.set_ylabel('Total Sold ($)')
        if self.year_wanted is not None:
            ax.set_title(f'Daily Sales and Trend in {self.year_wanted}')
        else:
            ax.set_title('Daily Sales and Trend')

        ax.legend(frameon=True)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    
    def best_item_per_month(self):
        monthly_item_sales = (
            self.current_year_data
            .groupby(['order_month', 'order_year', 'item_name'], as_index=False)['line_total']
            .sum()
        )
    
        # if you might have multiple years, group idxmax by BOTH month+year
        idx = monthly_item_sales.groupby(['order_year', 'order_month'])['line_total'].idxmax()
    
        highest_grossing_item_month = (
            monthly_item_sales
            .loc[idx]
            .sort_values(['order_year', 'order_month'])
            .reset_index(drop=True)
        )
    
        print("Highest grossing items by month:")
        print(highest_grossing_item_month)
    
        for _, row in highest_grossing_item_month.iterrows():
            current_month = row['order_month']
            current_year = row['order_year']
            top_item = row['item_name']
    
            subset = self.current_year_data.loc[
                (self.current_year_data['order_month'] == current_month) &
                (self.current_year_data['order_year'] == current_year) &
                (self.current_year_data['item_name'] == top_item)
            ].copy()
    
            if subset.empty:
                continue
    
            daily_qty = (
                subset
                .groupby('order_date', as_index=False)['quantity']
                .sum()
            )
    
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=daily_qty, x='order_date', y='quantity', marker='o', linewidth=1.8, ax=ax)
    
            ax.set_title(
                f"{calendar.month_name[current_month]} {current_year} – Daily Quantity of Top Item\n"
                f"Item: {top_item}"
            )
            plt.tight_layout()
            plt.show()

        
    def item_heatmap(self):
        matrix = (
            self.current_year_data
            .assign(value=1)
            .pivot_table(
                index='order_id', 
                columns='item_name', 
                values='value',
                aggfunc='max',
                fill_value=0
            )
        )

        co_matrix = matrix.T.dot(matrix)

        # Normalize (row-wise) for association strength
        norm = co_matrix.divide(co_matrix.max(axis=1).replace(0, np.nan), axis=0)
        np.fill_diagonal(norm.values, 0)

        # Optionally, limit to top N items by overall frequency for readability
        top_n = 20
        item_totals = co_matrix.sum(axis=1).sort_values(ascending=False)
        top_items = item_totals.head(top_n).index
        norm_top = norm.loc[top_items, top_items]

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            norm_top,
            ax=ax,
            cmap="Blues",
            square=True,
            cbar_kws={"label": "Relative Co-Occurrence Strength"},
        )
        ax.set_title("Item Co-Occurrence Heatmap (Top Items)")
        ax.set_xlabel("Item")
        ax.set_ylabel("Item")
        plt.tight_layout()
        plt.show()

        monthly_item_sales = (
            self.current_year_data
            .groupby(['order_month', 'item_name'], as_index=False)['line_total']
            .sum()
        )
        
        # keep if you want to return things later
        item_count_by_month = monthly_item_sales
        # return item_count_by_month, highest_grossing_item_month

    
#------------------------------------ Run Script ------------------------------------
# Import and prepare data for test
os.chdir(r'/Users/cody/Desktop/Projects/hana-pilot-pos-analytics/data/raw')
line_items = pd.read_csv('indian_food_pos_raw.csv')


if __name__ == '__main__':
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




