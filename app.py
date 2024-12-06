import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict

def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """Remove outliers using the Interquartile Range method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def get_zillow_data() -> List[Dict]:
    """Fetch Zillow data"""
    url = 'https://www.zillow.com/async-create-search-page-state'
    all_data = []
    
    headers = {
        'accept': '*/*',
        'accept-language': 'en-GB,en;q=0.9',
        'content-type': 'application/json',
        'origin': 'https://www.zillow.com',
        'referer': 'https://www.zillow.com/los-angeles-ca/sold/',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
    }
    
    with st.spinner('Fetching data from Zillow...'):
        progress_bar = st.progress(0)
        
        for page in range(1, 9):
            try:
                progress_bar.progress(page / 8)
                
                payload = {
                    "searchQueryState": {
                        "pagination": {"currentPage": page},
                        "isMapVisible": False,
                        "mapBounds": {
                            "west": -118.88551790039062,
                            "east": -117.93794709960937,
                            "south": 33.63490745983598,
                            "north": 34.405484226648646
                        },
                        "regionSelection": [
                            {"regionId": 12447, "regionType": 6}
                        ],
                        "filterState": {
                            "sortSelection": {"value": "globalrelevanceex"},
                            "isForSaleByAgent": {"value": False},
                            "isForSaleByOwner": {"value": False},
                            "isNewConstruction": {"value": False},
                            "isComingSoon": {"value": False},
                            "isAuction": {"value": False},
                            "isForSaleForeclosure": {"value": False},
                            "isRecentlySold": {"value": True},
                            "doz": {"value": "7"}
                        },
                        "isListVisible": True
                    },
                    "wants": {"cat1": ["listResults"]},
                    "requestId": 2,
                    "isDebugRequest": False
                }

                response = requests.put(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    listings = data['cat1']['searchResults']['listResults']
                    
                    for listing in listings:
                        try:
                            if listing['hdpData']['homeInfo'].get('livingArea'):
                                date_sold_ms = listing['hdpData']['homeInfo']['dateSold']
                                date_sold = datetime.fromtimestamp(date_sold_ms/1000)
                                
                                all_data.append({
                                    'date': date_sold,
                                    'price_per_sqft': listing['unformattedPrice'] / listing['hdpData']['homeInfo']['livingArea'],
                                    'address': listing['address'],
                                    'price': listing['unformattedPrice'],
                                    'sqft': listing['hdpData']['homeInfo']['livingArea']
                                })
                        except (KeyError, TypeError, ZeroDivisionError):
                            continue
                else:
                    st.error(f"Error on page {page}: {response.status_code}")
                    break
                    
            except Exception as e:
                st.error(f"Error processing page {page}: {str(e)}")
                break
        
        progress_bar.empty()
    
    return all_data

def calculate_price_feed(data: List[Dict], window_size: int = 3, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Calculate the price feed with user-specified parameters"""
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    
    results = []
    unique_dates = sorted(df['date'].unique())
    
    for date in unique_dates:
        day_data = df[df['date'] == date].copy()
        
        if len(day_data) > 0:
            clean_data = remove_outliers_iqr(day_data, 'price_per_sqft', multiplier=iqr_multiplier)
            
            if len(clean_data) > 0:
                daily_price = clean_data['price_per_sqft'].median()
                
                results.append({
                    'date': date,
                    'price_per_sqft': daily_price,
                    'transaction_count': len(clean_data)
                })
    
    results_df = pd.DataFrame(results)
    
    results_df['price_per_sqft_raw'] = results_df['price_per_sqft']
    results_df['price_per_sqft'] = results_df['price_per_sqft'].rolling(
        window=window_size, min_periods=window_size, center=True
    ).mean()
    
    return results_df.dropna(subset=['price_per_sqft'])

def main():
    st.set_page_config(page_title="Real Estate Price Feed", layout="wide")
    
    st.title("Real Estate Price Feed Analysis")
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    window_size = st.sidebar.slider("Rolling Window Size", min_value=1, max_value=7, value=3)
    iqr_multiplier = st.sidebar.slider("IQR Multiplier", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    
    # Main content
    if 'data' not in st.session_state:
        if st.button("Fetch New Data"):
            data = get_zillow_data()
            if data:
                st.session_state.data = data
                st.success(f"Successfully collected {len(data)} listings")
            else:
                st.error("Failed to collect data")
    
    if 'data' in st.session_state:
        data = st.session_state.data
        
        # Calculate price feed
        df = calculate_price_feed(data, window_size, iqr_multiplier)
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['price_per_sqft_raw'],
                mode='lines',
                name='Raw Daily Median',
                line=dict(color='gray', width=1),
                opacity=0.5
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['price_per_sqft'],
                mode='lines',
                name=f'{window_size}-Day Rolling Average',
                line=dict(color='blue', width=2)
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['transaction_count'],
                name='Transaction Count',
                yaxis='y2',
                opacity=0.3,
                marker_color='lightgray'
            )
        )
        
        fig.update_layout(
            title="Price per Square Foot Over Time",
            xaxis_title="Date",
            yaxis_title="Price per Square Foot ($)",
            yaxis2=dict(
                title="Transaction Count",
                overlaying="y",
                side="right"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Initial Price/sqft", f"${df['price_per_sqft'].iloc[0]:,.2f}")
        with col2:
            st.metric("Final Price/sqft", f"${df['price_per_sqft'].iloc[-1]:,.2f}")
        with col3:
            change = ((df['price_per_sqft'].iloc[-1] / df['price_per_sqft'].iloc[0] - 1) * 100)
            st.metric("Total Change", f"{change:,.2f}%")
        with col4:
            st.metric("Avg Daily Transactions", f"{df['transaction_count'].mean():,.1f}")

if __name__ == "__main__":
    main()