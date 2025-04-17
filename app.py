import numpy as np
import pandas as pd
import yfinance as yf
# Replace Keras import with joblib for loading scikit-learn models
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import os
import requests

# Use scikit-learn model instead of Keras model
model_path = 'gradient_boosting_stock_model.joblib'
# Only load the model if the file exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success(f"Model loaded from {model_path}")
else:
    # Try the random forest model as fallback
    model_path = 'random_forest_stock_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success(f"Model loaded from {model_path}")
    else:
        st.error(f"No model files found. Please train models first.")
        model = None

st.header('Stock Market Predictor')

# Function to get stock symbol from company name using multiple approaches
def get_stock_symbol(company_name):
    # Fallback for common companies with direct mapping
    common_companies = {
        'google': ('GOOGL', 'Alphabet Inc.'),
        'alphabet': ('GOOGL', 'Alphabet Inc.'),
        'apple': ('AAPL', 'Apple Inc.'),
        'microsoft': ('MSFT', 'Microsoft Corporation'),
        'amazon': ('AMZN', 'Amazon.com, Inc.'),
        'tesla': ('TSLA', 'Tesla, Inc.'),
        'facebook': ('META', 'Meta Platforms, Inc.'),
        'meta': ('META', 'Meta Platforms, Inc.'),
        'netflix': ('NFLX', 'Netflix, Inc.'),
        'nvidia': ('NVDA', 'NVIDIA Corporation'),
        'walmart': ('WMT', 'Walmart Inc.'),
        'disney': ('DIS', 'The Walt Disney Company')
    }
    
    try:
        # Check if the company name is in our common companies dictionary (case insensitive)
        company_lower = company_name.lower()
        if company_lower in common_companies:
            return common_companies[company_lower]
            
        # Try using Yahoo Finance API endpoint with proper headers
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}&quotesCount=1&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if 'quotes' in data and len(data['quotes']) > 0:
                symbol = data['quotes'][0]['symbol']
                name = data['quotes'][0].get('longname') or data['quotes'][0].get('shortname', company_name)
                return symbol, name
                
        return None, None
    except Exception as e:
        st.warning(f"Error searching for company: {e}")
        return None, None

# Create two columns for company name input and stock symbol display
col1, col2 = st.columns([3, 1])

with col1:
    company_name = st.text_input('Enter Company Name', 'Google')

# Default stock symbol
stock = "GOOG"
company_full_name = "Google Inc."

if company_name:
    symbol, full_name = get_stock_symbol(company_name)
    if symbol:
        stock = symbol
        company_full_name = full_name
        with col2:
            st.info(f"Symbol: {stock}")
    else:
        st.warning(f"Could not find stock symbol for '{company_name}'. Using default: GOOG")

# Display company info
st.subheader(f"{company_full_name} ({stock}) Stock Prediction")

start = '2012-01-01'
end = '2022-12-31'

# Try to download data with error handling
try:
    data = yf.download(stock, start, end)
    if data.empty:
        st.error(f"No data found for {stock}. Please check the company name or symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data for {stock}: {e}")
    st.stop()

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader(f'{company_full_name} - Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.title(f"{company_full_name} ({stock})")
plt.show()
st.pyplot(fig1)

st.subheader(f'{company_full_name} - Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.title(f"{company_full_name} ({stock})")
plt.show()
st.pyplot(fig2)

st.subheader(f'{company_full_name} - Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.title(f"{company_full_name} ({stock})")
plt.show()
st.pyplot(fig3)

if model:
    # Modified prediction section for scikit-learn models
    x = []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i].flatten())  # Flatten for scikit-learn models
    
    x = np.array(x)
    
    # Use the scikit-learn model for prediction
    predict = model.predict(x)
    
    # Get the actual values
    y = data_test_scale[100:, 0]
    
    # Scale back to original values
    scale = 1/scaler.scale_[0]
    predict = predict * scale
    y = y * scale
    
    st.subheader(f'{company_full_name} - Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Predicted Price')
    plt.plot(y, 'g', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title(f"{company_full_name} ({stock}) - Prediction")
    plt.show()
    st.pyplot(fig4)