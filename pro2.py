import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit App
st.title("Time Series Forecasting App")

# File upload section
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type="xlsx")
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    # Initial Data Exploration
    st.write("### Initial Data Exploration")
    st.write(f"Length of dataset: {len(df)}")
    st.write("Data Types:", df.dtypes)
    st.write("Shape of dataset:", df.shape)
    st.write("Missing values per column:", df.isnull().sum())
    
    # Drop missing values
    df.dropna(inplace=True)
    st.write(f"Length after dropping missing values: {len(df)}")
    
    # Drop unnecessary columns
    unnecessary_cols = ['Ship No', 'Sold-To Code', 'Sold-To City', 'Ctn Unit', 'Ctn Qty', 'Ctn Dt', 'Contrct Dt', 
                        'Ship Dt', 'Dlv Unit', 'Dlv Loc', 'Mat Code']
    df1 = df.drop(unnecessary_cols, axis=1)
    st.write("### Dataset After Dropping Columns")
    st.dataframe(df1.head())
    
    # Outlier treatment
    Q1 = df1['Dlv Qty'].quantile(0.25)
    Q3 = df1['Dlv Qty'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = df1['Dlv Qty'].median()
    df1['Dlv Qty'] = df1['Dlv Qty'].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
    
    # Set index to delivery date
    df1.set_index('Dlv Dt', inplace=True)
    
    # Time series plot
    st.write("### Time Series Plot")
    st.line_chart(df1['Dlv Qty'])
    
    # Train-test split
    split_point = int(len(df1) * 0.8)
    train = df1[:split_point]
    test = df1[split_point:]
    
    st.write("### Train-Test Split")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train['Dlv Qty'], label='Training Data')
    ax.plot(test['Dlv Qty'], label='Testing Data', color='orange')
    ax.set_title("Train-Test Split")
    ax.set_xlabel("Date")
    ax.set_ylabel("Delivery Quantity")
    ax.legend()
    st.pyplot(fig)
    
    # ADF Test
    adf_test = adfuller(train['Dlv Qty'])
    st.write(f"### ADF Test Results")
    st.write(f"ADF Test p-value: {adf_test[1]}")
    
    # ARIMA model
    st.write("### ARIMA Model")
    model = ARIMA(train['Dlv Qty'], order=(2, 1, 4))
    model_fit = model.fit()
    st.text(model_fit.summary())
    
    # Residual analysis
    st.write("### Residual Analysis")
    residuals = model_fit.resid[3:]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    st.pyplot(fig)
    
    # Forecasting
    st.write("### Forecasting")
    forecast_test = model_fit.forecast(len(test))
    df1['Forecast_Manual'] = [None] * len(train) + list(forecast_test)
    
    # Auto ARIMA
    st.write("### Auto ARIMA Model")
    auto_model = auto_arima(train['Dlv Qty'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    st.text(auto_model.summary())
    forecast_test_auto = auto_model.predict(n_periods=len(test))
    df1['Forecast_Auto'] = [None] * len(train) + list(forecast_test_auto)
    
    # Metrics
    st.write("### Model Performance Metrics")
    mae_manual = mean_squared_error(test['Dlv Qty'], forecast_test, squared=False)
    mape_manual = np.mean(np.abs((test['Dlv Qty'] - forecast_test) / test['Dlv Qty']))
    rmse_manual = sqrt(mean_squared_error(test['Dlv Qty'], forecast_test))
    
    st.write(f"Manual ARIMA - MAE: {mae_manual}, MAPE: {mape_manual}, RMSE: {rmse_manual}")
    
    mae_auto = mean_squared_error(test['Dlv Qty'], forecast_test_auto, squared=False)
    mape_auto = np.mean(np.abs((test['Dlv Qty'] - forecast_test_auto) / test['Dlv Qty']))
    rmse_auto = sqrt(mean_squared_error(test['Dlv Qty'], forecast_test_auto))
    
    st.write(f"Auto ARIMA - MAE: {mae_auto}, MAPE: {mape_auto}, RMSE: {rmse_auto}")

    # Display forecast results
    st.write("### Forecasted Results")
    st.line_chart(df1[['Dlv Qty', 'Forecast_Manual', 'Forecast_Auto']])
else:
    st.warning("Please upload a dataset to proceed.")
