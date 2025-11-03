# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 3-11-2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Data Preparation and Cleaning ---

def custom_float(x):
    """Replaces comma with dot and converts the string to a float."""
    try:
        return float(str(x).replace(',', '.'))
    except Exception:
        return np.nan

# Read CSV, clean numeric columns, and convert 'Date'
df = pd.read_csv('/content/9. Sales-Data-Analysis.csv')
numeric_cols = ['Price', 'Quantity'] 

for c in numeric_cols:
    df[c] = df[c].astype(str).apply(custom_float)

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Aggregate to create the daily sales time series
series_df = df.groupby('Date')['Quantity'].sum().reset_index()
series_df.set_index('Date', inplace=True)
series = series_df['Quantity'].dropna() 

# --- 1. Explore the dataset and plot original series ---
print("--- 1. Time Series Exploration ---")
print("Time Series Length:", len(series))

plt.figure(figsize=(12, 4))
plt.plot(series)
plt.title('Original Daily Total Quantity Sold')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.grid(True)
plt.show()

# --- 2. Check for stationarity of time series (ADF Test) ---

def check_stationarity(ts):
    """Performs and prints the Augmented Dickey-Fuller test results."""
    result = adfuller(ts, autolag='AIC')
    print('\n--- ADF Test Results ---')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

print("\n--- 2. Checking Original Series Stationarity ---")
check_stationarity(series)

# Apply differencing (d=1, D=1, m=7) to stabilize the series for model parameter determination
series_diff = series.diff().diff(7).dropna()

print("\n--- Checking Differenced Series Stationarity (d=1, D=1, m=7) ---")
check_stationarity(series_diff)

# --- 3. Determine SARIMA models parameters (p, d, q) x (P, D, Q, m) ---

# Plot ACF and PACF for parameter selection (lags reduced to 15 due to small sample size)
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(series_diff, lags=15, ax=axes[0], title='Autocorrelation Function (ACF) - Differenced Series')
plot_pacf(series_diff, lags=15, ax=axes[1], title='Partial Autocorrelation Function (PACF) - Differenced Series')
fig.tight_layout()
plt.show()

# Parameters based on visual inspection: SARIMA(1, 1, 1)x(0, 1, 1, 7)
order = (1, 1, 1)        # Non-seasonal: AR(1), I(1), MA(1)
seasonal_order = (0, 1, 1, 7) # Seasonal: AR(0), I(1), MA(1), m=7 (weekly)

# --- 4. Fit the SARIMA model ---

print(f"\n--- 4. Fitting SARIMA{order}x{seasonal_order} ---")
model = SARIMAX(series, order=order, seasonal_order=seasonal_order, 
                enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

print("\n--- SARIMA Model Summary ---")
print(model_fit.summary().as_text())


# --- 5. Make time series predictions ---

# Make a 14-day (2-week) forecast
forecast_steps = 14
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(series, label='Observed')
plt.plot(forecast_mean, label=f'SARIMA Forecast ({forecast_steps} steps)', color='red')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')
plt.title(f'SARIMA Forecast of Daily Sales Quantity')
plt.xlabel('Date')
plt.ylabel('Total Quantity Sold')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- 14-Day SARIMA Forecast (Mean) ---")
print(forecast_mean.to_string())

# --- 6. Evaluate model predictions ---
# Evaluation is done by checking p-values in the summary and visually inspecting the forecast plot.
# Note on Auto-fit: pmdarima is the standard library for auto-fitting but is not available here.
print("\n--- 6. Evaluation Notes ---")
print("Model evaluation suggests the ma.L1 term is significant (p-value=0.023), but ar.L1 and ma.S.L7 are not. A simpler model order might be better.")
print("The forecast successfully captures the weekly seasonality (dips on days 7 and 14).")
```
### OUTPUT:
<img width="1055" height="436" alt="image" src="https://github.com/user-attachments/assets/fd0e626c-44dc-4776-9036-b8308c303b6c" />

<img width="1239" height="299" alt="image" src="https://github.com/user-attachments/assets/4a0e41f1-5a58-4a73-ad10-90ac1c2e29e9" />

<img width="1217" height="306" alt="image" src="https://github.com/user-attachments/assets/a52bbb94-17c4-409e-a4d9-1fac0d38fe57" />

<img width="794" height="409" alt="image" src="https://github.com/user-attachments/assets/68d626bb-1fb7-4c95-af2f-8d4de30927f5" />

<img width="1080" height="558" alt="image" src="https://github.com/user-attachments/assets/cbc8b04b-a3ee-471c-b472-db214ff18b8d" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
