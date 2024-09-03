import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def SMA(data, period):
    """
    Calculate the Simple Moving Average (SMA) for a given dataset and period.

    Parameters:
    data (pandas.Series): The dataset containing price data.
    period (int): The number of periods over which to calculate the SMA.

    Returns:
    pandas.Series: The SMA values.
    """
    return data.rolling(window=period).mean()

def EMA(data, period):
    """
    Calculate the Exponential Moving Average (EMA) for a given dataset and period.

    Parameters:
    data (pandas.Series): The dataset containing price data.
    period (int): The number of periods over which to calculate the EMA.

    Returns:
    pandas.Series: The EMA values.
    """
    return data.ewm(span=period, adjust=False).mean()

def RSI(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given dataset and period.

    Parameters:
    data (pandas.Series): The dataset containing price data.
    period (int, optional): The number of periods over which to calculate the RSI. Default is 14.

    Returns:
    pandas.Series: The RSI values.
    """
    if len(data) < period:
        period = len(data)
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 0  # Ensure the initial period values are set to 0
    return rsi

def MACD(data, short_period=12, long_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given dataset.

    Parameters:
    data (pandas.Series): The dataset containing price data.
    short_period (int, optional): The short period for the fast EMA. Default is 12.
    long_period (int, optional): The long period for the slow EMA. Default is 26.
    signal_period (int, optional): The period for the signal line. Default is 9.

    Returns:
    tuple: A tuple containing three pandas.Series:
        - macd: The MACD line.
        - signal: The signal line.
        - histogram: The MACD histogram.
    """
    short_ema = EMA(data, short_period)
    long_ema = EMA(data, long_period)
    macd = short_ema - long_ema
    signal = EMA(macd, signal_period)
    histogram = macd - signal
    return macd, signal, histogram

def BollingerBands(data, period=20):
    """
    Calculate Bollinger Bands for a given dataset and period.

    Parameters:
    data (pandas.Series): The dataset containing price data.
    period (int, optional): The number of periods over which to calculate the bands. Default is 20.

    Returns:
    tuple: A tuple containing three pandas.Series:
        - sma: The Simple Moving Average (middle band).
        - upper_band: The upper band (sma + 2 * standard deviation).
        - lower_band: The lower band (sma - 2 * standard deviation).
    """
    sma = SMA(data, period)
    std = data.rolling(window=period).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return sma, upper_band, lower_band

def StochasticOscillator(data, period=14):
    """
    Calculate the Stochastic Oscillator for a given dataset and period.

    Parameters:
    data (pandas.Series): The dataset containing price data.
    period (int, optional): The number of periods over which to calculate the oscillator. Default is 14.

    Returns:
    tuple: A tuple containing two pandas.Series:
        - k_percent: The %K line of the Stochastic Oscillator.
        - d_percent: The %D line, which is a 3-period moving average of %K.
    """
    low_min = data.rolling(window=period).min()
    high_max = data.rolling(window=period).max()
    k_percent = (data - low_min) * 100 / (high_max - low_min)
    d_percent = k_percent.rolling(window=3).mean()
    return k_percent, d_percent

def ROC(data, period=12):
    """
    Calculate the Rate of Change (ROC) for a given dataset and period.

    Parameters:
    data (pandas.Series): The dataset containing price data.
    period (int, optional): The number of periods over which to calculate the ROC. Default is 12.

    Returns:
    pandas.Series: The ROC values.
    """
    return (data.diff(period) / data.shift(period)) * 100

def calculate_vpt(dc, dv):
    """
    
    Calculate the Volume Price Trend (VPT) indicator.
    
    Parameters:
    dc (pd.DataFrame): A pandas DataFrame for data close.
    dv (pd.DataFrame): A pandas DataFrame for data volume.
    
    Returns:
    pd.Series: A pandas Series representing the VPT values.
    """
    # Convert lists to pandas Series
    dc_series = pd.Series(dc)
    dv_series = pd.Series(dv)
    
    # Calculate the price change percentage
    price_change_percentage = dc_series.pct_change()
    
    # Calculate the VPT
    vpt = (price_change_percentage * dv_series).cumsum()
    
    return vpt

def calculate_obv(close_prices, volumes):
    """
    Calculate the On-Balance Volume (OBV) indicator.
    
    Parameters:
    close_prices (list of float): A list of closing prices.
    volumes (list of float): A list of traded volumes.
    
    Returns:
    list: A list representing the OBV values.
    """
    obv = [0]  # OBV starts with 0 for the first value
    
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif close_prices[i] < close_prices[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    
    return obv

def normalize_series(series):
    """
    Normalize a pandas Series to a range of -1 to 1.

    Parameters:
    series (pandas.Series): The input series to normalize.

    Returns:
    pandas.Series: The normalized series.
    """
    return (series - series.min()) / (series.max() - series.min()) * 2 - 1

def combined_indicator(close_prices, volumes):
    """
    Calculate a combined indicator using OBV and Bollinger Bands.

    Parameters:
    close_prices (pandas.Series): The series of closing prices.
    volumes (pandas.Series): The series of trading volumes.

    Returns:
    pandas.Series: The combined indicator values.
    """
    obv = calculate_obv(close_prices, volumes)
    sma, upper_band, lower_band = BollingerBands(close_prices)
    
    obv_normalized = normalize_series(pd.Series(obv))
    
    position_in_bands = (2 * (pd.Series(close_prices) - lower_band) / (upper_band - lower_band)) - 1
    
    combined_values = 0.5 * obv_normalized + 0.5 * position_in_bands
    combined_values = combined_values.fillna(0)  # Handling NaN values
    
    return combined_values

def calculate_delta_obv_custom(close_prices, volumes):
    """
    Calculate the delta of OBV and a custom combined indicator, then normalize the result.

    Parameters:
    close_prices (pandas.Series): The series of closing prices.
    volumes (pandas.Series): The series of trading volumes.

    Returns:
    list: The normalized delta values as a list.
    """
    obv = calculate_obv(close_prices, volumes)
    custom_indicator = combined_indicator(close_prices, volumes)
    
    delta = normalize_series(pd.Series(obv)) % custom_indicator
    
    return normalize_series(delta).tolist()

def price_increase_predictor(close_prices, volumes):
    """
    Predict price increases using several technical indicators.

    Parameters:
    close_prices (pandas.Series): The series of closing prices.
    volumes (pandas.Series): The series of trading volumes.

    Returns:
    pandas.Series: The combined signal for predicting price increases.
    """
    # Calculate indicators
    rsi = RSI(close_prices)
    macd, signal, histogram = MACD(close_prices)
    obv = pd.Series(calculate_obv(close_prices, volumes))
    combined = combined_indicator(close_prices, volumes)
    
    # Normalize indicators
    rsi_normalized = normalize_series(rsi)
    macd_normalized = normalize_series(macd - signal)
    obv_normalized = normalize_series(obv)
    combined_normalized = normalize_series(combined)
    
    # Define weights for each indicator
    weights = {
        'rsi': 0.333,
        'macd': 0.333,
        'obv': 0.333,
        'combined': 0.0
    }
    
    # Calculate combined signal
    signal = (
        weights['rsi'] * rsi_normalized +
        weights['macd'] * macd_normalized +
        weights['obv'] * obv_normalized +
        weights['combined'] * combined_normalized
    )
    
    # Handle NaN values
    signal = signal.fillna(0)
    
    return signal

# Function to calculate positive and negative volume
def calculate_positive_negative_volume(close_prices, volumes):
    positive_volume = [0]  # Start with an initial value to match the length
    negative_volume = [0]  # Start with an initial value to match the length
    
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i - 1]:
            positive_volume.append(volumes[i])
            negative_volume.append(0)
        elif close_prices[i] < close_prices[i - 1]:
            positive_volume.append(0)
            negative_volume.append(volumes[i])
        else:
            positive_volume.append(0)
            negative_volume.append(0)
            
    return positive_volume, negative_volume

def calculate_positive_negative_volume_normalized(close_prices, volumes):
    positive_volume = [0]  # Start with an initial value to match the length
    negative_volume = [0]  # Start with an initial value to match the length
    
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i - 1]:
            positive_volume.append(volumes[i])
            negative_volume.append(0)
        elif close_prices[i] < close_prices[i - 1]:
            positive_volume.append(0)
            negative_volume.append(volumes[i])
        else:
            positive_volume.append(0)
            negative_volume.append(0)
            
    # Normalize positive volume to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    positive_volume = scaler.fit_transform(np.array(positive_volume).reshape(-1, 1)).flatten()
    
    # Normalize negative volume to [-1, 0]
    negative_volume = scaler.fit_transform(np.array(negative_volume).reshape(-1, 1)).flatten()
    negative_volume = -negative_volume  # Scale to negative values

    return positive_volume.tolist(), negative_volume.tolist()