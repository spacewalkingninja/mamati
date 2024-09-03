import eel
import pandas as pd
import time
import threading
import numpy as np
from mexc_api.spot import Spot
from mexc_api.websocket import SpotWebsocketStreamClient
from mexc_api.common.enums import Side, OrderType, StreamInterval, Action
from mexc_api.common.exceptions import MexcAPIError
from mexc_api.common.enums import Interval as m_intervals
import statistic_anal as anal
from price_monitor import PriceMonitor
from liquidity_monitor import LiquidityMonitor

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.statespace.sarimax import SARIMAX
import sys
import json
import os 
import traceback


import random
import faulthandler
faulthandler.enable()

#os.system("taskkill /f /im geckodriver.exe /T")
#os.system("taskkill /f /im chrome.exe /T")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

SYMBOL="BTCUSDT"

if len(sys.argv) > 1:
    SYMBOL = sys.argv[1]

print(f"INIT FOR {SYMBOL}")


#MEXC CREDENTIALS!
KEY = ""
SECRET = ""

GRS=0
MDIC={}
MB=[]
POSITION={}
TOTAL_CAPTURED=0
TOTAL_SHORTED=0
SHORT_CLOSE = []
LONG_CLOSE = []
JUST_CLOSED = False
CLOSED_MSG = ''
TGT_STOP = 0
FLOAT_VOL = 0
FLOAT_DIR = 0

spot = Spot(KEY, SECRET)

server_time = spot.market.server_time()
print(server_time)

_LAST_DATA = {}



def log_traceback(ex):
    tb_lines = traceback.format_exception(ex.__class__, ex, ex.__traceback__)
    tb_text = ''.join(tb_lines)
    # I'll let you implement the ExceptionLogger class,
    # and the timestamping.
    print(f"###LOG###:::{tb_text}")


def find_thresholds(series, num_std=1.5):
    """
    Calculate dynamic thresholds based on statistical properties of the series.
    
    Parameters:
    - series: pandas Series, the data series to analyze.
    - num_std: float, the number of standard deviations from the mean to set as thresholds.
    
    Returns:
    - high_threshold: float, the threshold for high values.
    - low_threshold: float, the threshold for low values.
    """
    mean = series.mean()
    std = series.std()
    high_threshold = mean + num_std * std
    low_threshold = mean - num_std * std
    return high_threshold, low_threshold


def find_average_moves(interval, high_threshold, low_threshold):
    """
    Find the average moves to reach high or low threshold in an interval.
    
    Parameters:
    - interval: pandas Series, the interval of data to analyze.
    - high_threshold: float, the threshold for high values.
    - low_threshold: float, the threshold for low values.
    
    Returns:
    - moves_to_high: float, the number of moves to reach the high threshold (or NaN if not reached).
    - moves_to_low: float, the number of moves to reach the low threshold (or NaN if not reached).
    """
    moves_to_high = []
    moves_to_low = []
    
    for i in range(len(interval)):
        if interval.iloc[i] >= high_threshold:
            moves_to_high.append(i + 1)  # +1 because move count starts from 1
            break
    else:
        moves_to_high.append(np.nan)  # NaN if high is never reached in the interval
    
    for i in range(len(interval)):
        if interval.iloc[i] <= low_threshold:
            moves_to_low.append(i + 1)
            break
    else:
        moves_to_low.append(np.nan)  # NaN if low is never reached in the interval
    
    return moves_to_high[0], moves_to_low[0]


def calculate_direction_of_flow(series):
    return series.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0).sum()

def calculate_peaks(series):
    high_peaks = ((series.shift(1) < series) & (series.shift(-1) < series)).sum()
    low_peaks = ((series.shift(1) > series) & (series.shift(-1) > series)).sum()
    return high_peaks, low_peaks

def analyze_split(df):
    results = {}
    for col in df.columns:
        if col != 'RAW_CT' and col != 'KLINES':
            results[col] = {
                'direction_of_flow': calculate_direction_of_flow(df[col]),
                'close': df[col].iloc[-1],
                'open': df[col].iloc[0],
                'min': df[col].min(),
                'max': df[col].max(),
                'high_peaks': calculate_peaks(df[col])[0],
                'low_peaks': calculate_peaks(df[col])[1],
            }

    upper_bb_touches = (df['RAW_S'] >= df['UPPER_BB']).sum()
    lower_bb_touches = (df['RAW_S'] <= df['LOWER_BB']).sum()
    sma_above_ema = (df['SMA'] > df['EMA']).sum()

    results['upper_bb_touches'] = upper_bb_touches
    results['lower_bb_touches'] = lower_bb_touches
    results['sma_above_ema'] = sma_above_ema

    klines_above_sma = 0
    klines_below_sma = 0
    klines_above_ema = 0
    klines_below_ema = 0
    klines_above_upper_bb = 0
    klines_below_lower_bb = 0
    
    for i in range(len(df)):
        sma = float(df['SMA'].iloc[i])
        ema = float(df['EMA'].iloc[i])
        upper_bb = float(df['UPPER_BB'].iloc[i])
        lower_bb = float(df['LOWER_BB'].iloc[i])

        for kline in df['KLINES']:
            if float(kline['c']) > sma or float(kline['h']) > sma:
                klines_above_sma += 1
            if float(kline['c']) < sma or float(kline['l']) < sma:
                klines_below_sma += 1
            if float(kline['c']) > ema or float(kline['h']) > ema:
                klines_above_ema += 1
            if float(kline['c']) < ema or float(kline['l']) < ema:
                klines_below_ema += 1
            if float(kline['c']) > upper_bb or float(kline['h']) > upper_bb:
                klines_above_upper_bb += 1
            if float(kline['c']) < lower_bb or float(kline['l']) < lower_bb:
                klines_below_lower_bb += 1

    results['klines_above_sma'] = klines_above_sma
    results['klines_below_sma'] = klines_below_sma
    results['klines_above_ema'] = klines_above_ema
    results['klines_below_ema'] = klines_below_ema
    results['klines_above_upper_bb'] = klines_above_upper_bb
    results['klines_below_lower_bb'] = klines_below_lower_bb

    return results

def split_and_analyze(df):
    intervals = {
        '50_50': len(df) // 2,
        'last_2h': 120,
        'last_1h': 60,
        'last_30min': 30,
        'last_15min': 15,
        'last_10min': 10,
        'last_5min': 5,
    }

    analysis_results = {}

    #Normal is this: but this includes last 5 min always
    #for label, length in intervals.items():
    #    split_df = df.iloc[-length:] if length < len(df) else df
    #    analysis_results[label] = analyze_split(split_df)
    
    #This wont include last 5 min except in last_5min interval 
    for label, length in intervals.items():
        if label == 'last_5min':
            # Use the last 5 minutes as is
            split_df = df.iloc[-length:] if length < len(df) else df
        else:
            # Exclude the last 5 minutes for other intervals
            split_df = df.iloc[-(length + 5):-5] if (length + 5) < len(df) else df.iloc[:-5]
        
        analysis_results[label] = analyze_split(split_df)

    return analysis_results



def identify_wave_and_predict(data, sep_data):
    # Example structure for the input data
    heatmaps = sep_data['HEATMAP']
    klines_data = list(data['KLINES'].values())
    projections = sep_data['PROJECTIONS']
    statistical_data = sep_data['ANALYSIS']
    
    # Placeholder functions for analysis (implement detailed logic for each)
    #def analyze_intervals(intervals):
        # Analyze intervals to determine current phase of Eliott wave
        # Return the identified wave phase and relevant metrics
    #    wave_phase = 'Wave 3'  # Example placeholder
    #    metrics = {}
    #    return wave_phase, metrics
    
    def analyze_heatmaps(klines, heatmap):
        upswings = 0
        downswings = 0
        buy_signals = {}
        sell_signals = {}
        mmin = 0.01
        mmax = 0
        
        #dataframe counter
        df_cb = 0
        df_cs = 0
        #dataframe columns
        p_col = []
        tl_col = []
        dt_col = []
        s_col = []

        p_col_n = []
        tl_col_n = []
        dt_col_n = []
        s_col_n = []

        b_locator = {}
        s_locator = {}

        try:
            for kline in klines:
                close_time = float(kline['x'])
                close_price = float(kline['c'])
                
                for hclose_time, prices_data in heatmap.items():
                    if float(hclose_time) < close_time:
                        prices_data_last = prices_data
                        continue
                    if float(hclose_time) >= close_time:
                        price_data = heatmap[hclose_time]
                        if mmin > mmax:
                            for key, value in price_data.items():
                                value = float(value[2])
                                if value < mmin:
                                    mmin = value
                                if value > mmax:
                                    mmax = value

                        for key, value in price_data.items():
                            price = float(value[2])
                            if close_price <= price and price_data[key][0] > 0.18:  # If the closing price is below the price in heatmap and NORMALIZED liquidity value is significant
                                #if price not in buy_signals:
                                #    buy_signals[key]=[price_data[key][0], 0, ((price - close_price) - mmin) / (mmax - mmin)] #first value is heatdata, second value is change over time, third value is importance of buy_signal (importance must be used in ASC order, lower val = higher importance)
                                #    downswings+=1
                                #if prices_data_last[key][0] < buy_signals[key][0]:
                                #    buy_signals[key]=[price_data[key][0], buy_signals[key][1] + (price_data[key][0] - prices_data_last[key][0]), ((price - close_price) - mmin) / (mmax - mmin)]

                                #improved: dataframe 

                                if key not in b_locator:
                                    b_locator[key] = df_cb
                                    p_col.append(float(key))
                                    tl_col.append(price_data[key][0])
                                    dt_col.append(0)
                                    s_col.append(((price - close_price) - mmin) / (mmax - mmin))
                                    df_cb += 1
                                    downswings += 1
                                else:
                                    if prices_data_last[key][0] < tl_col[b_locator[key]]:
                                        t_df_cb = b_locator[key]
                                        p_col[t_df_cb] = float(key)
                                        tl_col[t_df_cb] = price_data[key][0]
                                        dt_col[t_df_cb] = dt_col[t_df_cb] + (price_data[key][0] - prices_data_last[key][0])
                                        s_col[t_df_cb] = ((price - close_price) - mmin) / (mmax - mmin)

                            if close_price >= price and price_data[key][0] > 0.18:  
                                #if price not in sell_signals:
                                #    sell_signals[key]=[price_data[key][0], 0, ((close_price - price) - mmin) / (mmax - mmin)] #first value is heatdata, second value is change over time, third value is importance of buy_signal
                                #    upswings+=1
                                #if prices_data_last[key][0] < sell_signals[key][0]:
                                #    sell_signals[key]=[price_data[key][0], sell_signals[key][1] + (price_data[key][0] - prices_data_last[key][0]), ((close_price - price) - mmin) / (mmax - mmin)]

                                if key not in s_locator:
                                    s_locator[key] = df_cs
                                    p_col_n.append(float(key))
                                    tl_col_n.append(price_data[key][0])
                                    dt_col_n.append(0)
                                    s_col_n.append(((price - close_price) - mmin) / (mmax - mmin))
                                    df_cs += 1
                                    downswings += 1
                                else:
                                    # Inside the loop where the sell signals are being handled
                                    if key in prices_data_last and key in s_locator:
                                        # Ensure s_locator[key] is a valid index for tl_col
                                        if s_locator[key] < len(tl_col) and prices_data_last[key][0] < tl_col[s_locator[key]]:
                                            t_df_cs = s_locator[key]
                                            p_col_n[t_df_cs] = float(key)
                                            tl_col_n[t_df_cs] = price_data[key][0]
                                            dt_col_n[t_df_cs] += (price_data[key][0] - prices_data_last[key][0])
                                            s_col_n[t_df_cs] = abs(((price - close_price) - mmin) / (mmax - mmin))

                        break
            b_df = pd.DataFrame({
                'PRICES': p_col,
                'LIQUIDITY': tl_col,
                'DELTA_LIQ': dt_col,
                'PROXIMITY': s_col
            })
            s_df = pd.DataFrame({
                'PRICES': p_col_n,
                'LIQUIDITY': tl_col_n,
                'DELTA_LIQ': dt_col_n,
                'PROXIMITY': s_col_n
            })

        
            b_df = b_df.fillna(0)  # Replace NaN values with 0
            s_df = s_df.fillna(0)  # Replace NaN values with 0
            upswings = b_df['LIQUIDITY'].sum()
            downswings = s_df['LIQUIDITY'].sum()
            b_df = b_df.sort_values(by=['DELTA_LIQ', 'PRICES', 'LIQUIDITY',  'PROXIMITY'], ascending=[False, True, False, True])
            low_mean_upb = b_df['PRICES'].head(10).mean()
            b_df = b_df.sort_values(by=['LIQUIDITY', 'PRICES', 'PROXIMITY', 'DELTA_LIQ'], ascending=[False, True, True, False])
            high_mean_upb = b_df['PRICES'].head(10).mean()
  
            s_df = s_df.sort_values(by=['DELTA_LIQ', 'PRICES', 'LIQUIDITY', 'PROXIMITY'], ascending=[False, True, False, True])
            low_mean_negb = s_df['PRICES'].head(10).mean()
            s_df = s_df.sort_values(by=['LIQUIDITY', 'PRICES', 'PROXIMITY', 'DELTA_LIQ'], ascending=[False, False, True, False])
            high_mean_negb = s_df['PRICES'].head(10).mean()
        except Exception as e: 
            log_traceback(e)
            b_df = {}
            s_df = {}
            upswings = 0
            downswings = 0
            low_mean_negb = 0
            low_mean_upb = 0
            high_mean_negb = 0
            high_mean_upb = 0

        return b_df, s_df, upswings, downswings, [low_mean_upb, high_mean_upb], [low_mean_negb, high_mean_negb]
 

    # Perform analyses
    #wave_phase, wave_metrics = analyze_intervals(intervals)
    key_levels = analyze_heatmaps(klines_data, heatmaps)
    
    final = {
        "PRICES": key_levels,
    }
    return final



@eel.expose
def analyze_asset(data):
    try:
        rawdata = spot.market.klines(
            symbol=SYMBOL,
            interval=m_intervals.ONE_MIN
        )
        data_o = [float(row[1]) for row in rawdata] #OPEN
        data_h = [float(row[2]) for row in rawdata] #HIGH
        data_l = [float(row[3]) for row in rawdata] #LOW
        data = [float(row[4]) for row in rawdata] #CLOSE
        data_v = [float(row[5]) for row in rawdata] #VOLUME
        data_ct = [float(row[6]) for row in rawdata] #CLOSE TIME
            
        # Create the final list of dictionaries
        finaldata = [
            {'c': close, 'h': high, 'l': low, 'o': open_, 'x': close_time}
            for open_, high, low, close, close_time in zip(data_o, data_h, data_l, data, data_ct)
        ]
        data_series = pd.Series(data)
        
        vpt = anal.calculate_vpt(data, data_v)
        obv = anal.EMA(anal.normalize_series(pd.Series(anal.calculate_obv(data, data_v))), 5).tolist()
        kci = anal.EMA(anal.combined_indicator(data_series, data_v), 5).tolist()
        #delta_obv_custom = calculate_delta_obv_custom(data_series, data_v)
        delta_obv_custom = anal.price_increase_predictor(data_series, data_v)
        
        sma = anal.SMA(data_series, 20).tolist()
        
        ema = anal.EMA(data_series, 20).tolist()
        rsi = anal.RSI(data_series, 14).tolist()
        macd, signal, histogram = anal.MACD(data_series)
        bb_sma, upper_band, lower_band = anal.BollingerBands(data_series)
        roc = anal.ROC(data_series, 12).tolist()
        k_percent, d_percent = anal.StochasticOscillator(data_series)
        
        # Find dynamic thresholds
        high_threshold, low_threshold = find_thresholds(data_series)

        
        # Add the positive and negative volume indicators to the analysis
        positive_volume, negative_volume = anal.calculate_positive_negative_volume_normalized(data, data_v)

        #analysis['Positive_Volume'] = [0] + positive_volume  # Adding a leading 0 for the initial entry
        #analysis['Negative_Volume'] = [0] + negative_volume  # Adding a leading 0 for the initial entry


        # Divide the series into intervals (e.g., intervals of 50)
        interval_size = 50
        intervals = [data_series[i:i + interval_size] for i in range(0, len(data_series), interval_size)]

        # Apply the function to each interval
        results = [find_average_moves(interval, high_threshold, low_threshold) for interval in intervals]

        # Convert results to a DataFrame for analysis
        results_df = pd.DataFrame(results, columns=['moves_to_high', 'moves_to_low'])

        # Calculate average moves
        average_moves_to_high = results_df['moves_to_high'].mean(skipna=True)
        average_moves_to_low = results_df['moves_to_low'].mean(skipna=True)

        print(f'Average moves to reach high value: {average_moves_to_high}')
        print(f'Average moves to reach low value: {average_moves_to_low}')
        

        def smooth_data(finaldata, window_size=10):
            """Smooth the closing prices using a simple moving average (SMA)."""
            closes = np.array([kline['c'] for kline in finaldata])
            smoothed_closes = np.convolve(closes, np.ones(window_size)/window_size, mode='valid')
            
            # Adjust the finaldata to include the smoothed closing prices
            for i in range(len(smoothed_closes)):
                finaldata[i + window_size - 1]['c'] = smoothed_closes[i]
            
            return finaldata

        def find_latest_trend_inversion(finaldata, window_size=10):
            def get_trend(kline):
                """ Determine the trend for a single kline. 
                Returns 1 for bullish, -1 for bearish, 0 for neutral."""
                if kline['c'] > kline['o']:
                    return 1
                elif kline['c'] < kline['o']:
                    return -1
                else:
                    return 0

            # Calculate trends for each kline
            trends = [get_trend(kline) for kline in finaldata]

            # Iterate through the klines to find the latest trend inversion
            for i in range(len(trends) - window_size - 1, 0, -1):
                current_window = trends[i:i + window_size]
                previous_window = trends[i - window_size:i]
                
                # Check for inversion (from bullish to bearish or vice versa)
                if sum(previous_window) > 0 and sum(current_window) < 0:
                    # Bullish to Bearish Inversion
                    return finaldata[i + window_size - 1], "Bearish"
                elif sum(previous_window) < 0 and sum(current_window) > 0:
                    # Bearish to Bullish Inversion
                    return finaldata[i + window_size - 1], "Bullish"
            
            return None, "No Inversion Found"


        def format_timedelta(delta):
            """Format a timedelta object for human readability."""
            days = delta.days
            hours, remainder = divmod(delta.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            time_parts = []
            if days > 0:
                time_parts.append(f"{days} day{'s' if days > 1 else ''}")
            if hours > 0:
                time_parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
            if minutes > 0:
                time_parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
            if seconds > 0:
                time_parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")

            return " ".join(time_parts) if time_parts else "0 seconds"

        def time_since_last_price(finaldata, inversion_point, tolerance = 0.001):
            """Find the time difference between the inversion point and the last occurrence of the same price."""
            target_price = inversion_point['c']
            inversion_time = inversion_point['x']
            
            for i in range(len(finaldata) - 2, -1, -1):  # Search backwards in time
                if abs(finaldata[i]['c'] - target_price) <= tolerance:
                    time_difference =  pd.to_timedelta( pd.to_datetime(inversion_time, unit='ms')  -  pd.to_datetime(finaldata[i]['x'], unit='ms'), unit="min")
                    return format_timedelta(time_difference)
            
            return None  # Price not found before the inversion point

        # Add tolerance factor
        tolerance = 0.01
        # First, smooth the data
        smoothed_data = smooth_data(finaldata, window_size=3)
        # Then, find the latest trend inversion
        latest_inversion, inversion_type = find_latest_trend_inversion(smoothed_data, window_size=3)

        # Get the latest price information
        latest_price_data = smoothed_data[-1]
        current_time = smoothed_data[-1]['x']

        analysis = pd.DataFrame({
            'SMA': sma,
            'EMA': ema,
            'RSI': rsi,
            'MACD': anal.normalize_series(macd),
            'SIGNAL': signal,
            'RAW_S': data,
            'RAW_CT': data_ct,
            'RAW_VOLUME': data_v,
            'KLINES': finaldata,
            'HISTOGRAM': histogram,
            'BB_SMA': bb_sma,
            'UPPER_BB': upper_band,
            'LOWER_BB': lower_band,
            'ROC': roc,
            'STOCHASTIC_K': k_percent,
            'STOCHASTIC_D': d_percent,
            'VPT': vpt,
            'OBV': obv,
            'KCI': kci,
            'DPV': delta_obv_custom,
            'PV':positive_volume,
            'NV':negative_volume
        })
        
        analysis = analysis.fillna(0)  # Replace NaN values with 0

        # Perform split and analyze
        results = split_and_analyze(analysis)

        noncyclical = ['SMA','EMA','MACD','SIGNAL','RAW_S','RAW_VOLUME','UPPER_BB','LOWER_BB','OBV','DPV']
        cyclical = ['RSI','HISTOGRAM','STOCHASTIC_K','STOCHASTIC_D','KCI', 'ROC']

        print("\n -END OF ANALYSIS- \n ")


        #print(f"Fibonacci retracement levels: {fib_levels}")
        future_projection_minutes = 30
        future_index = pd.date_range(start=pd.to_datetime(data_ct[-1], unit='ms'), periods=future_projection_minutes, freq='T')

        
        ra = analysis.to_dict()
        ro = {}


        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            f = open(os.path.join(dir_path, f'{SYMBOL}_last_heatmapdata.json'))
            # returns JSON object as 
            # a dictionary
            heatmapdata = json.load(f)
            iwdata = {}
            iwdata['ANALYSIS'] = results
            iwdata['HEATMAP'] = heatmapdata
            iwdata['PROJECTIONS'] = ro

            fa = identify_wave_and_predict(ra, iwdata)
            print("ENTERING FINAL PRJ \n")
            print(fa)
            print("END PRJ \n")


            
        except Exception as ex:
            log_traceback(ex)



            
        def smooth_data(y_values, window_size):
            smoothed = []
            for i in range(len(y_values)):
                window = y_values[max(i-window_size+1, 0):i+1]
                smoothed.append(sum(window) / len(window))
            return smoothed

        market_direction = results['last_1h']['EMA']['direction_of_flow'] + results['last_1h']['SMA']['direction_of_flow'] + results['last_30min']['EMA']['direction_of_flow'] + results['last_30min']['SMA']['direction_of_flow'] + results['last_15min']['EMA']['direction_of_flow'] + results['last_15min']['SMA']['direction_of_flow']
        
        # Pseudocode for the strategy
        #market_direction = sum([results[interval]['direction_of_flow'] for interval in intervals])

        fa_up_tgt = min(fa['PRICES'][4]) #upper price target
        fa_neg_tgt = max(fa['PRICES'][5]) #lower price target
        up_dist = fa_up_tgt - results['last_5min']['RAW_S']['close'] #distance to up target
        neg_dist = results['last_5min']['RAW_S']['close'] - fa_neg_tgt #distance to low target
        closest_tgt = fa_up_tgt if up_dist>neg_dist else fa_neg_tgt
        market_pull = fa['PRICES'][2] - fa['PRICES'][3]
        if market_pull > 0:
            market_pw = 'positive'
            market_lev = fa['PRICES'][2]/fa['PRICES'][2]
        else:
            market_pw = 'negative'
            market_lev = fa['PRICES'][3]/fa['PRICES'][2]

        global JUST_CLOSED
        global FLOAT_VOL
        global FLOAT_DIR
        global TGT_STOP

        print("cls")

        #if strat>0 and results['last_5min']['RAW_S']['close'] > results['last_5min']['RAW_S']['open'] and results['last_10min']['RAW_S']['open'] > results['last_5min']['RAW_S']['close']:
        print(f"2fa {fa['PRICES'][2]}")
        print(f"3fa {fa['PRICES'][3]}")
        print(f"market_direction {market_direction}")
        print(f"2hO {results['last_2h']['RAW_S']['open']}")
        print(f"1hO {results['last_1h']['RAW_S']['open']}")
        print(f"5mC {results['last_5min']['RAW_S']['close']}")
        bal=0
        bbal = ''
        


        global POSITION
        global TOTAL_CAPTURED
        global TOTAL_SHORTED

        if 'ORDER' in POSITION:
            if 'LONG' in POSITION['ORDER']:
                bal = POSITION['ORDER']['LONG']/results['last_5min']['RAW_S']['close']
                bbal = f"LONG BAL: {POSITION['ORDER']['LONG']}/{results['last_5min']['RAW_S']['close']} = {bal}"
            if 'SHORT' in POSITION['ORDER']:
                bal = results['last_5min']['RAW_S']['close']/POSITION['ORDER']['SHORT']
                bbal = f"SHORT BAL: {POSITION['ORDER']['SHORT']}/{results['last_5min']['RAW_S']['close']} = {bal}"
        print(bbal)
        

    

        try:

            dir_path = os.path.dirname(os.path.realpath(__file__))
            f = open(os.path.join(dir_path, f'{SYMBOL}_position.json'))
            # returns JSON object as 
            # a dictionary
            POSITION = json.load(f)
            
            f = open(os.path.join(dir_path, f'{SYMBOL}_score.json'))
            # returns JSON object as 
            # a dictionary
            SCORES = json.load(f)
        except Exception as ex:
            log_traceback(ex)
            print("NO POSITION OR SCORE TO USE")
            SCORES = {'SHORT':0, 'LONG':0}        
        if not JUST_CLOSED:

            ss2h = pd.Series([results['last_2h']['RAW_S']['open'], results['last_2h']['RAW_S']['max'], results['last_2h']['RAW_S']['min']]).mean()
            ss1h = pd.Series([results['last_1h']['RAW_S']['open'], results['last_1h']['RAW_S']['max'], results['last_1h']['RAW_S']['min']]).mean()
            ssMax = pd.Series([results['last_1h']['RAW_S']['max'], results['last_2h']['RAW_S']['max'], results['last_30min']['RAW_S']['max'], results['last_15min']['RAW_S']['max'], results['last_10min']['RAW_S']['max']]).mean()
            ssMin = pd.Series([results['last_1h']['RAW_S']['min'], results['last_2h']['RAW_S']['min'], results['last_30min']['RAW_S']['min'], results['last_15min']['RAW_S']['min'], results['last_10min']['RAW_S']['min']]).mean()
            #ss30m = pd.Series([results['last_1h']['RAW_S']['open'], results['last_1h']['RAW_S']['max'], results['last_1h']['RAW_S']['min']]).mean()
            ss5m = pd.Series([results['last_5min']['RAW_S']['open'], results['last_5min']['RAW_S']['max'], results['last_5min']['RAW_S']['min'],results['last_5min']['RAW_S']['close']]).mean()

            if fa['PRICES'][2]>fa['PRICES'][3] and fa['PRICES'][2]-fa['PRICES'][3]>2 :
                print("L1")
                print(f"{ss2h} > {ss5m}")
                print(f"{ss1h} > {ss5m}")
                print(f"{market_direction} < 0")
                print(f"{ssMin} > {ss5m}")
                if ss2h > ss5m  and ss1h > ss5m:
                    print("L2")
            #if len(nowLongs)>1:
            #    if (results['last_5min']['STOCHASTIC_K']['close'] < 70 and results['last_5min']['STOCHASTIC_D']['close'] < 80) or \
            #        ((results['last_5min']['STOCHASTIC_K']['close'] > results['last_5min']['STOCHASTIC_K']['open'] or results['last_5min']['STOCHASTIC_K']['direction_of_flow'] > 0) and \
            #         (results['last_5min']['STOCHASTIC_D']['close'] > results['last_5min']['STOCHASTIC_D']['open'] or results['last_5min']['STOCHASTIC_D']['direction_of_flow'] > 0) ):
                    if 'ORDER' not in POSITION and market_direction < 0:
                        TGT_STOP = min(fa['PRICES'][4]) #lower price target
                        FLOAT_VOL = fa['PRICES'][2]
                        FLOAT_DIR = 2
                        POSITION={'ORDER':{'LONG': results['last_5min']['RAW_S']['close'], 'TGT': TGT_STOP, 'DIR': 2}}
                        print(POSITION['ORDER'])
                else:
                    if 'ORDER' not in POSITION and ss5m > ssMin and market_direction < 0:
                        TGT_STOP = ssMin #lower price target
                        FLOAT_VOL = fa['PRICES'][3]
                        FLOAT_DIR = 3
                        POSITION={'ORDER':{'SHORT': results['last_5min']['RAW_S']['close'], 'TGT': TGT_STOP, 'DIR': 3}}
                        print(POSITION['ORDER'])
                    if 'ORDER' not in POSITION and ss5m < ssMin and market_direction > 0:
                        TGT_STOP = ssMax #lower price target
                        FLOAT_VOL = fa['PRICES'][3]
                        FLOAT_DIR = 3
                        POSITION={'ORDER':{'LONG': results['last_5min']['RAW_S']['close'], 'TGT': TGT_STOP, 'DIR': 2}}
                        print(POSITION['ORDER'])
                #else:
                #    print("MAYBE I SHOULDDA SHOOOORTz?")
                #    nowShorts.append("SHORT")
            #if len(LONG_CLOSE)>4 or (len(LONG_CLOSE) > 2 and (results['last_5min']['STOCHASTIC_K']['close'] < 60 and results['last_5min']['STOCHASTIC_D']['close'] < 70) or \
            #    ((results['last_5min']['STOCHASTIC_K']['close'] < results['last_5min']['STOCHASTIC_K']['open'] or results['last_5min']['STOCHASTIC_K']['direction_of_flow'] < 0) and \
            #     (results['last_5min']['STOCHASTIC_D']['close'] < results['last_5min']['STOCHASTIC_D']['open'] or results['last_5min']['STOCHASTIC_D']['direction_of_flow'] < 0) )):
            if 'ORDER' in POSITION:
                TGT_STOP = POSITION['ORDER']['TGT']
                FLOAT_VOL = POSITION['ORDER']['DIR']
                INVERSION_W = False
                if 'LONG' in POSITION['ORDER']:
                    if market_direction < 10:
                        INVERSION_W = True
                if 'SHORT' in POSITION['ORDER']:
                    if market_direction > -10:
                        INVERSION_W = True
                if abs(fa['PRICES'][2]-fa['PRICES'][3]) < 1 or bal < 0.9931 or (bal < 0.9975 and INVERSION_W) or bal>1.005:
                    if 'ORDER' in POSITION:
                        if 'LONG' in POSITION['ORDER']:
                            FLOAT_VOL = 0
                            FLOAT_DIR = 0
                            print("\n  POSITION LONG \n")
                            print(POSITION['ORDER'])
                            print("\n CLOSED @ \n")
                            print(results['last_5min']['RAW_S']['close'])
                            LONG_CLOSE = []
                            TOTAL_CAPTURED = (results['last_5min']['RAW_S']['close']-POSITION['ORDER']['LONG']) + TOTAL_CAPTURED
                            position_bal = "profit" if results['last_5min']['RAW_S']['close']-POSITION['ORDER']['LONG']>0 else "unprofitable"
                            POSITION={}
                            MB=[]
                            CLOSED_MSG = f"You just closed a {position_bal} LONG position. Consider WAIT?"
                            JUST_CLOSED=True
                #if strat<1:
                #    if 'ORDER' in POSITION:
                #        if 'LONG' in POSITION['ORDER']:
                #            print("\n POSITION LONG \n")
                #            print(POSITION['ORDER'])
                #            print("\n CLOSED @ \n")
                #            print(results['last_5min']['RAW_S']['close'])

                #            TOTAL_CAPTURED = (results['last_5min']['RAW_S']['close']-POSITION['ORDER']['LONG']) + TOTAL_CAPTURED
                #            POSITION={}
                #            MB=[]
                #if strat>-5:
                #    if 'ORDER' in POSITION:
                #        if 'SHORT' in POSITION['ORDER'] and results['last_5min']['RAW_S']['close'] < POSITION['ORDER']['SHORT']:
                #            print("\n POSITION SHORT \n")
                #            print(POSITION['ORDER'])
                #            print("\n CLOSED @ \n")
                #            print(results['last_5min']['RAW_S']['close'])
                #            TOTAL_SHORTED = (POSITION['ORDER']['SHORT']-results['last_5min']['RAW_S']['close']) + TOTAL_SHORTED
                #            POSITION={}
                #            MB=[]
                #if len(SHORT_CLOSE)>1 or (len(SHORT_CLOSE)>0 and (results['last_5min']['STOCHASTIC_K']['close'] > 15 and results['last_5min']['STOCHASTIC_D']['close'] > 10) or \
                #    ((results['last_5min']['STOCHASTIC_K']['close'] > results['last_5min']['STOCHASTIC_K']['open'] or results['last_5min']['STOCHASTIC_K']['direction_of_flow'] > 0) and \
                #     (results['last_5min']['STOCHASTIC_D']['close'] > results['last_5min']['STOCHASTIC_D']['open'] or results['last_5min']['STOCHASTIC_D']['direction_of_flow'] > 0) )):
                    if 'ORDER' in POSITION:
                        if 'SHORT' in POSITION['ORDER']:
                            FLOAT_VOL = 0
                            
                            print("\n  POSITION SHORT \n")
                            print(POSITION['ORDER'])
                            print("\n CLOSED @ \n")
                            print(results['last_5min']['RAW_S']['close'])
                            JUST_CLOSED=True
                            position_bal = "profitable" if POSITION['ORDER']['SHORT'] - results['last_5min']['RAW_S']['close'] > 0 else "unprofitable"
                            CLOSED_MSG = f"You just closed a {position_bal} SHORT position. Consider WAIT?"

                            SHORT_CLOSE=[]
                            TOTAL_SHORTED = (POSITION['ORDER']['SHORT']-results['last_5min']['RAW_S']['close']) + TOTAL_SHORTED
                            POSITION={}
                            MB=[]

                        ### U REALLY SURE ABT THE WHOLE LOTTA <20 and <10?? Maybe smth diffrnt?
            #if len(nowShorts)>0 and ((results['last_5min']['STOCHASTIC_K']['close'] < 60 and results['last_5min']['STOCHASTIC_D']['close'] < 30) or \
            #    ((results['last_5min']['STOCHASTIC_K']['close'] < results['last_5min']['STOCHASTIC_K']['open'] or results['last_5min']['STOCHASTIC_K']['direction_of_flow'] < 0) and \
            #     (results['last_5min']['STOCHASTIC_D']['close'] < results['last_5min']['STOCHASTIC_D']['open'] or results['last_5min']['STOCHASTIC_D']['direction_of_flow'] < 0) )):

            if fa['PRICES'][2]<fa['PRICES'][3] and fa['PRICES'][3]-fa['PRICES'][2]>2:
                print("S1")
                print(f"{ss2h} < {ss5m}")
                print(f"{ss1h} < {ss5m}")
                print(f"{market_direction} > 0")
                print(f"{ssMax} < {ss5m}")
                if ss2h < ss5m  and ss1h < ss5m:
                    if 'ORDER' not in POSITION and fa['PRICES'][3]-fa['PRICES'][2]>3 and market_direction > 0:
                        TGT_STOP = max(fa['PRICES'][5]) #lower price target
                        FLOAT_VOL = fa['PRICES'][3]
                        FLOAT_DIR = 3
                        POSITION={'ORDER':{'SHORT': results['last_5min']['RAW_S']['close'], 'TGT': TGT_STOP, 'DIR': 3}}
                        print(POSITION['ORDER'])
                else:
                    if 'ORDER' not in POSITION and ss5m > ssMax and market_direction < 0:
                        TGT_STOP = ssMin #lower price target
                        FLOAT_VOL = fa['PRICES'][2]
                        FLOAT_DIR = 2
                        POSITION={'ORDER':{'SHORT': results['last_5min']['RAW_S']['close'], 'TGT': TGT_STOP, 'DIR': 2}}
                        print(POSITION['ORDER'])
                    if 'ORDER' not in POSITION and ss5m < ssMax and market_direction > 0:
                        TGT_STOP = ssMax #lower price target
                        FLOAT_VOL = fa['PRICES'][2]
                        FLOAT_DIR = 2
                        POSITION={'ORDER':{'LONG': results['last_5min']['RAW_S']['close'], 'TGT': TGT_STOP, 'DIR': 2}}
                        print(POSITION['ORDER'])

        else:
            JUST_CLOSED = False
            print("JUST CLOSED")
        print(f"TOTAL LONG {SCORES['LONG']}")            
        print(f"TOTAL SHORT {SCORES['SHORT']}")
        if TOTAL_CAPTURED > 0:
            SCORES['LONG']+=TOTAL_CAPTURED
            TOTAL_CAPTURED = 0
        if TOTAL_SHORTED > 0:
            SCORES['SHORT']+=TOTAL_SHORTED
            TOTAL_SHORTED = 0

        print(SYMBOL)
        ra['POS'] = POSITION
        ra['TOTAL_SHORT'] = SCORES['SHORT']
        ra['TOTAL_LONG'] = SCORES['LONG']

        global _LAST_DATA
        global DIR_PATH

        #dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(DIR_PATH, f'{SYMBOL}_position.json'), 'w') as fp:
            json.dump(POSITION, fp)

        with open(os.path.join(DIR_PATH, f'{SYMBOL}_score.json'), 'w') as fp:
            json.dump(SCORES, fp)

        with open(os.path.join(DIR_PATH, f'{SYMBOL}_last_data.json'), 'w') as fp:
            json.dump(ra, fp)

        _LAST_DATA = ra


        return ra
    except Exception as ex:
        print('Eror')
        log_traceback(ex)

if __name__ == "__main__":

    
    price_monitor = PriceMonitor(SYMBOL, spot)
    price_monitor.start()

    liquidity_monitor = LiquidityMonitor(SYMBOL)
    liquidity_monitor.start()


    @eel.expose
    def get_heatmap():
        return liquidity_monitor.get_map()
    
    #eel.init('web')  # Initialize your eel web folder

    #eel.start('klines_btc.html', block=False, size=(1920, 1080))  # Start your eel application without blocking
    print(f"Init overrided! Headless mode init for {SYMBOL}")
    analyze_asset(1)
    try:
        while True:
            print(f"Current price: {price_monitor.get_price()}")
            #print(f"Current heatmap: {liquidity_monitor.get_map()}")

            #eel.update_heatmap(liquidity_monitor.get_map())
            #eel.update_chart(price_monitor.get_price())
            print("NOW analysis..")
            analyze_asset(1)
            eel.sleep(60)  # Keep the main loop alive
    except KeyboardInterrupt:
        print("Stopping price monitor...")
        price_monitor.stop()
        #print("Price monitor stopped.")